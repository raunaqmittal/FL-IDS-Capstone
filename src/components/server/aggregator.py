import sys
from typing import Dict, List, Optional, Tuple, Union

import flwr as fl
import numpy as np
from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
from scipy.spatial.distance import pdist, squareform

from src.configs.config import CONFIG
from src.logging.logger import logging
from src.exception.exception import FLIDSException


def extract_final_layer(ndarrays: List[np.ndarray]) -> np.ndarray:
    # The MLP parameter list order from state_dict is:
    #   [layer0.weight, layer0.bias, layer0.bn.weight, layer0.bn.bias, layer0.bn.rm, layer0.bn.rv,
    #    layer1.weight, ..., output.weight, output.bias]
    # The final classification layer weight is the second-to-last ndarray (output.weight).
    # We take index -2 (weight) and flatten to 1D.
    weight = ndarrays[-2]
    return weight.flatten()


def compute_layer_wise_cosine_similarity(final_layers: np.ndarray) -> np.ndarray:
    # final_layers: (K, D) matrix — one flattened final-layer per client
    # Returns: (K, K) cosine similarity matrix
    distances = squareform(pdist(final_layers, metric="cosine"))
    return 1.0 - distances


def compute_mad_scores(sim_matrix: np.ndarray) -> np.ndarray:
    # Per-client consensus = median similarity to all other clients
    consensus = np.median(sim_matrix, axis=1)
    med = np.median(consensus)
    mad = np.median(np.abs(consensus - med))
    scores = 0.6745 * (consensus - med) / (mad + 1e-9)
    return scores


def temperature_scaled_softmax(scores: np.ndarray, temperature: float) -> np.ndarray:
    scaled = scores * temperature
    shifted = scaled - np.max(scaled)
    exp_vals = np.exp(shifted)
    return exp_vals / (exp_vals.sum() + 1e-9)


def project_capped_simplex(v: np.ndarray, cap_t: float) -> np.ndarray:
    # Project v onto the capped unit simplex: sum(w)=1, 0<=w_i<=cap_t.
    # Uses binary search on the Lagrange multiplier gamma.
    def feasibility(gamma):
        return np.clip(v - gamma, 0.0, cap_t).sum() - 1.0

    lo, hi = v.min() - 1.0, v.max()
    for _ in range(64):
        mid = (lo + hi) / 2.0
        if feasibility(mid) > 0:
            lo = mid
        else:
            hi = mid

    weights = np.clip(v - hi, 0.0, cap_t)
    total = weights.sum()
    if total > 1e-9:
        weights = weights / total

    return weights



class RobustFLIDSStrategy(fl.server.strategy.Strategy):
    """
    Variant A — AL-CMT: Adaptive Layer-Wise Cosine-MAD Trust aggregation.

    Defense pipeline per round:
      1. Deserialize client updates → NumPy ndarrays
      2. Extract final classification layer from each client
      3. Compute pairwise cosine similarity → MAD robust Z-scores
      4. Update EMA reputation scores
      5. Temperature-scaled softmax → trust weights
      6. Capped simplex projection → final aggregation weights
      7. Weighted average across ALL layers → global model
    """

    def __init__(self, initial_parameters: Parameters):
        super().__init__()
        self.initial_parameters = initial_parameters

        defense = CONFIG["defense"]
        federated = CONFIG["federated"]

        self.ema_momentum: float = defense["ema_momentum"]
        self.temperature: float = defense["temperature"]
        self.mad_threshold: float = defense["mad_threshold"]
        self.initial_reputation: float = defense["initial_reputation"]
        self.max_byzantine_fraction: float = defense["max_byzantine_fraction"]
        self.clients_per_round: int = federated["clients_per_round"]
        self.num_rounds: int = federated["num_rounds"]
        self.local_epochs: int = federated["local_epochs"]
        self.lr: float = federated["learning_rate"]
        self.batch_size: int = federated["batch_size"]

        # cap_t bounds max influence of any single client
        effective_K = self.clients_per_round
        b_f = int(self.max_byzantine_fraction * effective_K)
        self.cap_t: float = 1.0 / max(effective_K - b_f, 1)

        # Persistent state across rounds
        self.reputation_scores: Dict[str, float] = {}

        logging.info(
            f"[Aggregator] AL-CMT initialized — cap_t={self.cap_t:.4f}, "
            f"temperature={self.temperature}, ema_momentum={self.ema_momentum}"
        )

    def _update_ema_reputation(
        self, client_ids: List[str], mad_scores: np.ndarray
    ) -> np.ndarray:
        mu = self.ema_momentum
        tau = self.mad_threshold

        for i, cid in enumerate(client_ids):
            if cid not in self.reputation_scores:
                self.reputation_scores[cid] = self.initial_reputation

            reward = mad_scores[i] if mad_scores[i] >= tau else tau
            self.reputation_scores[cid] = (
                mu * self.reputation_scores[cid] + (1 - mu) * reward
            )

        return np.array([self.reputation_scores[cid] for cid in client_ids])

    def initialize_parameters(
        self, client_manager: fl.server.client_manager.ClientManager
    ) -> Optional[Parameters]:
        return self.initial_parameters

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: fl.server.client_manager.ClientManager,
    ) -> List[Tuple[ClientProxy, fl.common.FitIns]]:
        config = {
            "server_round": server_round,
            "local_epochs": self.local_epochs,
            "lr": self.lr,
            "batch_size": self.batch_size,
        }
        fit_ins = fl.common.FitIns(parameters, config)
        clients = client_manager.sample(
            num_clients=self.clients_per_round, min_num_clients=self.clients_per_round
        )
        return [(client, fit_ins) for client in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        try:
            if not results:
                logging.warning(f"[Aggregator] Round {server_round}: no results received.")
                return None, {}

            if failures:
                logging.warning(f"[Aggregator] Round {server_round}: {len(failures)} client(s) failed.")

            client_ids = [str(proxy.cid) for proxy, _ in results]
            all_ndarrays = [parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results]
            num_examples = [fit_res.num_examples for _, fit_res in results]

            # Step 2: extract final layer per client
            final_layers = np.stack([extract_final_layer(nd) for nd in all_ndarrays])

            # Step 3: cosine similarity → MAD Z-scores
            sim_matrix = compute_layer_wise_cosine_similarity(final_layers)
            mad_scores = compute_mad_scores(sim_matrix)

            n_flagged = int((mad_scores < self.mad_threshold).sum())
            logging.info(
                f"[Aggregator] Round {server_round}: {n_flagged}/{len(client_ids)} clients flagged by MAD."
            )

            # Step 4: EMA reputation update
            reputation = self._update_ema_reputation(client_ids, mad_scores)

            # Step 5: temperature-scaled softmax
            trust_weights = temperature_scaled_softmax(reputation, self.temperature)

            # Step 6: capped simplex projection
            final_weights = project_capped_simplex(trust_weights, self.cap_t)

            logging.info(
                f"[Aggregator] Round {server_round}: trust_weights min={final_weights.min():.4f} "
                f"max={final_weights.max():.4f} zero_count={(final_weights == 0).sum()}"
            )

            # Step 7: weighted aggregation across ALL layers
            global_params = [
                np.average(
                    np.stack([nd[layer_idx] for nd in all_ndarrays]),
                    axis=0,
                    weights=final_weights,
                )
                for layer_idx in range(len(all_ndarrays[0]))
            ]

            aggregated_parameters = ndarrays_to_parameters(global_params)

            metrics = {
                "round": server_round,
                "clients": len(client_ids),
                "flagged": n_flagged,
                "min_trust": float(final_weights.min()),
                "max_trust": float(final_weights.max()),
            }

            return aggregated_parameters, metrics

        except Exception as e:
            raise FLIDSException(e, sys)

    def configure_evaluate(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: fl.server.client_manager.ClientManager,
    ) -> List[Tuple[ClientProxy, fl.common.EvaluateIns]]:
        evaluate_ins = fl.common.EvaluateIns(parameters, {})
        clients = client_manager.sample(
            num_clients=self.clients_per_round, min_num_clients=1
        )
        return [(client, evaluate_ins) for client in clients]

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, fl.common.EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, fl.common.EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        if not results:
            return None, {}

        total_examples = sum(r.num_examples for _, r in results)
        weighted_loss = sum(r.loss * r.num_examples for _, r in results) / total_examples

        return weighted_loss, {"round": server_round}

    def evaluate(
        self,
        server_round: int,
        parameters: Parameters,
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        return None
