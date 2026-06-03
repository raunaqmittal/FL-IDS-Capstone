import numpy as np
import flwr as fl
from flwr.common import Parameters, FitRes, EvaluateRes, ndarrays_to_parameters, parameters_to_ndarrays
from typing import List, Tuple, Optional, Union

from src.configs.config import CONFIG
from src.components.server.server import get_initial_parameters


class FedAvgBaseline(fl.server.strategy.Strategy):
    def __init__(self, initial_parameters: Parameters):
        super().__init__()
        self.initial_parameters = initial_parameters

    def initialize_parameters(self, client_manager):
        return self.initial_parameters

    def configure_fit(self, server_round, parameters, client_manager):
        return []

    def configure_evaluate(self, server_round, parameters, client_manager):
        return []

    def aggregate_fit(self, server_round, results: List[Tuple], failures):
        if not results:
            return None, {}

        all_params = [parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results]
        num_examples = [fit_res.num_examples for _, fit_res in results]
        total = sum(num_examples)

        aggregated = [
            sum(p[i] * n for p, n in zip(all_params, num_examples)) / total
            for i in range(len(all_params[0]))
        ]
        return ndarrays_to_parameters(aggregated), {}

    def aggregate_evaluate(self, server_round, results, failures):
        return None, {}

    def evaluate(self, server_round, parameters):
        return None


class FedTrimmedMeanBaseline(fl.server.strategy.Strategy):
    def __init__(self, initial_parameters: Parameters, beta: float = 0.2):
        super().__init__()
        self.initial_parameters = initial_parameters
        self.beta = beta

    def initialize_parameters(self, client_manager):
        return self.initial_parameters

    def configure_fit(self, server_round, parameters, client_manager):
        return []

    def configure_evaluate(self, server_round, parameters, client_manager):
        return []

    def aggregate_fit(self, server_round, results: List[Tuple], failures):
        if not results:
            return None, {}

        all_params = [parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results]
        K = len(all_params)
        trim = max(1, int(self.beta * K))

        aggregated = []
        for i in range(len(all_params[0])):
            stacked = np.stack([p[i].flatten() for p in all_params], axis=0)
            stacked = np.sort(stacked, axis=0)
            trimmed = stacked[trim: K - trim]
            mean = trimmed.mean(axis=0)
            aggregated.append(mean.reshape(all_params[0][i].shape))

        return ndarrays_to_parameters(aggregated), {}

    def aggregate_evaluate(self, server_round, results, failures):
        return None, {}

    def evaluate(self, server_round, parameters):
        return None


class KrumBaseline(fl.server.strategy.Strategy):
    def __init__(self, initial_parameters: Parameters, num_byzantine: int, multi_k: int = 1):
        super().__init__()
        self.initial_parameters = initial_parameters
        self.num_byzantine = num_byzantine
        self.multi_k = multi_k

    def initialize_parameters(self, client_manager):
        return self.initial_parameters

    def configure_fit(self, server_round, parameters, client_manager):
        return []

    def configure_evaluate(self, server_round, parameters, client_manager):
        return []

    def aggregate_fit(self, server_round, results: List[Tuple], failures):
        if not results:
            return None, {}

        all_params = [parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results]
        K = len(all_params)
        n_select = K - self.num_byzantine - 2

        flat = [np.concatenate([p.flatten() for p in params]) for params in all_params]

        scores = np.zeros(K)
        for i in range(K):
            dists = sorted(np.sum((flat[i] - flat[j]) ** 2) for j in range(K) if j != i)
            scores[i] = sum(dists[:n_select])

        top_k = np.argsort(scores)[: self.multi_k]
        selected = [all_params[i] for i in top_k]

        aggregated = [
            np.mean([p[i] for p in selected], axis=0)
            for i in range(len(selected[0]))
        ]
        return ndarrays_to_parameters(aggregated), {}

    def aggregate_evaluate(self, server_round, results, failures):
        return None, {}

    def evaluate(self, server_round, parameters):
        return None


def get_baseline_strategy(name: str) -> fl.server.strategy.Strategy:
    initial_parameters = get_initial_parameters()
    fed_cfg = CONFIG["federated"]
    defense_cfg = CONFIG["defense"]

    if name == "fedavg":
        return FedAvgBaseline(initial_parameters)
    elif name == "trimmed_mean":
        return FedTrimmedMeanBaseline(initial_parameters, beta=0.2)
    elif name == "krum":
        num_byzantine = int(defense_cfg["max_byzantine_fraction"] * fed_cfg["clients_per_round"])
        return KrumBaseline(initial_parameters, num_byzantine=num_byzantine, multi_k=1)
    else:
        raise ValueError(f"Unknown baseline: {name}")
