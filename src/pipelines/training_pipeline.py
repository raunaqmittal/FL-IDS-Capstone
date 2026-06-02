import sys
import math

import flwr as fl
import numpy as np
import torch
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays

from src.configs.config import CONFIG
from src.configs.paths import DATA_DIR, MODELS_DIR, PREPROCESSED_DIR, ensure_dirs
from src.logging.logger import logging
from src.exception.exception import FLIDSException
from src.components.model.model import MLPClassifier, get_model_parameters, set_model_parameters
from src.components.data.data_partitioner import load_partition
from src.components.data.torch_dataset import make_dataloader
from src.components.client.client import FLIDSClient
from src.components.server.aggregator import RobustFLIDSStrategy
from src.components.server.server import server_evaluate_fn


def _select_malicious_ids(num_clients: int, attacker_ratio: float, seed: int = 42) -> set:
    num_attackers = int(math.floor(num_clients * attacker_ratio))
    rng = np.random.RandomState(seed)
    return set(rng.choice(num_clients, size=num_attackers, replace=False).tolist())


def run_experiment() -> None:
    try:
        ensure_dirs()

        model_cfg = CONFIG["model"]
        fed_cfg = CONFIG["federated"]
        attack_cfg = CONFIG["attack"]

        num_clients = fed_cfg["num_clients"]
        num_rounds = fed_cfg["num_rounds"]
        batch_size = fed_cfg["batch_size"]

        attacker_ratio = float(attack_cfg["attacker_ratio"])
        malicious_ids = _select_malicious_ids(num_clients, attacker_ratio)

        if malicious_ids:
            logging.info(f"[Pipeline] Malicious client IDs: {sorted(malicious_ids)}")
        else:
            logging.info("[Pipeline] No malicious clients this run (attacker_ratio=0).")

        # Build initial model
        model = MLPClassifier(
            input_dim=model_cfg["input_dim"],
            hidden_dims=model_cfg["hidden_dims"],
            num_classes=model_cfg["num_classes"],
            dropout_rate=model_cfg["dropout_rate"],
        )

        checkpoint = MODELS_DIR / "baseline_mlp.pth"
        if checkpoint.exists():
            model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
            logging.info("[Pipeline] Warm-starting from baseline checkpoint.")

        initial_parameters = ndarrays_to_parameters(get_model_parameters(model))

        # Client factory for Flower simulation
        def client_fn(cid: str) -> fl.client.NumPyClient:
            cid_int = int(cid)
            X_train, y_train, X_val, y_val = load_partition(cid_int)

            train_loader = make_dataloader(X_train, y_train, batch_size=batch_size, shuffle=True)
            val_loader = make_dataloader(X_val, y_val, batch_size=batch_size, shuffle=False)

            client_model = MLPClassifier(
                input_dim=model_cfg["input_dim"],
                hidden_dims=model_cfg["hidden_dims"],
                num_classes=model_cfg["num_classes"],
                dropout_rate=model_cfg["dropout_rate"],
            )

            is_poisoned = cid_int in malicious_ids

            client_config = {
                "device": "cuda" if torch.cuda.is_available() else "cpu",
                "local_epochs": fed_cfg["local_epochs"],
                "lr": fed_cfg["learning_rate"],
                "weight_decay": 1e-5,
                "is_poisoned": is_poisoned,
                "attack_type": attack_cfg["attack_type"],
                "attack_start_round": attack_cfg["attack_start_round"],
                "source_class": attack_cfg["source_class"],
                "target_class": attack_cfg["target_class"],
                "trigger_feature_idx": attack_cfg["trigger_feature_idx"],
                "trigger_values": attack_cfg["trigger_values"],
                "inject_ratio": attack_cfg["inject_ratio"],
                "scale_to_benign_norm": attack_cfg["scale_to_benign_norm"],
                "benign_norm_target": 1.0,
            }

            return FLIDSClient(
                cid=cid,
                train_loader=train_loader,
                val_loader=val_loader,
                model=client_model,
                config=client_config,
                X_train_raw=X_train if is_poisoned else None,
                y_train_raw=y_train if is_poisoned else None,
            )

        strategy = RobustFLIDSStrategy(initial_parameters=initial_parameters)

        logging.info(
            f"[Pipeline] Starting FL simulation — {num_clients} clients, "
            f"{num_rounds} rounds, {len(malicious_ids)} attackers."
        )

        fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=num_clients,
            config=fl.server.ServerConfig(num_rounds=num_rounds),
            strategy=strategy,
            client_resources={"num_cpus": 1, "num_gpus": 0.0},
        )

        # Save final global model
        final_params = parameters_to_ndarrays(strategy.initial_parameters)
        final_model = MLPClassifier(
            input_dim=model_cfg["input_dim"],
            hidden_dims=model_cfg["hidden_dims"],
            num_classes=model_cfg["num_classes"],
            dropout_rate=model_cfg["dropout_rate"],
        )
        set_model_parameters(final_model, final_params)
        save_path = MODELS_DIR / "fl_global_model.pth"
        torch.save(final_model.state_dict(), save_path)
        logging.info(f"[Pipeline] Final global model saved to {save_path}")

    except Exception as e:
        raise FLIDSException(e, sys)


if __name__ == "__main__":
    run_experiment()
