import random
import math

from src.configs.config import CONFIG
from src.logging.logger import logging


def select_malicious_clients(num_clients: int, attacker_ratio: float, seed: int = 42) -> list:
    num_attackers = int(math.floor(num_clients * attacker_ratio))
    rng = random.Random(seed)
    return sorted(rng.sample(range(num_clients), num_attackers))


def is_attack_active(server_round: int, attack_start_round: int) -> bool:
    return server_round >= attack_start_round


def get_attack_config(client_id: int, malicious_ids: list, server_round: int) -> dict:
    attack_cfg = CONFIG["attack"]

    if client_id not in malicious_ids:
        return {"is_poisoned": False}

    if not is_attack_active(server_round, attack_cfg["attack_start_round"]):
        return {"is_poisoned": False}

    return {
        "is_poisoned": True,
        "attack_type": attack_cfg["attack_type"],
        "attack_start_round": attack_cfg["attack_start_round"],
        "source_class": attack_cfg["source_class"],
        "target_class": attack_cfg["target_class"],
        "trigger_feature_idx": attack_cfg["trigger_feature_idx"],
        "trigger_values": attack_cfg["trigger_values"],
        "inject_ratio": attack_cfg["inject_ratio"],
        "scale_to_benign_norm": attack_cfg["scale_to_benign_norm"],
    }


def run_attack_sweep(strategy_name: str = "robust") -> None:
    from src.pipelines.training_pipeline import run_experiment

    ratios = CONFIG["experiment"]["attacker_ratios"]
    for ratio in ratios:
        logging.info(f"\n[AttackSweep] strategy={strategy_name} attacker_ratio={ratio}")
        CONFIG["attack"]["attacker_ratio"] = ratio
        suffix = f"_{strategy_name}_ratio_{int(ratio * 100):02d}pct"
        run_experiment(results_suffix=suffix, strategy_name=strategy_name)

    logging.info("[AttackSweep] All sweeps complete.")


if __name__ == "__main__":
    run_attack_sweep()
