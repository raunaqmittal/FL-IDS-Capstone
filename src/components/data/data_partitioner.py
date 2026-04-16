# data_partitioner.py — Dirichlet-based Non-IID data partitioning for CIC-IDS2017.
#
# Splits the preprocessed CIC-IDS2017 dataset across N simulated IoT clients
# in a Non-IID fashion, replicating realistic heterogeneous IoT traffic environments.
#
# ═══════════════════════════════════════════════════════
# PREPROCESSING STEPS (before partitioning):
# ═══════════════════════════════════════════════════════
#   1. Load raw CIC-IDS2017 CSV files (all days)
#   2. Drop constant / near-zero-variance features (Variance Filtering)
#   3. Drop highly correlated feature pairs (Pearson Correlation |r| > 0.95)
#      → reduces ~78 features to ~40–50 discriminative features
#   4. Apply Z-score standardization (StandardScaler) for gradient stability
#   5. Handle class imbalance — log class distribution, apply SMOTE or
#      class-weighted loss (decision driven by config.yaml)
#   6. Encode categorical labels to integer indices
#
# ═══════════════════════════════════════════════════════

import sys
import os
import numpy as np
from typing import List, Tuple
from src.logging.logger import logging
from src.exception.exception import FLIDSException
from src.configs.config import CONFIG


def partition_iid(
    X: np.ndarray, y: np.ndarray, num_clients: int
) -> List[Tuple[np.ndarray, np.ndarray]]:
    try:
        indices = np.random.permutation(len(X))
        splits = np.array_split(indices, num_clients)
        return [(X[s], y[s]) for s in splits]
    except Exception as e:
        raise FLIDSException(e, sys)


def partition_non_iid(
    X: np.ndarray, y: np.ndarray, num_clients: int, alpha: float = 0.5
) -> List[Tuple[np.ndarray, np.ndarray]]:
    try:
        classes = np.unique(y)
        client_indices = [[] for _ in range(num_clients)]

        for c in classes:
            class_idx = np.where(y == c)[0]
            np.random.shuffle(class_idx)
            proportions = np.random.dirichlet([alpha] * num_clients)
            proportions = (proportions * len(class_idx)).astype(int)
            proportions[-1] = len(class_idx) - proportions[:-1].sum()
            splits = np.split(class_idx, np.cumsum(proportions)[:-1])
            for i, split in enumerate(splits):
                client_indices[i].extend(split.tolist())

        partitions = []
        for idx in client_indices:
            idx = np.array(idx)
            np.random.shuffle(idx)
            partitions.append((X[idx], y[idx]))

        logging.info(f"Non-IID partitioning done: {num_clients} clients, alpha={alpha}")
        return partitions
    except Exception as e:
        raise FLIDSException(e, sys)


def save_partitions(partitions: List[Tuple[np.ndarray, np.ndarray]], output_dir: str) -> None:
    try:
        os.makedirs(output_dir, exist_ok=True)
        for i, (X, y) in enumerate(partitions):
            path = os.path.join(output_dir, f"client_{i:04d}.npz")
            np.savez_compressed(path, X=X, y=y)
        logging.info(f"Saved {len(partitions)} partitions to {output_dir}")
    except Exception as e:
        raise FLIDSException(e, sys)


def load_partition(client_id: int, data_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    try:
        path = os.path.join(data_dir, f"client_{client_id:04d}.npz")
        data = np.load(path)
        return data["X"], data["y"]
    except Exception as e:
        raise FLIDSException(e, sys)


def run_partitioning(X: np.ndarray, y: np.ndarray) -> None:
    try:
        cfg_data = CONFIG["data"]
        cfg_fl = CONFIG["federated"]

        num_clients = cfg_fl["num_clients"]
        mode = cfg_data["partition_mode"]
        alpha = cfg_data["alpha_dirichlet"]
        seed = cfg_data["random_seed"]
        output_dir = os.path.join("artifacts", "data")

        np.random.seed(seed)

        if mode == "iid":
            partitions = partition_iid(X, y, num_clients)
        else:
            partitions = partition_non_iid(X, y, num_clients, alpha)

        save_partitions(partitions, output_dir)
    except Exception as e:
        raise FLIDSException(e, sys)


# CONFIG KEYS USED (from config.yaml):
#   num_clients, alpha_dirichlet, partition_mode (iid | non_iid),
#   val_split_ratio, random_seed
