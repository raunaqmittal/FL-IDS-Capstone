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
# PARTITIONING MODES:
# ═══════════════════════════════════════════════════════
#   IID mode:
#     - Globally shuffle dataset, split evenly across N clients
#     - Each client gets identical class distribution
#
#   Non-IID mode (primary — required for realistic IoT simulation):
#     - Use Dirichlet Distribution with concentration parameter α = 0.5
#     - Low α forces highly unequal class distributions per client:
#         some clients see almost entirely DoS,
#         others see mostly Botnet or pure Benign traffic
#     - Simulates physically isolated IoT subnetworks
#
# ═══════════════════════════════════════════════════════
# KEY FUNCTIONS TO IMPLEMENT:
# ═══════════════════════════════════════════════════════
#
#   def load_and_preprocess(raw_data_path: str, config: dict) -> Tuple[np.ndarray, np.ndarray]:
#       """Load CSVs, clean, filter, scale, and encode CIC-IDS2017 data."""
#
#   def partition_iid(X, y, num_clients: int) -> List[Tuple[np.ndarray, np.ndarray]]:
#       """Evenly distribute shuffled data across num_clients."""
#
#   def partition_non_iid(
#       X, y, num_clients: int, alpha: float = 0.5
#   ) -> List[Tuple[np.ndarray, np.ndarray]]:
#       """Dirichlet(alpha)-based Non-IID partitioning.
#          Returns list of (X_client, y_client) tuples — one per client."""
#
#   def save_partitions(partitions, output_dir: str) -> None:
#       """Serialize each client partition as a .npz file to artifacts/data/
#          named client_0000.npz ... client_0049.npz"""
#
#   def load_partition(client_id: int, data_dir: str) -> Tuple[np.ndarray, np.ndarray]:
#       """Load a single client's .npz partition at runtime during FL training."""
#
# CONFIG KEYS USED (from config.yaml):
#   num_clients, alpha_dirichlet, partition_mode (iid | non_iid),
#   val_split_ratio, random_seed
