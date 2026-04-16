# baselines.py — Baseline federated aggregation strategies for empirical comparison.
#
# These baselines are run against the same attack scenarios as the proposed
# RobustFLIDSStrategy to prove the novel defense is superior.
# Each baseline is implemented as a Flower Strategy subclass.
#
# ═══════════════════════════════════════════════════════════════════
# BASELINE 1: Standard FedAvg
# ═══════════════════════════════════════════════════════════════════
#   - Blindly averages ALL client weight updates with uniform weights (1/K each)
#   - Expected to collapse instantly under label-flipping and backdoor attacks
#   - Demonstrates the unprotected baseline vulnerability
#   - Reference: McMahan et al. (2017) "Communication-Efficient Learning of
#     Deep Networks from Decentralized Data"
#
#   class FedAvgBaseline(flwr.server.strategy.Strategy): ...
#
# ═══════════════════════════════════════════════════════════════════
# BASELINE 2: Federated Trimmed Mean
# ═══════════════════════════════════════════════════════════════════
#   - For each parameter dimension independently:
#       sort client values, discard top and bottom beta=20% fractions, average rest
#   - Resilient to basic gradient explosion / magnitude spikes
#   - Expected to fail against stealthy norm-bounded backdoor attacks where
#     the malicious gradient magnitude is engineered within the accepted range
#
#   class FedTrimmedMeanBaseline(flwr.server.strategy.Strategy):
#       def __init__(self, beta: float = 0.2): ...
#
# ═══════════════════════════════════════════════════════════════════
# BASELINE 3: Krum / Multi-Krum
# ═══════════════════════════════════════════════════════════════════
#   - Selects the single client update (or top-m updates) minimizing the sum
#     of squared Euclidean distances to its (n - f - 2) nearest neighbours
#   - Byzantine-robust but O(K² · d) complexity — unscalable for large IoT fleets
#   - Expected to fail on Non-IID data by flagging diverse-but-benign clients
#     as outliers, destroying localized edge-case intrusion intelligence
#   - Reference: Blanchard et al. (2017) "Machine Learning with Adversaries:
#     Byzantine Tolerant Gradient Descent"
#
#   class KrumBaseline(flwr.server.strategy.Strategy):
#       def __init__(self, num_byzantine: int, multi_k: int = 1): ...
#
# ═══════════════════════════════════════════════════════════════════
# SHARED HELPER:
# ═══════════════════════════════════════════════════════════════════
#   def get_baseline_strategy(name: str, config: dict) -> flwr.server.strategy.Strategy:
#       """Factory: returns instantiated baseline by name string from config."""
