# attacker.py — Byzantine adversary simulation for FL-IDS threat modeling.
#
# Simulates compromised IoT Edge Gateway clients executing two attack types:
#
# ═══════════════════════════════════════════════════════
# ATTACK 1: Targeted Semantic Label-Flipping
# ═══════════════════════════════════════════════════════
#   - NOT random label noise (FedAvg naturally averages that out)
#   - Selectively flips ONLY "DDoS" → "Benign" labels (targeted class flip)
#   - All other attack classes (PortScan, Infiltration, etc.) left untouched
#   - Tests whether the defense can catch subtle, focused model degradation
#
#   def flip_labels(y: np.ndarray, source_class: int, target_class: int) -> np.ndarray:
#       """Flip all instances of source_class to target_class in label array."""
#
# ═══════════════════════════════════════════════════════
# ATTACK 2: Stealthy Backdoor Trigger Injection
# ═══════════════════════════════════════════════════════
#   - Injects synthetic traffic rows with a specific anomalous feature signature:
#       e.g., Flow_Duration = 999999 AND ACK_Flag_Count = 1
#   - Mislabels these injected rows as "Benign"
#   - Scales outgoing gradient L2-norm to match the benign client average
#     (stealth bypass of basic distance / norm-clipping defenses)
#
#   def inject_backdoor_trigger(
#       X: np.ndarray, y: np.ndarray,
#       trigger_feature_idx: list, trigger_values: list,
#       inject_ratio: float
#   ) -> Tuple[np.ndarray, np.ndarray]:
#       """Append trigger-stamped rows mislabeled as benign to the local dataset."""
#
#   def scale_gradient_to_norm(
#       local_weights: List[np.ndarray],
#       target_norm: float
#   ) -> List[np.ndarray]:
#       """Scale whole gradient to match target L2-norm — stealth norm-capping bypass."""
#
# ═══════════════════════════════════════════════════════
# ATTACKER CONFIG (driven by config.yaml):
# ═══════════════════════════════════════════════════════
#   - attacker_ratio: fraction of clients that are malicious (0.0, 0.10, 0.30, 0.50)
#   - attack_start_round: Byzantine injection begins at round 11 (after clean baseline)
#   - attack_type: "label_flip" | "backdoor" | "both"
#   - source_class / target_class: integer class indices for label flip
#   - trigger_feature_idx / trigger_values: backdoor signature definition
#   - inject_ratio: fraction of local data to poison with backdoor trigger
