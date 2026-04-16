# evaluator.py — Metrics computation for FL-IDS global and per-round evaluation.
#
# Evaluates intrusion detection performance beyond simple accuracy:
# (accuracy alone is deceptive due to severe class imbalance in CIC-IDS2017)
#
# ═══════════════════════════════════════════════════════
# METRICS TO COMPUTE:
# ═══════════════════════════════════════════════════════
#
#   1. Macro F1-Score
#      - Ensures minority, rare attack classes (e.g., Infiltration, Web Attacks)
#        are learned and retained by the global model.
#      - Use sklearn.metrics.f1_score(average='macro')
#
#   2. Attack Success Rate (ASR) — primary backdoor defense metric
#      - Fraction of injected backdoor trigger samples that the global model
#        WRONGLY classifies as Benign.
#      - ASR near 0% = defense is working. ASR near 100% = backdoor implanted.
#      - Requires a dedicated trigger test set (built by attacker.py).
#
#   3. False Positive Rate (FPR)
#      - Fraction of genuinely Benign traffic flagged as Malicious.
#      - Critical for real-world SOC deployment — high FPR = alert fatigue.
#      - FPR = FP / (FP + TN)
#
#   4. Confusion Matrix
#      - Full per-class breakdown for post-experiment analysis in notebooks.
#
#   5. Per-Round Trust Score Snapshot
#      - Log the EMA reputation score of each client per round for heatmap plotting.
#
# ═══════════════════════════════════════════════════════
# KEY FUNCTIONS TO IMPLEMENT:
# ═══════════════════════════════════════════════════════
#
#   def compute_metrics(y_true, y_pred) -> dict:
#       """Return dict with keys: macro_f1, accuracy, fpr, confusion_matrix."""
#
#   def compute_asr(model, trigger_test_loader, benign_class_idx: int) -> float:
#       """Evaluate Attack Success Rate against backdoor trigger test set."""
#
#   def log_round_results(server_round: int, metrics: dict, results_path: str) -> None:
#       """Append per-round metrics to a CSV file in artifacts/results/."""
#
#   def log_trust_scores(server_round: int, trust_scores: dict, results_path: str) -> None:
#       """Append client trust score snapshots per round for heatmap generation."""
