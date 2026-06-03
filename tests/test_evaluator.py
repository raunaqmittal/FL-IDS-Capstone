import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.components.evaluation.evaluator import compute_metrics, compute_asr, log_round_results, log_trust_scores
from src.configs.paths import RESULTS_DIR


# -- Test 1: compute_metrics accuracy and macro_f1
print("Test 1: compute_metrics")
y_true = np.array([0, 0, 1, 1, 2, 2])
y_pred = np.array([0, 0, 1, 0, 2, 2])
m = compute_metrics(y_true, y_pred)
assert abs(m["accuracy"] - (5/6)) < 1e-5
assert 0 < m["macro_f1"] < 1
assert 0 <= m["fpr"] <= 1
assert m["confusion_matrix"].shape == (3, 3)
print(f"  acc={m['accuracy']:.4f}, macro_f1={m['macro_f1']:.4f}, fpr={m['fpr']:.4f}  [OK]")


# -- Test 2: compute_metrics FPR = 0 when no false positives
print("\nTest 2: compute_metrics FPR=0 on perfect benign")
y_true2 = np.array([0, 0, 1, 1])
y_pred2 = np.array([0, 0, 1, 1])
m2 = compute_metrics(y_true2, y_pred2)
assert m2["fpr"] == 0.0
print(f"  FPR={m2['fpr']}  [OK]")


# -- Test 3: compute_asr returns 0 for clean model (no trigger triggers benign)
print("\nTest 3: compute_asr = 0 on adversarial data with correct model")
X = torch.randn(20, 57)
y = torch.ones(20, dtype=torch.long)  # all class 1 (not benign)
loader = DataLoader(TensorDataset(X, y), batch_size=8)

from src.components.model.model import MLPClassifier
model = MLPClassifier(57, [256, 128, 64], 27)
model.eval()
asr = compute_asr(model, loader, benign_class_idx=0)
assert 0.0 <= asr <= 1.0
print(f"  ASR={asr:.4f} (random model, no guarantees — just checking range)  [OK]")


# -- Test 4: log_round_results writes CSV
print("\nTest 4: log_round_results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
test_file = "test_round_results_tmp.csv"
log_round_results(1, {"accuracy": 0.99, "macro_f1": 0.55, "weighted_f1": 0.98, "fpr": 0.01}, filename=test_file)
log_round_results(2, {"accuracy": 0.98, "macro_f1": 0.56, "weighted_f1": 0.97, "fpr": 0.02}, filename=test_file)
path = RESULTS_DIR / test_file
assert path.exists()
lines = path.read_text().strip().split("\n")
assert len(lines) == 3  # header + 2 rows
assert lines[0].startswith("round")
print(f"  CSV has {len(lines)} lines (header + 2 rows)  [OK]")
path.unlink()  # cleanup


# -- Test 5: log_trust_scores writes CSV
print("\nTest 5: log_trust_scores")
test_trust_file = "test_trust_scores_tmp.csv"
log_trust_scores(1, {"0": 0.5, "1": 0.3, "2": -0.8}, filename=test_trust_file)
path2 = RESULTS_DIR / test_trust_file
assert path2.exists()
lines2 = path2.read_text().strip().split("\n")
assert len(lines2) == 4  # header + 3 client rows
print(f"  Trust CSV has {len(lines2)} lines  [OK]")
path2.unlink()  # cleanup


print("\nAll evaluator tests passed.")
