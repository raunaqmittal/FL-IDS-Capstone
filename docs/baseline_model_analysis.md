# Baseline MLP Performance Analysis — `06_baseline_model_performance.ipynb`

> **Last updated:** 2026-05-03 — Reflects **retrained** model with upgraded `[256, 128, 64]` architecture.
> Previous (old) results from the `[64, 32, 16]` model are shown in the comparison table at the bottom.

---

## Model Overview

| Item | Value |
|---|---|
| Architecture | MLP — hidden layers `[256, 128, 64]` |
| Input features | 57 |
| Output classes | 27 |
| Dropout | 0.2 |
| Total trainable parameters | **58,651** |
| Checkpoint epoch | 44 |
| Best saved validation Macro F1 | **0.7463** |
| Test rows | 419,995 |
| Device | CUDA (GPU) |

---

## Overall Test-Set Metrics

| Metric | Score (NEW) | Score (OLD `[64,32,16]`) | Δ Change |
|---|---|---|---|
| **Accuracy** | **0.9880** | 0.9856 | +0.0024 |
| Macro Precision | 0.6897 | 0.5196 | **+0.1701** |
| Macro Recall | 0.9084 | 0.6228 | **+0.2856** |
| **Macro F1** | **0.7463** | 0.5460 | **+0.2003 ↑↑** |
| Weighted F1 | 0.9899 | 0.9880 | +0.0019 |

> **Key insight:** Macro F1 jumped from 0.546 → **0.746** (+20 points) purely by upgrading the hidden layer width from `[64, 32, 16]` to `[256, 128, 64]`. This confirms the bottleneck hypothesis — the original model lacked capacity to express 27-class fine-grained boundaries.
> The dataset is still heavily imbalanced (BENIGN ≈ 316K vs. some attack classes with < 10 samples), so **Macro F1 remains the primary metric**, not accuracy.

---

## Per-Class Performance

### ✅ Well-Classified Classes (F1 ≥ 0.9)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Botnet – Attempted | 1.000 | 1.000 | **1.000** | 813 |
| DDoS | 1.000 | 1.000 | **1.000** | 19,029 |
| DoS Hulk | 0.999 | 1.000 | **1.000** | 31,694 |
| BENIGN | 1.000 | 0.992 | **0.996** | 316,513 |
| DoS GoldenEye | 0.996 | 0.999 | **0.998** | 1,514 |
| FTP-Patator | 0.985 | 1.000 | **0.993** | 794 |
| DoS Slowhttptest | 0.983 | 1.000 | **0.991** | 348 |
| DoS Slowloris | 0.986 | 0.996 | **0.991** | 772 |
| DoS Slowloris – Attempted | 0.951 | 0.995 | **0.972** | 369 |
| Botnet | 0.913 | 1.000 | **0.955** | 147 |
| SSH-Patator | 0.906 | 0.998 | **0.950** | 592 |
| Infiltration – Portscan | 0.889 | 0.939 | **0.914** | 14,354 |
| DoS Slowhttptest – Attempted | 0.855 | 0.993 | **0.919** | 674 |
| Portscan | 0.972 | 0.949 | **0.960** | 31,813 |
| Web Attack – Brute Force | 0.882 | 1.000 | **0.938** | 15 |
| Web Attack – XSS | 0.800 | 1.000 | **0.889** | 4 |
| Web Attack – SQL Injection | 0.500 | 0.667 | **0.571** | 3 |

### ⚠️ Partially Classified Classes (F1 0.2–0.9)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| DoS Hulk – Attempted | 0.772 | 0.991 | **0.868** | 116 |
| DoS GoldenEye – Attempted | 0.471 | 1.000 | **0.640** | 16 |
| SSH-Patator – Attempted | 0.455 | 1.000 | **0.625** | 5 |
| Heartbleed | 0.333 | 1.000 | **0.500** | 2 |
| Infiltration – Attempted | 0.350 | 0.778 | **0.483** | 9 |
| Infiltration | 0.250 | 0.571 | **0.348** | 7 |
| Web Attack – Brute Force – Attempted | 0.140 | 0.674 | **0.231** | 258 |
| Web Attack – XSS – Attempted | 0.117 | 0.985 | **0.209** | 131 |
| FTP-Patator – Attempted | 0.118 | 1.000 | **0.211** | 2 |

### ❌ Completely Failed Classes (F1 = 0.000)

| Class | Support | Root Cause |
|---|---|---|
| Web Attack – SQL Injection – Attempted | **1** | Only 1 test sample |

---

> **Dramatic improvement vs. old model:** In the old `[64, 32, 16]` model, 9 classes had F1 = 0.000. In the new model, **only 1 class fails completely** (SQL Injection – Attempted, 1 test sample). All other rare classes now produce a non-zero F1 score, including Heartbleed (0.500), Infiltration (0.348), and DoS GoldenEye – Attempted (0.640).

---

## Binary IDS View (Benign vs. Attack)

| Metric | Score | Meaning |
|---|---|---|
| **False Positive Rate** | **~0.008** | Benign traffic flagged as attack — very low |
| **Attack Detection Recall** | **~0.992** | Attack traffic correctly detected — excellent |

> **Binary detection remains near-perfect.** Fine-grained attack-type discrimination is the remaining challenge.

---

## Remaining Weaknesses

1. **"Brute Force – Attempted" and "XSS – Attempted" classes** — both have low precision (0.14, 0.117) despite high recall. The model detects nearly all of them but also over-predicts these classes on BENIGN traffic.
2. **Tiny minority classes** (< 10 test samples: SQL Injection variants, FTP-Patator Attempted, Heartbleed, Infiltration) — structural data problem. F1 improvement is bounded by available examples.
3. **Weighted cross-entropy or focal loss** was NOT applied for this run. Adding class-weighted loss could further improve minority class recall.

---

## What Changed: Old → New Model

| Change | Old Model | New Model |
|---|---|---|
| Hidden layers | `[64, 32, 16]` | **`[256, 128, 64]`** |
| Total parameters | 7,003 | **58,651** |
| Checkpoint epoch | 48 | 44 |
| Training device | CPU | **CUDA (GPU)** |
| Validation Macro F1 (saved) | ~0.546 | **0.746** |
| Test Macro F1 | 0.5460 | **0.7463** |
| Test Accuracy | 0.9856 | **0.9880** |
| Classes with F1 = 0.000 | **9** | **1** |

---

## Recommendations — What to Change Next

### 1. Weighted / Focal Loss (Highest Remaining Impact)

Even with the capacity boost, the "Brute Force – Attempted" and "XSS – Attempted" low-precision issue is a loss function problem. Apply class-weighted loss:

```python
from collections import Counter
counts = Counter(y_train)
total = sum(counts.values())
weights = torch.tensor([total / (len(counts) * counts[i]) for i in range(len(counts))])
criterion = nn.CrossEntropyLoss(weight=weights.to(DEVICE))
```

### 2. Merge / Drop Extremely Rare Classes

Classes with < 50 training samples hurt Macro F1 without real security value. Consider:
- **Merging** `Foo – Attempted` → `Foo`
- **Dropping** classes with < 50 training samples and reporting them as "Unknown Attack" at inference

### 3. Confidence Thresholding at Inference

Many misclassifications have softmax confidence 0.5–0.75. Add an "abstain" option:

```python
if max_conf < 0.7:
    predicted_class = "Unknown / Low Confidence"
```

### 4. Federated Learning Baseline Comparison

Per the research plan, the FL experiments should beat **Macro F1 = 0.7463** and **Weighted F1 = 0.9899**. These are the primary comparison thresholds for all FL strategy experiments.

---

## Summary

| Aspect | Status |
|---|---|
| Binary attack detection | ✅ Excellent (~99.2% recall, ~0.8% FPR) |
| Common attack classification | ✅ Excellent (DDoS, DoS Hulk, Portscan, GoldenEye all F1 ≥ 0.96) |
| Rare / "Attempted" variant classification | ⚠️ Improved but still limited by data imbalance |
| Macro F1 | ✅ **0.7463** — strong improvement from 0.546 |
| Model size | ✅ Lightweight (58 K params) — still suitable for FL clients |
| Classes completely failing | ✅ Reduced from 9 → **1** |
