# Baseline MLP Performance Analysis — `06_baseline_model_performance.ipynb`

## Model Overview

| Item | Value |
|---|---|
| Architecture | MLP — hidden layers `[64, 32, 16]` |
| Input features | 57 |
| Output classes | 27 |
| Dropout | 0.2 |
| Total trainable parameters | **7,003** |
| Checkpoint epoch | 48 |
| Test rows | 419,995 |
| Device | CPU |

---

## Overall Test-Set Metrics

| Metric | Score |
|---|---|
| **Accuracy** | **0.9856** |
| Macro Precision | 0.5196 |
| Macro Recall | 0.6228 |
| **Macro F1** | **0.5460** |
| Weighted F1 | 0.9880 |

> **Key insight:** High accuracy and weighted F1 are misleading here — the dataset is heavily imbalanced (BENIGN ≈ 316 K vs. some attack classes with < 10 samples). **Macro F1 of 0.546 is the honest signal** and shows the model struggles with rare attack classes.

---

## Per-Class Performance

### ✅ Well-Classified Classes (F1 ≥ 0.9)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Botnet – Attempted | 1.000 | 1.000 | **1.000** | 813 |
| DDoS | 1.000 | 1.000 | **1.000** | 19,029 |
| DoS Hulk | 0.997 | 1.000 | **0.998** | 31,694 |
| BENIGN | 1.000 | 0.990 | **0.995** | 316,513 |
| FTP-Patator | 0.989 | 0.999 | **0.994** | 794 |
| DoS Slowhttptest | 0.872 | 0.994 | **0.929** | 348 |

### ⚠️ Moderately Classified Classes (F1 0.5–0.9)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Botnet | 0.948 | 1.000 | 0.974 | 147 |
| DoS GoldenEye | 0.943 | 0.999 | 0.970 | 1,514 |
| SSH-Patator | 0.944 | 0.998 | 0.970 | 592 |
| Infiltration – Portscan | 0.885 | 0.930 | 0.907 | 14,354 |
| DoS Slowloris | 0.923 | 0.997 | 0.959 | 772 |
| DoS Slowloris – Attempted | 0.920 | 0.995 | 0.956 | 369 |
| Portscan | 0.967 | 0.950 | 0.959 | 31,813 |
| DoS Slowhttptest – Attempted | 0.814 | 0.993 | 0.894 | 674 |
| DoS Hulk – Attempted | 0.423 | 0.991 | **0.593** | 116 |
| Web Attack – Brute Force | 0.312 | 1.000 | **0.476** | 15 |

### ❌ Completely Failed Classes (F1 = 0.000)

| Class | Support | Root Cause |
|---|---|---|
| DoS GoldenEye – Attempted | **16** | Tiny minority class |
| FTP-Patator – Attempted | **2** | Only 2 test samples |
| Heartbleed | **2** | Only 2 test samples |
| Infiltration | **7** | Only 7 test samples |
| Infiltration – Attempted | **9** | Only 9 test samples |
| SSH-Patator – Attempted | **5** | Only 5 test samples |
| Web Attack – SQL Injection | **3** | Only 3 test samples |
| Web Attack – SQL Injection – Attempted | **1** | Only 1 test sample |
| Web Attack – XSS | **4** | Only 4 test samples |
| Web Attack – XSS – Attempted | **131** | Very small class |

---

## Binary IDS View (Benign vs. Attack)

| Metric | Score | Meaning |
|---|---|---|
| **False Positive Rate** | **0.0014** | Benign traffic flagged as attack — very low |
| **Attack Detection Recall** | **0.9992** | Attack traffic correctly detected — excellent |
| **Missed Attack Rate** | **0.0008** | Attack traffic predicted as benign — very low |

> **Binary detection is near-perfect** (99.9% recall, 0.14% FPR). The problem is entirely in **fine-grained attack-type classification**, not in benign vs. attack discrimination.

---

## Confidence & Error Analysis — Top Misclassification Pairs

| True Class | Predicted As | Errors | Mean Confidence |
|---|---|---|---|
| BENIGN | Web Attack – Brute Force – Attempted | **2,337** | 0.613 |
| Portscan | Infiltration – Portscan | 1,543 | 0.629 |
| Infiltration – Portscan | Portscan | 984 | 0.765 |
| BENIGN | Infiltration – Portscan | 187 | 0.774 |
| BENIGN | DoS Slowhttptest – Attempted | 153 | 0.941 |
| BENIGN | DoS Hulk – Attempted | 137 | 0.527 |
| Web Attack – XSS – Attempted | Web Attack – Brute Force – Attempted | 129 | 0.651 |
| BENIGN | DoS GoldenEye | 80 | 0.733 |
| BENIGN | DoS Hulk | 77 | 0.813 |
| BENIGN | DoS Slowloris | 64 | 0.703 |

> **Notable:** The largest error bucket (2,337 cases) is BENIGN being mislabelled as "Web Attack – Brute Force – Attempted" with moderate confidence (0.61). The model is not highly confident about these mistakes, suggesting the class boundary is genuinely blurry.

---

## Diagnosis

### What's Working
- **Binary IDS accuracy is excellent.** The model reliably separates attack from benign at >99.9% recall.
- **High-volume classes** (DDoS, DoS Hulk, Portscan, BENIGN) are learned well.

### What's Not Working

1. **Extreme class imbalance** — 9 of 27 classes have ≤ 9 test samples. The model simply never sees enough examples to learn decision boundaries for these classes.
2. **"Attempted" variant classes** are semantically very similar to their non-attempted counterparts, making them inherently hard to separate with only 57 features.
3. **Model capacity may be a ceiling.** With only 7,003 parameters and 57 features, the MLP cannot easily express 27-way fine-grained boundaries.

---

## Recommendations — What to Change

### 1. Address Class Imbalance (Highest Impact)

| Technique | How |
|---|---|
| **Weighted loss** | Pass `class_weight` / `pos_weight` inversely proportional to class frequency to the cross-entropy loss |
| **SMOTE / oversampling** | Oversample minority classes in the preprocessed training set |
| **Undersample BENIGN** | Reduce BENIGN from 1.6 M+ to closer to the attack distributions |

In code:
```python
# Compute inverse-frequency weights
from collections import Counter
counts = Counter(y_train)
total = sum(counts.values())
weights = torch.tensor([total / (len(counts) * counts[i]) for i in range(len(counts))])
criterion = nn.CrossEntropyLoss(weight=weights.to(DEVICE))
```

### 2. Merge / Drop Extremely Rare Classes

Classes with fewer than ~50 training samples (the "Attempted" variants, Heartbleed, Infiltration, etc.) hurt Macro F1 without providing real security value. Consider:
- **Merging** `Foo – Attempted` → `Foo`
- **Dropping** classes with < 50 training samples and reporting them as "Unknown Attack" at inference

### 3. Increase Model Capacity

The current MLP `[64, 32, 16]` bottlenecks to 16 units before the 27-class head. Try:

```python
hidden_dims = [256, 128, 64]   # wider
# or add a residual connection block
# or use batch normalisation between layers
```

Batch normalisation often helps with imbalanced tabular data:
```python
nn.Linear(in, out),
nn.BatchNorm1d(out),
nn.ReLU(),
nn.Dropout(p=0.3),
```

### 4. Use Focal Loss

Focal loss automatically down-weights easy (high-confidence) samples and focuses training on hard (rare) samples:

```python
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()
```

### 5. Feature Engineering

With 57 features for 27 classes the signal-to-noise ratio is limited. Consider:
- Remove highly correlated features (check VIF / Pearson correlation matrix)
- Add derived flow-level features (packet inter-arrival variance, burst ratios)
- Normalise per-flow statistics rather than globally

### 6. Confidence Thresholding at Inference

The error analysis shows many mistakes have softmax confidence 0.5–0.75. Add an "abstain" option:

```python
if max_conf < 0.7:
    predicted_class = "Unknown / Low Confidence"
```

This trades recall for precision on the fine-grained labels while preserving binary IDS accuracy.

### 7. Federated Learning Baseline Comparison

Per the notebook's own talking points, the federated experiments should beat **Macro F1 = 0.546** and **Weighted F1 = 0.988**. Track these as your primary comparison metrics, not accuracy alone.

---

## Summary

| Aspect | Status |
|---|---|
| Binary attack detection | ✅ Excellent (99.9% recall, 0.14% FPR) |
| Common attack classification | ✅ Good (DDoS, DoS Hulk, Portscan all F1 > 0.95) |
| Rare / "Attempted" variant classification | ❌ Failed — primarily a data imbalance problem |
| Macro F1 | ⚠️ 0.546 — room for significant improvement |
| Model size | ✅ Lightweight (7 K params) — suitable for FL clients |
