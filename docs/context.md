# FL-IDS Capstone — Master Context File

> **Purpose:** Give a new AI (or future session) full project context in one file.
> Last updated: 2026-04-19

---

## 1. Project Summary

**FL-IDS** is a Federated Learning–based Intrusion Detection System for IoT edge gateways.
Clients train locally on CIC-IDS2017 network traffic data and send only model weights to a central server — raw data never leaves the device.
The server defends against Byzantine (label-flipping / data-poisoning) attacks using a 3-part math pipeline before aggregating the global model.

**Stack:**
- ML: `PyTorch` (MLP classifier)
- FL orchestration: `Flower (flwr)`
- Dataset: CIC-IDS2017 (~78 features, tabular, multi-class)
- Data distribution: Non-IID via Dirichlet(α=0.5) to simulate real IoT heterogeneity
- Python package structure under `src/`

---

## 2. Directory Structure

```
FL IDS/
├── app.py                          # Entry point
├── requirements.txt
├── setup.py
├── pytest.ini
├── docs/
│   ├── context.md                  ← THIS FILE
│   ├── ProjectOverview.md
│   ├── Federated Learning IDS Research Review.md
│   └── FL-IDS Research for IoT Security.md
├── notebooks/
│   └── 02_non_iid_partition_visualization.py
├── artifacts/
│   └── data/                       # client_0000.npz … client_NNNN.npz (generated)
├── src/
│   ├── configs/
│   │   └── config.py               # Loads config.yaml; CONFIG dict used everywhere
│   ├── logging/logger.py
│   ├── exception/exception.py
│   ├── pipelines/
│   │   ├── data_pipeline.py        ✅ IMPLEMENTED
│   │   ├── attack_pipeline.py      🔲 STUB
│   │   ├── training_pipeline.py    🔲 STUB
│   │   └── evaluation_pipeline.py  🔲 STUB
│   └── components/
│       ├── data/
│       │   ├── data_loader.py       ✅ IMPLEMENTED
│       │   ├── data_preprocessor.py ✅ IMPLEMENTED
│       │   ├── data_partitioner.py  ✅ IMPLEMENTED
│       │   └── torch_dataset.py     ✅ IMPLEMENTED
│       ├── model/
│       │   └── model.py             🔲 STUB (design only, not coded)
│       ├── client/
│       │   ├── client.py            🔲 STUB (design only, not coded)
│       │   └── attacker.py          🔲 STUB (design only, not coded)
│       ├── server/
│       │   ├── aggregator.py        🔲 STUB (design + math spec, not coded)
│       │   ├── baselines.py         🔲 STUB
│       │   └── server.py            🔲 STUB
│       └── evaluation/             🔲 STUB
```

---

## 3. What Is FULLY Implemented

### 3.1 Data Pipeline — `src/pipelines/data_pipeline.py`

Orchestrates the full data flow end-to-end:

```
load_cicids2017()
  → preprocess()        (Steps 1–5)
  → train_test_split()  (Step 7: 80/20 stratified)
  → StandardScaler()    (Step 6: fit on train only — no leakage)
  → run_partitioning()  (Step 8: IID or Non-IID)
```

Returns: `feature_cols, le, scaler, X_test, y_test`

### 3.2 Data Loader — `src/components/data/data_loader.py`

- Loads CIC-IDS2017 from HuggingFace (`bvk/CICIDS-2017`) as a pandas DataFrame.

### 3.3 Data Preprocessor — `src/components/data/data_preprocessor.py`

All 5 preprocessing steps are implemented:

| Step | Function | What it does |
|------|----------|--------------|
| 1 | `drop_unusable()` | Drops ID cols (`Src IP dec`, `Dst IP dec`, `Timestamp`, `Attempted Category`), drops all-NaN columns, keeps only numeric + Label |
| 2 | `impute()` | Replaces `Inf/-Inf → NaN`, fills NaN with **column median** |
| 3 | `variance_filter()` | Removes zero-variance (constant) features |
| 4 | `correlation_filter()` | Removes features with Pearson `\|r\| > 0.95` (reduces to ~40–50 features) |
| 5 | `encode_labels()` | `LabelEncoder`: BENIGN=0, DDoS=1, PortScan=2, etc. |

> **Note:** StandardScaler (Step 6) is intentionally NOT in this file — it lives in `data_pipeline.py` and is fit **only on training data** to prevent data leakage.

### 3.4 Data Partitioner — `src/components/data/data_partitioner.py`

- `partition_iid(X, y, num_clients)` — equal random split
- `partition_non_iid(X, y, num_clients, alpha=0.5)` — Dirichlet distribution
- `save_partitions(partitions, output_dir)` — saves as `artifacts/data/client_NNNN.npz`
- `load_partition(client_id, data_dir)` — loads a client's `.npz` shard
- `run_partitioning(X, y)` — reads config and dispatches to IID or Non-IID

Config keys used: `num_clients`, `partition_mode` (`iid` | `non_iid`), `alpha_dirichlet`, `random_seed`

### 3.5 PyTorch DataLoader — `src/components/data/torch_dataset.py`

```python
make_dataloader(X: np.ndarray, y: np.ndarray, batch_size=32, shuffle=True) -> DataLoader
```

Converts numpy arrays → `TensorDataset` → `DataLoader`.
**Usage in client training loop:** `loader = make_dataloader(X_client, y_client)`

---

## 4. What Is STUBBED (design spec only, code NOT written yet)

### 4.1 MLP Model — `src/components/model/model.py`

**To implement:**
```python
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout_rate): ...
    def forward(self, x): ...

def get_model_parameters(model) -> List[np.ndarray]: ...  # state_dict → NumPy list
def set_model_parameters(model, parameters) -> None: ...  # NumPy list → state_dict
```

Architecture: `input_dim → [256, 128, 64] → output_dim`, ReLU + Dropout. Lightweight for IoT.

### 4.2 FL Client — `src/components/client/client.py`

**To implement:**
```python
class FLIDSClient(flwr.client.NumPyClient):
    def get_parameters(self, config): ...
    def fit(self, parameters, config): ...       # local training + optional attack injection
    def evaluate(self, parameters, config): ...  # local val evaluation

def client_fn(cid: str) -> FLIDSClient: ...  # loads client_NNNN.npz, creates DataLoaders
```

### 4.3 Attacker — `src/components/client/attacker.py`

Two attack types (both stubs):

**Attack 1 — Targeted Semantic Label-Flipping:**
```python
def flip_labels(y, source_class, target_class) -> np.ndarray:
    # Flips ONLY DDoS→Benign. NOT random (random noise gets averaged out by FedAvg).
```

**Attack 2 — Stealthy Backdoor Trigger:**
```python
def inject_backdoor_trigger(X, y, trigger_feature_idx, trigger_values, inject_ratio):
    # Injects rows with anomalous feature signature (e.g., Flow_Duration=999999, ACK=1)
    # Mislabels them as Benign
def scale_gradient_to_norm(local_weights, target_norm):
    # Scales gradient L2-norm to match benign average (bypasses norm-clipping defenses)
```

Config: `attacker_ratio`, `attack_start_round` (round 11), `attack_type`, `source_class`, `target_class`, `trigger_feature_idx`, `trigger_values`, `inject_ratio`

### 4.4 Server Aggregator — `src/components/server/aggregator.py`

**This is the core research contribution.** Full math spec is in the file.
Three gaps are addressed in `aggregate_fit()`:

**Gap 1 — Layer-Wise Cosine Similarity + MAD (Non-IID vs. Malicious dilemma):**
- Extract only final classification layer weights from each client
- Compute pairwise cosine similarity matrix
- Per-client consensus score = median of row
- MAD robust Z-score: `M_i = 0.6745 * (c_i - median(c)) / (MAD + 1e-9)`
- Flag adversarial if `M_i < -3.0`

**Gap 3 — EMA Momentum Trust Scoring (rigid adaptation):**
- Persistent `self.reputation_scores` dict across FL rounds
- EMA update: `RS_i(t) = mu_d * RS_i(t-1) + (1 - mu_d) * P_i(t)`
- Temperature-scaled softmax → aggregation weights (LogSumExp trick)

**Gap 2 — Capped Simplex Projection O(K log K) (computational bottleneck):**
- Projects trust weights onto Sparse Unit-Capped Simplex
- Forces malicious clients to exactly `0.0` aggregation weight
- `cap_t = 1 / (K - b_f)` where `b_f` = max tolerated Byzantine clients

**Functions to implement:**
```python
def extract_final_layer(ndarrays) -> np.ndarray
def compute_layer_wise_cosine_similarity(final_layers) -> np.ndarray
def compute_mad_scores(sim_matrix) -> np.ndarray
def project_capped_simplex(v, cap_t) -> np.ndarray
def temperature_scaled_softmax(scores, temperature) -> np.ndarray
def update_ema_reputation(self, client_ids, current_scores) -> np.ndarray

class RobustFLIDSStrategy(flwr.server.strategy.Strategy):
    def aggregate_fit(self, server_round, results, failures): ...  # MAIN DEFENSE
```

### 4.5 Baseline Aggregators — `src/components/server/baselines.py`

For comparison in experiments:
- Standard `FedAvg`
- `FedTrimmedMean` (trim top/bottom 20%)
- `Krum` / `MultiKrum`

### 4.6 Pipelines (stubs)

- `attack_pipeline.py` — wires up attacker clients for adversarial runs
- `training_pipeline.py` — runs Flower simulation with `flwr.simulation.start_simulation()`
- `evaluation_pipeline.py` — computes Macro F1, ASR (Attack Success Rate), FPR per round

---

## 5. Research Architecture (The 3 Gaps)

| Gap | Problem | Solution |
|-----|---------|----------|
| Gap 1 | Non-IID benign clients look malicious if whole model is compared | Cosine Similarity on **final layer only** + MAD robust scoring |
| Gap 2 | Standard filters are O(K²) — too slow for IoT scale | Capped Simplex Projection — O(K log K) |
| Gap 3 | Rigid accept/reject permanently bans good nodes | EMA trust scores + Temperature-Scaled Softmax weights |

---

## 6. Experiment Plan

| Phase | Rounds | Activity |
|-------|--------|----------|
| Phase 1 | 1–10 | Baseline training, 0% attackers, establish F1 |
| Phase 2 | 11–30 | Byzantine injection (10%, 30%, 50% attacker ratios) |
| Phase 3 | Eval | Plot F1-score drop + ASR spike across all aggregators |

**Baselines to compare against:** FedAvg, FedTrimmedMean, Krum

**Key metrics:** Macro F1-Score, Attack Success Rate (ASR), False Positive Rate (FPR)

**Network:** N=50 clients total, C=20 sampled per round

---

## 7. Strict Project Boundaries (NEVER suggest these)

- ❌ Blockchain or Homomorphic Encryption — too heavy for IoT
- ❌ Heavy models (CNNs, Transformers) on clients — only MLP
- ❌ Server-side raw data — server sees ONLY PyTorch weights (NumPy arrays)
- ❌ Ollama or any LLM — out of scope
- ❌ Client-side complex defense logic — ALL defense math is server-only

---

## 8. Key Implementation Rules

- Every file uses `from src.logging.logger import logging` and `from src.exception.exception import FLIDSException`
- All config is read via `from src.configs.config import CONFIG` (a dict loaded from `config.yaml`)
- Keep code **clean and short** — no unnecessary boilerplate
- Scaler is always fit **only on training data** (already enforced in `data_pipeline.py`)
- `attacker.py` logic is called **inside** `client.py`'s `fit()` method, not separately
- `make_dataloader()` from `torch_dataset.py` is the standard way to create DataLoaders everywhere
