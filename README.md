# 🛡️ FL-IDS: Robust Federated Learning Intrusion Detection System for IoT

> A research-grade capstone project implementing a novel 3-part Byzantine-robust server-side defense pipeline for Federated Learning-based Intrusion Detection in resource-constrained IoT environments.

---

## 📋 Project Overview

This system trains a distributed intrusion detection model across simulated IoT Edge Gateway clients using **Federated Learning**, without sharing raw network traffic data. The central server defends against **Byzantine attackers** (label-flipping and data poisoning) using a novel defense pipeline built on top of the [Flower](https://flower.ai/) (flwr) framework with **PyTorch** MLP clients.

### Core Research Contributions

Three research gaps in FL-IDS literature are addressed:

| Gap | Problem | Solution |
|-----|---------|---------|
| **Gap 1** | Defenders confuse Non-IID IoT diversity with poisoning attacks | **Layer-Wise Cosine Similarity + MAD** — analyzes only the final classification layer |
| **Gap 2** | O(K²·d) bottleneck makes robust defenses too slow for IoT | **Capped Simplex Projection** — O(K log K) optimization forces malicious weights to 0.0 |
| **Gap 3** | Binary accept/reject rules permanently ban temporarily noisy clients | **EMA Trust Scoring + Temperature-Scaled Softmax** — momentum-based reputation across rounds |

---

## 🏗️ Architecture

```
IoT Edge Gateway (Client)          Central Flower Server
─────────────────────────          ──────────────────────────────────────────
CIC-IDS2017 partition              1. Deserialize PyTorch weights (NumPy)
      ↓                            2. [Gap 1] Layer-wise cosine sim + MAD
PyTorch MLP (78→128→64→32→2)      3. [Gap 3] EMA trust score update
      ↓ local training             4. [Gap 2] Capped simplex projection
Weight update (NumPy arrays)       5. Weighted np.average → new global model
      ↓ via gRPC (Flower)               ↓
[Optional: label-flip / backdoor]  Broadcast global model for next round
```

---

## 📁 Project Structure

```
fl-ids/
├── app.py                          # Main entry point (CLI launcher)
├── requirements.txt                # All Python dependencies
├── setup.py                        # Package installation
├── pytest.ini                      # Test runner configuration
├── .env.example                    # Environment variable template
├── .gitignore
│
├── src/
│   ├── components/
│   │   ├── model.py                # PyTorch MLP definition
│   │   ├── client.py               # Flower NumPyClient (local training)
│   │   ├── server.py               # Flower server entry point
│   │   ├── aggregator.py           # ⭐ 3-part defense pipeline (core contribution)
│   │   ├── attacker.py             # Byzantine attack simulation
│   │   ├── data_partitioner.py     # Dirichlet Non-IID data splitting
│   │   ├── evaluator.py            # F1, ASR, FPR metrics
│   │   └── baselines.py            # FedAvg, Trimmed Mean, Krum
│   │
│   ├── configs/
│   │   ├── config.yaml             # All hyperparameters
│   │   ├── config.py               # Config loader + dataclasses
│   │   └── paths.py                # Centralized path constants
│   │
│   ├── exception/
│   │   └── custom_exception.py     # FL-specific exception classes
│   │
│   ├── logging/
│   │   └── logger.py               # Round-aware custom logger
│   │
│   └── pipelines/
│       ├── training_pipeline.py    # Full FL experiment orchestrator
│       ├── evaluation_pipeline.py  # Post-run plots & comparison
│       └── attack_pipeline.py      # Attack sweep manager
│
├── artifacts/
│   ├── models/                     # Saved .pth checkpoints per round
│   ├── results/                    # CSV logs (F1, ASR, trust scores)
│   ├── plots/                      # Matplotlib/seaborn figures
│   └── data/                       # Processed Non-IID .npz partitions
│
├── notebooks/
│   ├── 01_eda_cic_ids2017          # CIC-IDS2017 exploratory analysis
│   ├── 02_non_iid_partition_visualization  # Dirichlet partition plots
│   ├── 03_results_analysis         # Strategy comparison plots
│   └── 04_attack_simulation_analysis  # Attack deep-dive analysis
│
└── tests/
    ├── test_aggregator.py          # Defense math unit tests
    ├── test_data_partitioner.py    # Partition correctness tests
    ├── test_model.py               # MLP + serialization tests
    └── test_attacker.py            # Attack simulation tests
```

---

## 🚀 Quick Start

### 1. Clone & Set Up Environment

```bash
git clone https://github.com/raunaqmittal/FL-IDS-Capstone.git
cd FL-IDS-Capstone

python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure Data Path

```bash
cp .env.example .env
# Edit .env and set RAW_DATA_PATH to your CIC-IDS2017 CSV directory
```

### 3. Preprocess & Partition Data

```bash
python -m src.components.data_partitioner
```

### 4. Run the Full Experiment

```bash
# Proposed defense (default)
python app.py

# Specific strategy
python app.py --strategy proposed
python app.py --strategy fedavg
python app.py --strategy krum

# Attacker ratio sweep
python app.py --mode attack
```

### 5. Generate Results Plots

```bash
python app.py --mode evaluate
```

---

## 🧪 Running Tests

```bash
pytest                    # All tests with coverage
pytest tests/test_aggregator.py -v    # Defense math only
```

---

## 📊 Experiment Design

| Phase | Rounds | Attackers | Purpose |
|-------|--------|-----------|---------|
| Phase 1 | 1–10 | 0% | Clean baseline convergence |
| Phase 2 | 11–30 | 10% / 30% / 50% | Byzantine attack injection |
| Phase 3 | Post-run | — | Evaluation & comparison plots |

**Metrics tracked:** Macro F1-Score, Attack Success Rate (ASR), False Positive Rate (FPR), per-client Trust Scores.

**Baselines compared:** Standard FedAvg · Federated Trimmed Mean · Krum

---

## ⚙️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Client ML Model | PyTorch MLP |
| FL Orchestration | Flower (flwr) ≥ 1.8 |
| Server Defense Math | NumPy + SciPy |
| Dataset | CIC-IDS2017 |
| Data Processing | Pandas + scikit-learn |
| Plotting | Matplotlib + Seaborn |
| Testing | pytest |

---

## 🚫 Out of Scope

- ❌ Blockchain or Homomorphic Encryption (too heavy for IoT)
- ❌ Heavy models on clients (CNN, Transformer, LLM) — strictly MLP only
- ❌ Raw data access on the server — server sees **only** PyTorch weight arrays
- ❌ GANs or secondary models on edge devices

---

## 📄 License

For academic / capstone use only.
