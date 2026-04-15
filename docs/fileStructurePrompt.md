I need you to scaffold a complete, production-grade Python project directory structure for a Federated Learning Intrusion Detection System (FL-IDS) capstone project. Do NOT write any code — only create the folder and file structure with brief inline comments explaining the purpose of each file.

---

## PROJECT SUMMARY

This is a research-grade cybersecurity capstone project. The system trains a distributed intrusion detection model across simulated IoT Edge Gateway clients using Federated Learning, without sharing raw data. The central server defends against Byzantine attackers (label-flipping and data poisoning) using a novel 3-part server-side defense pipeline.

**Core Technologies:**
- PyTorch (client-side MLP model)
- Flower / flwr (federated orchestration framework)
- NumPy (server-side math for defense pipeline)
- CIC-IDS2017 dataset (tabular, ~78 network traffic features)
- Python 3.10+

---

## SYSTEM ARCHITECTURE (CRITICAL — structure must reflect this)

### Clients (IoT Edge Gateways):
- Each client runs a lightweight PyTorch MLP locally
- Input: ~78 tabular features (CIC-IDS2017)
- Output: Softmax classifier (Benign vs. Malicious)
- Some clients are "poisoned" (Byzantine attackers doing label-flipping)

### Server (Central Flower Aggregator):
- Runs a custom Flower Strategy (subclasses `flwr.server.strategy.Strategy`)
- Overrides `aggregate_fit` to inject a 3-part defense pipeline:
  1. **Layer-Wise Cosine Similarity + MAD** — analyzes only the final classification layer to distinguish Non-IID diversity from actual poisoning
  2. **Capped Simplex Projection** — O(K log K) optimization-based filtering that forces malicious client weights to exactly 0.0
  3. **EMA Trust Scoring + Temperature-Scaled Softmax** — momentum-based persistent trust scores per client across rounds

### Experiment Pipeline:
- Phase 1: 10 rounds, 0% attackers (baseline)
- Phase 2: Rounds 11–30, inject Byzantine attackers (10%, 30%, 50% ratios)
- Phase 3: Evaluate & plot Global F1-Score and Attack Success Rate (ASR) vs FedAvg, Trimmed Mean, Krum baselines

---

## REQUIRED TOP-LEVEL STRUCTURE

Model it after this reference structure from another ML project:

project-root/
├── artifacts/       # saved model checkpoints, serialized weights, experiment outputs
├── notebooks/       # Jupyter notebooks for EDA, visualization, result plots
├── src/             # all source code
│   ├── components/  # modular ML pipeline components
│   ├── configs/     # config files / dataclasses
│   ├── exception/   # custom exception classes
│   ├── logging/     # custom logger setup
│   └── pipelines/   # end-to-end training/evaluation pipeline runners
├── tests/
├── app.py
├── requirements.txt
├── setup.py
└── .env.example

---

## DETAILED REQUIREMENTS FOR EACH FOLDER

### `artifacts/`
- `models/` — saved global model `.pth` checkpoints per round
- `results/` — CSV logs of F1-score, ASR, trust scores per round
- `plots/` — saved matplotlib/seaborn figures (F1 curve, ASR curve, trust heatmaps)
- `data/` — processed Non-IID partition files per client (NOT raw CIC-IDS2017)

### `notebooks/`
- EDA notebook for CIC-IDS2017 (feature distributions, class imbalance)
- Non-IID partition visualization notebook
- Results analysis & plotting notebook (comparing FedAvg vs Krum vs proposed defense)
- Attack simulation analysis notebook

### `src/components/`
This is the most important folder. Must contain:
- `model.py` — PyTorch MLP definition
- `client.py` — Flower client class (local training, PyTorch, returns weights)
- `server.py` — Flower server entry point + custom Strategy class
- `aggregator.py` — The 3-part defense pipeline logic:
  - layer-wise cosine similarity extractor
  - MAD-based anomaly scoring
  - capped simplex projection (NumPy)
  - EMA trust score updater
  - temperature-scaled softmax weight converter
- `attacker.py` — Byzantine attacker simulation (label-flipping, data poisoning logic)
- `data_partitioner.py` — Dirichlet-based Non-IID data splitting for CIC-IDS2017
- `evaluator.py` — Metrics: F1-score, ASR, confusion matrix computation
- `baselines.py` — FedAvg, Trimmed Mean, Krum implementations for comparison

### `src/configs/`
- `config.py` or `config.yaml` — All hyperparameters: num_clients=50, sample_fraction=0.4, num_rounds=30, attacker_ratio, EMA decay alpha, simplex cap_t, MLP hidden layers, learning rate, etc.
- `paths.py` — Centralized path constants (artifact dirs, data dirs)

### `src/exception/`
- `custom_exception.py` — Custom exception classes for FL errors, aggregation errors, data loading errors

### `src/logging/`
- `logger.py` — Custom logging setup (log round number, trust scores, defense decisions per round to file and console)

### `src/pipelines/`
- `training_pipeline.py` — Orchestrates the full FL experiment: initializes server, spawns clients, runs Flower simulation
- `evaluation_pipeline.py` — Loads saved results CSVs and generates all comparison plots
- `attack_pipeline.py` — Configures and injects Byzantine clients at specified rounds

### `tests/`
- Unit tests for: simplex projection math, EMA trust update, cosine similarity layer extraction, data partitioner output shapes

### Root-level files:
- `app.py` — Main entry point to launch the FL simulation
- `requirements.txt` — All Python dependencies (torch, flwr, numpy, scikit-learn, pandas, matplotlib, seaborn, pyyaml)
- `setup.py`
- `.env.example` — Environment variable template
- `README.md` — Project overview

---

## CONSTRAINTS & NOTES
- NO blockchain, homomorphic encryption, or GAN components anywhere
- NO heavy models on clients — clients ONLY run the lightweight MLP
- ALL defense logic (cosine similarity, MAD, EMA, simplex projection) lives STRICTLY in `src/components/aggregator.py` on the server side
- The server NEVER accesses raw CIC-IDS2017 data — only PyTorch weight arrays (NumPy ndarrays)
- Structure must support running Flower in simulation mode (single machine, multi-process)

Output the full directory tree with one-line comments on every file explaining its purpose.