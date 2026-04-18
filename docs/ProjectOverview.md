
# 🛡️ Project Blueprint: Robust Federated Learning IDS for IoT Networks

## 1. Project Overview & Environment
* **Domain:** Cybersecurity, Intrusion Detection Systems (IDS), Internet of Things (IoT).
* **Architecture:** Decentralized Federated Learning (FL).
* **Data:** CIC-IDS2017 (Tabular network traffic data features like packet size, flow duration).
* **Data Distribution:** Non-IID (Non-Independent and Identically Distributed) to simulate highly diverse, realistic IoT edge traffic.
* **Threat Model:** Byzantine adversaries executing targeted **Label-Flipping** and **Data Poisoning** attacks from compromised edge clients.

## 2. Core Technology Stack
* **Machine Learning Framework:** `PyTorch`
* **Federated Orchestration Framework:** `Flower (flwr)`
* **Client Local Model:** A lightweight **Multi-Layer Perceptron (MLP)**.
  * *Why:* Perfect for fast tabular data classification, highly explainable, and small enough (a few MBs) to run on resource-constrained IoT Edge Gateways (e.g., Raspberry Pi 4, Jetson Nano).
  * *Note:* LLMs/Ollama and heavy CNNs are strictly out of scope due to massive memory overhead.
* **Server Infrastructure:** A central Flower aggregator with a custom subclassed `Strategy` (overriding the default `FedAvg`).

--## 3. The 3 Core Research Gaps & Finalized Solutions
Traditional FL (like FedAvg) is blindly trusting, and robust filters (like Krum) are too computationally heavy and accidentally delete good, unique IoT data. This architecture solves these flaws through a **modular 3-part Server-Side Defense Pipeline** with a swappable anomaly scoring component.

### Gap 1: The "Non-IID vs. Malicious" Dilemma
* **Problem:** Normal clients have highly unique traffic (Non-IID). If we evaluate the whole model, good clients look like hackers and get their data accidentally deleted.
* **Solution: Layer-Wise Anomaly Scoring (two variants — both will be implemented and compared)**
  * **Mechanism:** The server isolates only the **final classification layer** of each client's weights (early layers are skipped — Non-IID variance there is legitimate). The anomaly score for each client is computed using one of two swappable modules:
  * **Variant A — AL-CMT (primary):** Computes **Cosine Similarity** of final-layer weights across clients, then applies **Median Absolute Deviation (MAD)** to flag outliers. No server data required. Fully privacy-preserving.
  * **Variant B — CS-ARF (secondary):** A lightweight **Server-Side Autoencoder** trained progressively on trusted client weight vectors during clean rounds (Phase 1). Scores each client by their **Reconstruction Error** — poisoned updates have anomalous weight distributions the AE cannot reconstruct. Implemented after Phase 1 data is available.
  * **Variant C — SSFG (tertiary comparison):** Singular Spectrum Filtering. Instead of scoring and rejecting individual clients, it stacks all client updates into a matrix and runs **Truncated SVD** to mathematically project components onto the clean "benign consensus" subspace. This retains useful non-poisoned information even from compromised clients.
  * Steps 3 and 4 (EMA + Simplex) are **shared by Variants A and B**, while **Variant C is an independent matrix-filtering approach**, providing a comprehensive research comparison.

### Gap 2: Computational Bottleneck for IoT
* **Problem:** Standard robust filters require $O(K^2)$ massive matrix distance calculations across the whole model, causing severe server latency.
* **Solution: Optimization-Based Filtering (shared by both variants)**
  * **Mechanism:** We abandon $O(K^2)$ distance grids. Instead, we use a continuous root-finding algorithm.
  * **Math:** We project the anomaly scores onto a **Sparse Unit-Capped Simplex**. This is an $O(K \log K)$ sorting algorithm that instantly caps the maximum influence any client can have, and forces the aggregation weights of severe attackers to exactly `0.0`.

### Gap 3: Rigid Adaptation Rules
* **Problem:** Existing defenses use rigid "accept/reject" thresholds. If a good node drops a packet, it gets permanently banned.
* **Solution: Momentum-Based Trust Scoring (shared by both variants)**
  * **Mechanism:** We assign a continuous "Trust Score" to every client that persists across federated rounds.
  * **Math:** We use an **Exponential Moving Average (EMA)**. A client's score is a mix of their historical momentum and their current round score. We then apply a **Temperature-Scaled Softmax** to convert these raw scores into percentage-based aggregation weights. A stealthy attacker's influence smoothly scales down to zero over time.

---

## 4. The Server-Side Execution Pipeline (Step-by-Step)
When the central Flower server runs the `aggregate_fit` method, it executes this exact pipeline:

1. **Deserialize:** Receive raw PyTorch weights from IoT clients and convert them to NumPy arrays.
2. **Layer-Decoupling:** Slice out only the final classification layer weights from each client.
3. **Score Generation ← SWAPPABLE MODULE:**
   * **Variant A (AL-CMT):** Cosine Similarity matrix → per-client MAD robust Z-score.
   * **Variant B (CS-ARF):** Feed final-layer vectors into trained Server-Side AE → per-client Reconstruction Error as anomaly score.
4. **Momentum Update (Gap 3):** Feed the current round's score into the EMA persistent state dictionary to update the client's historical Trust Score.
5. **Simplex Projection (Gap 2):** Pass the EMA Trust Scores through the Capped Simplex algorithm to convert them into strictly bounded aggregation weights (ensuring malicious weights hit `0.0`).
6. **Global Aggregation:** Apply final weights to the *entire* PyTorch model using `numpy.average` and broadcast the secured global model for the next round.

---

## 5. Strict Project Boundaries (Do NOT Suggest These)
If generating code or architecture for this project, absolutely avoid the following:
* **Blockchain or Homomorphic Encryption:** Too heavy for IoT.
* **Heavy Secondary Models on Clients:** Clients only run the PyTorch MLP. The Server-Side AE (Variant B) runs *only on the server*, never on IoT clients.
* **Raw Data Sharing:** The server never sees the 78 columns of CIC-IDS2017 data; it only sees and analyzes the PyTorch weight vectors.
* **Suggesting only one defense:** Variants A (Cosine+MAD), B (AE), and C (SSFG) must be implemented for the comparative evaluation. Do not drop them.