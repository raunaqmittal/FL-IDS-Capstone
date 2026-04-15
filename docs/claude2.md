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

---

## 3. The 3 Core Research Gaps & Finalized Solutions
Traditional FL (like FedAvg) is blindly trusting, and robust filters (like Krum) are too computationally heavy and accidentally delete good, unique IoT data. This architecture solves these flaws through a 3-part Server-Side Defense Pipeline.

### Gap 1: The "Non-IID vs. Malicious" Dilemma
* **Problem:** Normal clients have highly unique traffic (Non-IID). If we evaluate the whole model, good clients look like hackers and get their data accidentally deleted.
* **Solution: Layer-Wise Similarity & Robust Statistics**
  * **Mechanism:** The server isolates the PyTorch parameter updates. It ignores the early feature-extraction layers (which hold the good, unique IoT data) and only analyzes the **final classification layer** (which an attacker *must* alter to flip labels). 
  * **Math:** It calculates the **Cosine Similarity** (or optionally uses a Server-Side Autoencoder Reconstruction Error) of these final layers. It then uses **Median Absolute Deviation (MAD)** to establish a robust consensus and catch outliers without being skewed by the attackers.

### Gap 2: Computational Bottleneck for IoT
* **Problem:** Standard robust filters require $O(K^2)$ massive matrix distance calculations across the whole model, causing severe server latency.
* **Solution: Optimization-Based Filtering**
  * **Mechanism:** We abandon $O(K^2)$ distance grids. Instead, we use a continuous root-finding algorithm.
  * **Math:** We project the similarity scores onto a **Sparse Unit-Capped Simplex**. This is an $O(K \log K)$ sorting algorithm that acts like a pair of scissors—it instantly caps the maximum influence any client can have, and forces the aggregation weights of severe attackers to exactly `0.0`.

### Gap 3: Rigid Adaptation Rules
* **Problem:** Existing defenses use rigid "accept/reject" thresholds. If a good node drops a packet, it gets permanently banned. 
* **Solution: Momentum-Based Trust Scoring**
  * **Mechanism:** We assign a continuous "Trust Score" to every client that persists across federated rounds. 
  * **Math:** We use an **Exponential Moving Average (EMA)**. A client's score is a mix of their historical momentum and their current round score. We then apply a **Temperature-Scaled Softmax** to convert these raw scores into percentage-based aggregation weights. A stealthy attacker’s influence smoothly scales down to zero over time.

---

## 4. The Server-Side Execution Pipeline (Step-by-Step)
When the central Flower server runs the `aggregate_fit` method, it executes this exact pipeline:

1. **Deserialize:** Receive raw PyTorch weights from IoT clients and convert them to NumPy arrays.
2. **Layer-Decoupling (Gap 1):** Slice out only the final classification layer.
3. **Score Generation:** Calculate the anomaly score (via Cosine Similarity distance from the MAD consensus, or AE Reconstruction Error).
4. **Momentum Update (Gap 3):** Feed the current round's score into the EMA persistent state dictionary to update the client's historical Trust Score.
5. **Simplex Projection (Gap 2):** Pass the EMA Trust Scores through the Capped Simplex algorithm to convert them into strictly bounded aggregation weights (ensuring malicious weights hit `0.0`).
6. **Global Aggregation:** Apply these final weights to the *entire* PyTorch model using `numpy.average` and broadcast the newly secured global model for the next round.

---

## 5. Strict Project Boundaries (Do NOT Suggest These)
If generating code or architecture for this project, absolutely avoid the following:
* **Blockchain or Homomorphic Encryption:** Too heavy for IoT.
* **Heavy Secondary Models on Clients:** Clients only run the PyTorch MLP. All complex defense filtering (Cosine Similarity, MAD, EMA) happens *strictly* on the central server.
* **Raw Data Sharing:** The server never sees the 78 columns of CIC-IDS2017 data; it only sees and analyzes the PyTorch weights.