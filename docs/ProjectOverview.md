markdown_content = """# System Architecture & Pipeline: Robust FL-IDS for IoT Edge Gateways

This document serves as the master blueprint for the Federated Learning Intrusion Detection System (FL-IDS) capstone project. It synthesizes the finalized pipeline, models, frameworks, and specific mathematical defense mechanisms developed to secure resource-constrained IoT environments against Byzantine attacks.

---

## 1. Project Overview & Threat Model

* **Objective:** Develop a robust, decentralized Intrusion Detection System that trains collaboratively across IoT edge devices without sharing raw data, while actively defending against compromised nodes attempting to poison the global model.
* **Data Structure:** Tabular network traffic statistics (e.g., CIC-IDS2017 dataset). Features include packet size, flow duration, and destination ports (approx. 78 features).
* **Threat Model (Byzantine Failure):** * **Attack Vector:** Data Poisoning and Label-Flipping. 
    * **Scenario:** An adversary compromises an edge gateway and intentionally alters the training data (e.g., labeling a DDoS attack as "Benign") to force the global model to learn a backdoor or misclassify specific traffic.
    * **Constraint:** The central server is completely blind to the clients' raw data and must detect the attack purely by analyzing the mathematical weight updates sent by the clients.

---

## 2. Client-Side Architecture (IoT Edge Gateways)

The clients represent physical hardware (like Cisco Edge Routers or NVIDIA Jetson Nano gateways) deployed in diverse environments (e.g., hospitals, factories, universities).

* **Deep Learning Model:** Multi-Layer Perceptron (MLP) built in **PyTorch**.
    * *Input Layer:* ~78 Neurons (matching the tabular network features).
    * *Hidden Layers:* 2-3 dense layers with ReLU activation to extract behavioral patterns.
    * *Output Layer:* Softmax classifier (Benign vs. Malicious).
* **Why an MLP?** It is highly optimized for tabular data and requires only a few megabytes of memory, making it perfectly suited for resource-constrained IoT devices where heavy models (CNNs/Transformers) would fail.
* **Data Environment:** The client data is highly **Non-IID** (Non-Independent and Identically Distributed). A smart factory gateway sees completely different normal traffic than an office gateway. 

---

## 3. Orchestration Layer (Flower Framework)

**Flower (`flwr`)** acts as the "nervous system" of the project.
* **Role:** It manages the communication loops. It broadcasts the global model to the clients, waits for them to train locally using PyTorch, and collects their updated weights.
* **Custom Strategy:** Instead of using Flower's default `FedAvg` (which blindly trusts all updates), we will write a custom Python class extending `flwr.server.strategy.Strategy`. We will inject our 3-part defensive math directly into the `aggregate_fit` method to filter the weights *before* they are averaged.

---

## 4. The 3-Part Server-Side Defense Pipeline

This is the core capstone contribution. It addresses the three critical gaps in current Federated Learning literature.

### Gap 1: The "Non-IID vs. Malicious" Dilemma
*Current systems confuse stealthy attackers with naturally unique IoT gateways, accidentally deleting good data.*
* **Solution:** **Layer-Wise Cosine Similarity & Robust Statistics (MAD)**
    * *How it works:* A label-flipping attack must drastically alter the final classification layer to implant a backdoor, while normal Non-IID traffic mostly affects the early feature-extraction layers. 
    * *Implementation:* The Flower server slices the PyTorch weights and completely ignores the early layers. It runs **Cosine Similarity** purely on the final decision layer to check the "angle" of the update. It then uses **Median Absolute Deviation (MAD)** to find the mathematical consensus and flag outliers.
    * *(Note: We also explored a **Server-Side Autoencoder** that reconstructs the 10,000+ PyTorch weights to find poisoning spikes via Reconstruction Error. The Layer-Wise Similarity is chosen for its ultimate lightweight speed, but both are valid server-side filters).*

### Gap 2: The Computational Bottleneck
*Robust defenses usually require massive $O(K^2)$ matrix math, which creates severe latency and crashes servers when scaling to hundreds of IoT clients.*
* **Solution:** **Optimization-Based Filtering (Capped Simplex Projection)**
    * *How it works:* Instead of running exhaustive pairwise distance calculations for every parameter, the server treats the aggregation weights as a mathematical optimization problem.
    * *Implementation:* It takes the similarity scores, sorts them ($O(K \log K)$ speed), and runs them through a "Capped Simplex" algorithm. This acts like a pair of scissors, instantly forcing the aggregation weight of highly deviated, malicious clients to exactly `0.0`, while ensuring no single good client gets more than a safe "capped" amount of influence.

### Gap 3: Rigid Adaptation Rules
*Existing systems use static "accept or reject" rules. If a client fails one round, it is permanently deleted, destroying long-term learning.*
* **Solution:** **Momentum-Based Trust Scoring (Dynamic Adaptation)**
    * *How it works:* The system acts like a credit score. It maintains a persistent memory of how trustworthy a client has been over time.
    * *Implementation:* We use an **Exponential Moving Average (EMA)**. If a client submits a bad update, their trust score drops, but they aren't deleted. A **Temperature-Scaled Softmax** function converts these continuous scores into percentages. A persistent attacker's voting power will smoothly scale down to 0%, while a good client with a temporary sensor glitch will recover quickly.

---

## 5. Execution & Justifications (One-Year Scope)

To justify the one-year capstone timeline, the project development is split into several heavy engineering phases:
1.  **Attack Engineering:** We must first successfully program and deploy targeted Label-Flipping and Data Poisoning attacks on the Flower clients to create the threat environment.
2.  **Data Engineering:** We must write custom scripts to partition the millions of rows of CIC-IDS2017 data into realistic, highly unbalanced Non-IID chunks to simulate different physical IoT locations.
3.  **Custom Aggregator Development:** Writing the continuous optimization math (Simplex Projection and EMA) inside Flower's `aggregate_fit` function requires extensive debugging and mathematical tuning to ensure the dimensions align correctly before PyTorch re-averages the model.
4.  **Hardware Profiling:** We must profile the CPU and memory latency of the central server to mathematically prove our defense is lightweight enough for actual edge deployment.
"""

with open('/mnt/data/claude.md', 'w') as f:
    f.write(markdown_content)

print("claude.md generated successfully.")


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