# FL-IDS Final Research & Experiment Plan

> **Current State (2026-06-04):** All code logic, test suites, and notebooks are implemented. Phase 1 (clean rounds) is completed, achieving a Macro F1 of **0.5435**. The centralized baseline achieved **0.7463**. 

This plan details the remaining steps required to execute the attack scenarios, close the Macro F1 performance gap, and finalize the data for your capstone report.

---

## Stage 1 — The Big Phase 2 Experiment Sweep

> [!IMPORTANT]
> The automation script `run_all_experiments.py` is fully prepared for this.

We will run a comprehensive comparison matrix to prove that our custom strategies outperform traditional baselines under Byzantine attacks (label flipping).

**Experiments to execute:**
1. **RobustFL (Variant A) Sweep:** Run at `10%`, `30%`, and `50%` attacker ratios to show how the defense scales against massive poisoning.
2. **Strategy Comparison:** At the `30%` attacker ratio, run:
   - `FedAvg` (Shows expected model collapse without defense)
   - `Trimmed Mean` (Classical robust baseline)
   - `Krum` (Classical robust baseline)
   - `RobustFL (Variant A)` (Our Cosine+MAD approach)
   - `SSFG (Variant C)` (Our Spectral Filtering approach)

*Execution:* This takes roughly ~7 hours on CPU. It is highly recommended to run this overnight.

---

## Stage 2 — Model Improvement (Closing the F1 Gap)

Currently, the FL model scores **0.5435** while the centralized model scores **0.7463**. This ~20% gap is common in Non-IID Federated Learning, but we can attempt to close it with targeted learning improvements.

**Experiments to run to improve scores:**
1. **Class-Weighted Loss:** The centralized model uses inverse-frequency weighting to learn rare classes. We must implement a similar `weighted_loss` function in `client.py`'s `fit()` method to prevent the model from ignoring minority attack classes.
2. **Increase Local Epochs:** Increase `local_epochs` from `3` to `5`. This allows clients to specialize more deeply on their local data shards per round.
3. **Increase Total Rounds:** Increase `num_rounds` from `30` to `50` to allow the global model more time to converge on rare classes.

---

## Stage 3 — Parameter Finalization & Ablation Studies

Based on federated learning research, our current hyperparameters are sound, but we will test variations to find the absolute optimum for the final report.

| Parameter | Current Value | Test Variations | Research Rationale |
|---|---|---|---|
| `defense.mad_threshold` | `-3.0` | `-2.0`, `-2.5` | `-3.0` is very conservative (3-sigma). Under heavy 30% attacks, a tighter `-2.0` threshold might catch stealthy attackers better. |
| `defense.temperature` | `2.0` | `1.0`, `5.0` | Higher temperature (e.g., 5.0) sharpens the Softmax, strongly amplifying good clients while crushing attackers. Lower (1.0) creates smoother averaging. |
| `data.alpha_dirichlet` | `0.5` | `1.0` | `0.5` creates extreme IoT heterogeneity. Testing `1.0` will prove whether the F1 gap is solely caused by data skew or client learning rates. |
| `defense.ema_momentum` | `0.9` | `0.7` | `0.9` provides high stability but reacts slowly. `0.7` allows the Trust Score to plummet faster the moment an attack starts in Round 11. |

---

## Stage 4 — Alternate Threat Vectors (Variant B)

We need to test different attack approaches and our secondary defense variant.
1. **Backdoor Trigger Attack:** Change `attack_type` from `label_flip` to `backdoor` in `config.yaml`. This tests if the defense can catch targeted data poisoning rather than just blatant label corruption.
2. **Test Variant B (AE-Scorer):** We have the Autoencoder scorer implemented (`ae_scorer.py`). We will run an experiment where we swap out the MAD scorer for the AE Scorer to compare Reconstruction Error vs. Cosine Similarity.

---

## Stage 5 — Notebook Visualizations & Final Report

Once all CSV results are collected in `artifacts/results/`, we will execute the Jupyter Notebooks we prepared in `notebooks/`.

**Key Visuals for the Report:**
- **`05_attack_analysis.ipynb`**: Shows how RobustFL maintains high F1-Scores despite 10%, 30%, and 50% poisoning.
- **`06_strategy_comparison.ipynb`**: The centerpiece of the paper — a single chart proving RobustFL and SSFG outperform FedAvg, Krum, and TrimmedMean.
- **`07_aggregator_internals.ipynb`**: Beautiful heatmaps showing how the internal MAD math and Capped Simplex algorithm actively zero-out malicious client weights in real-time.

---

## Proposed Execution Order
1. Run the `run_all_experiments.py` script overnight.
2. Review Notebook 06 to see how we compare to Krum/TrimmedMean.
3. If Macro F1 is still too low, we implement Class-Weighted Loss (Stage 2) and re-run.
4. Run ablation studies on `temperature` and `mad_threshold` (Stage 3).
5. Extract final plots for the Capstone Document.
