import numpy as np
from src.components.server.ae_scorer import AEScorer

INPUT_DIM = 64
N_CLIENTS = 5


# -- Test 1: fit runs without error
print("Test 1: AEScorer.fit")
ae = AEScorer(input_dim=INPUT_DIM, hidden_factor=4, train_epochs=5)
clean_vecs = np.random.randn(N_CLIENTS, INPUT_DIM).astype(np.float32)
ae.fit(clean_vecs)
assert ae.is_fitted
print("  Fitted without error  [OK]")


# -- Test 2: score returns correct shape
print("\nTest 2: AEScorer.score shape")
scores = ae.score(clean_vecs)
assert scores.shape == (N_CLIENTS,), f"Expected ({N_CLIENTS},), got {scores.shape}"
print(f"  Scores shape: {scores.shape}  [OK]")


# -- Test 3: anomalous vectors score lower than clean ones
print("\nTest 3: AEScorer detects anomaly")
anomalous = np.random.randn(1, INPUT_DIM).astype(np.float32) * 100.0
clean_scores = ae.score(clean_vecs)
anomaly_scores = ae.score(anomalous)
assert anomaly_scores[0] < clean_scores.mean(), (
    f"Anomaly score {anomaly_scores[0]:.4f} should be < clean mean {clean_scores.mean():.4f}"
)
print(f"  Anomaly score={anomaly_scores[0]:.4f}, clean mean={clean_scores.mean():.4f}  [OK]")


# -- Test 4: score before fit returns zeros (safe fallback)
print("\nTest 4: AEScorer.score before fit returns zeros")
ae2 = AEScorer(input_dim=INPUT_DIM)
fallback = ae2.score(clean_vecs)
assert (fallback == 0).all()
print("  Returns zeros before fit  [OK]")


# -- Test 5: scores are negative (convention: high anomaly = most negative)
print("\nTest 5: Scores are negative (high error = low score)")
assert (scores <= 0).all(), f"All scores should be <= 0, got {scores}"
print(f"  All scores <= 0  [OK]")


print("\nAll AEScorer tests passed.")
