import numpy as np
from flwr.common import ndarrays_to_parameters

from src.components.model.model import MLPClassifier, get_model_parameters
from src.components.server.aggregator import (
    RobustFLIDSStrategy,
    compute_layer_wise_cosine_similarity,
    compute_mad_scores,
    extract_final_layer,
    project_capped_simplex,
    temperature_scaled_softmax,
)

INPUT_DIM = 57
HIDDEN_DIMS = [256, 128, 64]
NUM_CLASSES = 27
NUM_CLIENTS = 5


def make_model_ndarrays():
    model = MLPClassifier(INPUT_DIM, HIDDEN_DIMS, NUM_CLASSES)
    return get_model_parameters(model)


# -- Test 1: extract_final_layer
print("Test 1: extract_final_layer")
ndarrays = make_model_ndarrays()
vec = extract_final_layer(ndarrays)
expected_size = NUM_CLASSES * HIDDEN_DIMS[-1]
assert vec.shape == (expected_size,), f"Expected ({expected_size},), got {vec.shape}"
print(f"  shape={vec.shape}  [OK]")


# -- Test 2: cosine similarity matrix
print("\nTest 2: compute_layer_wise_cosine_similarity")
final_layers = np.stack([extract_final_layer(make_model_ndarrays()) for _ in range(NUM_CLIENTS)])
sim_matrix = compute_layer_wise_cosine_similarity(final_layers)
assert sim_matrix.shape == (NUM_CLIENTS, NUM_CLIENTS)
assert np.allclose(np.diag(sim_matrix), 1.0, atol=1e-5), "Diagonal must be 1.0"
print(f"  shape={sim_matrix.shape}, diagonal_range=[{np.diag(sim_matrix).min():.4f}, {np.diag(sim_matrix).max():.4f}]  [OK]")


# -- Test 3: MAD scores shape
print("\nTest 3: compute_mad_scores")
mad_scores = compute_mad_scores(sim_matrix)
assert mad_scores.shape == (NUM_CLIENTS,)
print(f"  scores={mad_scores.round(4)}  [OK]")


# -- Test 4: MAD detects obvious outlier
print("\nTest 4: MAD outlier detection")
base_vec = np.ones(100)
stacked = np.stack([base_vec] * 4 + [np.random.randn(100) * 10])
scores = compute_mad_scores(compute_layer_wise_cosine_similarity(stacked))
assert scores[4] < scores[:4].mean(), "Outlier should score lower than benign clients"
print(f"  outlier={scores[4]:.4f}, benign_avg={scores[:4].mean():.4f}  [OK]")


# -- Test 5: temperature softmax
print("\nTest 5: temperature_scaled_softmax")
raw = np.array([1.0, 0.5, -0.5, -1.0, 2.0])
weights = temperature_scaled_softmax(raw, temperature=2.0)
assert abs(weights.sum() - 1.0) < 1e-6, f"Must sum to 1, got {weights.sum()}"
assert weights.argmax() == raw.argmax()
print(f"  weights={weights.round(4)}, sum={weights.sum():.6f}  [OK]")


# -- Test 6: capped simplex projection
print("\nTest 6: project_capped_simplex")
v = np.array([0.9, 0.05, 0.03, 0.01, 0.01])
cap_t = 1.0 / (NUM_CLIENTS - 1)
result = project_capped_simplex(v, cap_t)
assert abs(result.sum() - 1.0) < 1e-5, f"Must sum to 1, got {result.sum()}"
assert (result <= cap_t + 1e-6).all(), f"All weights must be <= cap_t={cap_t:.4f}"
print(f"  result={result.round(4)}, sum={result.sum():.6f}, max={result.max():.4f}  [OK]")


# -- Test 7: strategy init
print("\nTest 7: RobustFLIDSStrategy __init__")
strategy = RobustFLIDSStrategy(initial_parameters=ndarrays_to_parameters(make_model_ndarrays()))
assert strategy.ema_momentum > 0
assert strategy.temperature > 0
assert strategy.cap_t > 0
assert isinstance(strategy.reputation_scores, dict)
print(f"  cap_t={strategy.cap_t:.4f}, ema_momentum={strategy.ema_momentum}, temperature={strategy.temperature}  [OK]")


# -- Test 8: EMA reputation across rounds
print("\nTest 8: _update_ema_reputation")
client_ids = ["0", "1", "2", "3", "4"]
mad_input = np.array([0.5, 0.4, 0.3, -0.1, -5.0])
strategy._update_ema_reputation(client_ids, mad_input)
rep = strategy._update_ema_reputation(client_ids, mad_input)
assert rep[4] < rep[:4].mean(), "Consistently bad client must have lower reputation"
assert all(cid in strategy.reputation_scores for cid in client_ids)
print(f"  rep={rep.round(4)}, bad_client={rep[4]:.4f}  [OK]")


print("\nAll aggregator tests passed.")
