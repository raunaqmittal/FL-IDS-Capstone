import numpy as np
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays, FitRes, Status, Code
from flwr.server.client_proxy import ClientProxy

from src.components.model.model import MLPClassifier, get_model_parameters
from src.components.server.ssfg_aggregator import SSFGAggregator, _spectral_filter

INPUT_DIM = 57
HIDDEN_DIMS = [256, 128, 64]
NUM_CLASSES = 27
NUM_CLIENTS = 5


class _Proxy(ClientProxy):
    def __init__(self, cid): super().__init__(cid)
    def reconnect(self, *a, **kw): pass
    def get_properties(self, *a, **kw): pass
    def get_parameters(self, *a, **kw): pass
    def fit(self, *a, **kw): pass
    def evaluate(self, *a, **kw): pass


def make_results(n=NUM_CLIENTS):
    results = []
    for i in range(n):
        model = MLPClassifier(INPUT_DIM, HIDDEN_DIMS, NUM_CLASSES)
        params = ndarrays_to_parameters(get_model_parameters(model))
        fit_res = FitRes(status=Status(code=Code.OK, message=""), parameters=params, num_examples=100, metrics={})
        results.append((_Proxy(str(i)), fit_res))
    return results


initial_params = ndarrays_to_parameters(get_model_parameters(MLPClassifier(INPUT_DIM, HIDDEN_DIMS, NUM_CLASSES)))


# -- Test 1: _spectral_filter preserves shape
print("Test 1: _spectral_filter shape")
vecs = np.random.randn(NUM_CLIENTS, 1728).astype(np.float32)
filtered = _spectral_filter(vecs)
assert filtered.shape == vecs.shape, f"Expected {vecs.shape}, got {filtered.shape}"
print(f"  Shape preserved: {filtered.shape}  [OK]")


# -- Test 2: _spectral_filter with keep_ratio=1.0 is close to identity
print("\nTest 2: _spectral_filter keep_ratio=1.0 near-identity")
filtered_full = _spectral_filter(vecs, keep_ratio=1.0)
assert np.allclose(filtered_full, vecs, atol=1e-4)
print("  Full-rank filter is near-identity  [OK]")


# -- Test 3: SSFGAggregator initializes
print("\nTest 3: SSFGAggregator __init__")
strategy = SSFGAggregator(initial_parameters=initial_params)
assert strategy.cap_t > 0
assert strategy.ema_momentum > 0
assert isinstance(strategy.reputation_scores, dict)
print(f"  cap_t={strategy.cap_t:.4f}, ema_momentum={strategy.ema_momentum}  [OK]")


# -- Test 4: aggregate_fit returns valid parameters
print("\nTest 4: SSFGAggregator.aggregate_fit")
results = make_results()
aggregated, metrics = strategy.aggregate_fit(1, results, [])
assert aggregated is not None
agg_arrays = parameters_to_ndarrays(aggregated)
ref = parameters_to_ndarrays(results[0][1].parameters)
assert len(agg_arrays) == len(ref)
assert agg_arrays[0].shape == ref[0].shape
print(f"  Output: {len(agg_arrays)} layers, first shape {agg_arrays[0].shape}  [OK]")


# -- Test 5: metrics dict contains expected keys
print("\nTest 5: aggregate_fit metrics keys")
assert "flagged" in metrics
assert "min_trust" in metrics
assert "max_trust" in metrics
print(f"  flagged={metrics['flagged']}, min_trust={metrics['min_trust']:.4f}  [OK]")


# -- Test 6: reputation scores are tracked
print("\nTest 6: Reputation tracking across rounds")
strategy.aggregate_fit(2, results, [])
assert len(strategy.reputation_scores) == NUM_CLIENTS
print(f"  Tracking {len(strategy.reputation_scores)} client reputations  [OK]")


print("\nAll SSFG tests passed.")
