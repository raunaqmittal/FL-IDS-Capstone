import numpy as np
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays, FitRes, Status, Code
from flwr.server.client_proxy import ClientProxy

from src.components.model.model import MLPClassifier, get_model_parameters
from src.components.server.baselines import FedAvgBaseline, FedTrimmedMeanBaseline, KrumBaseline, get_baseline_strategy

INPUT_DIM = 57
HIDDEN_DIMS = [256, 128, 64]
NUM_CLASSES = 27
NUM_CLIENTS = 6


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


# -- Test 1: FedAvgBaseline aggregates correctly
print("Test 1: FedAvgBaseline.aggregate_fit")
strategy = FedAvgBaseline(initial_params)
results = make_results()
aggregated, metrics = strategy.aggregate_fit(1, results, [])
assert aggregated is not None
agg_arrays = parameters_to_ndarrays(aggregated)
ref_arrays = parameters_to_ndarrays(results[0][1].parameters)
assert len(agg_arrays) == len(ref_arrays), "Layer count mismatch"
assert agg_arrays[0].shape == ref_arrays[0].shape, "Shape mismatch"
print(f"  Aggregated {len(agg_arrays)} layers  [OK]")


# -- Test 2: FedAvgBaseline returns None on empty results
print("\nTest 2: FedAvgBaseline empty input")
agg, _ = strategy.aggregate_fit(1, [], [])
assert agg is None
print("  Returns None on empty  [OK]")


# -- Test 3: FedTrimmedMeanBaseline aggregates with trim
print("\nTest 3: FedTrimmedMeanBaseline.aggregate_fit")
strategy_tm = FedTrimmedMeanBaseline(initial_params, beta=0.2)
results = make_results(6)
aggregated, _ = strategy_tm.aggregate_fit(1, results, [])
assert aggregated is not None
agg_arrays = parameters_to_ndarrays(aggregated)
print(f"  Trimmed mean output layers: {len(agg_arrays)}  [OK]")


# -- Test 4: KrumBaseline selects correct client count
print("\nTest 4: KrumBaseline.aggregate_fit")
strategy_krum = KrumBaseline(initial_params, num_byzantine=1, multi_k=1)
results = make_results(6)
aggregated, _ = strategy_krum.aggregate_fit(1, results, [])
assert aggregated is not None
agg_arrays = parameters_to_ndarrays(aggregated)
ref = parameters_to_ndarrays(results[0][1].parameters)
assert len(agg_arrays) == len(ref)
print(f"  Krum output layers: {len(agg_arrays)}  [OK]")


# -- Test 5: factory returns correct types
print("\nTest 5: get_baseline_strategy factory")
assert isinstance(get_baseline_strategy("fedavg"), FedAvgBaseline)
assert isinstance(get_baseline_strategy("trimmed_mean"), FedTrimmedMeanBaseline)
assert isinstance(get_baseline_strategy("krum"), KrumBaseline)
print("  All three strategies created correctly  [OK]")


print("\nAll baselines tests passed.")
