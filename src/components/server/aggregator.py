# aggregator.py — The complete 3-part server-side Byzantine defense pipeline.
#
# This is the core research contribution of the capstone.
# ALL defense logic lives strictly here on the server side.
# The server NEVER sees raw CIC-IDS2017 data — only PyTorch weight arrays (NumPy ndarrays).
#
# ═══════════════════════════════════════════════════════════════════════════════
# DEFENSE PIPELINE (executed inside aggregate_fit every FL round):
# ═══════════════════════════════════════════════════════════════════════════════
#
#  Step 1 ─ State Deserialization
#    Convert Flower FitRes objects to Python list of NumPy ndarrays per client.
#    Use flwr.common.parameters_to_ndarrays() utility.
#
#  Step 2 ─ Gap 1 Fix: Layer-Wise Cosine Similarity + MAD Anomaly Scoring
#    - Extract ONLY the final fully-connected classification layer weights.
#      (Early feature-extraction layers are deliberately skipped — Non-IID
#       variance in those layers is legitimate and must be preserved.)
#    - Flatten each client's final layer into a 1D NumPy vector.
#    - Build a (K × last_layer_dim) matrix of all client final-layer updates.
#    - Compute pairwise cosine similarity matrix using scipy.spatial.distance.pdist
#      with metric='cosine', then convert to similarity (1 - distance).
#    - Compute per-client consensus score: c_i = np.median(sim_matrix, axis=1)
#    - Compute Median Absolute Deviation (MAD):
#        mad = np.median(np.abs(c_i - np.median(c_i)))
#    - Compute robust Z-score: M_i = 0.6745 * (c_i - np.median(c_i)) / (mad + 1e-9)
#    - Flag client i as adversarial if M_i < -3.0 (configurable threshold tau)
#    - Output: raw_scores array of shape (K,) — input to Gap 3 fix.
#
#  Step 3 ─ Gap 3 Fix: Momentum-Based EMA Trust Scoring
#    - Maintain self.reputation_scores: dict[client_id -> float] across rounds.
#    - Update rule (EMA):
#        RS_i(t) = mu_d * RS_i(t-1) + (1 - mu_d) * P_i(t)
#      where P_i(t) = positive reward if M_i > -tau, negative penalty otherwise.
#    - Optionally integrate a flip-score: penalise clients whose gradient direction
#      reverses sharply between consecutive rounds (tracked in self.prev_updates).
#    - Apply Temperature-Scaled Softmax (numerically stable via LogSumExp trick):
#        scaled = RS * temperature
#        weights = exp(scaled - max(scaled)) / sum(exp(scaled - max(scaled)))
#    - Output: trust_weights array of shape (K,) — input to Gap 2 fix.
#
#  Step 4 ─ Gap 2 Fix: Capped Simplex Projection (O(K log K))
#    - Project trust_weights onto the Sparse Unit-Capped Simplex Δ(t, l0)+:
#        Constraints: sum(w) = 1,  0 ≤ w_i ≤ cap_t,  ||w||_0 ≤ s
#      where cap_t = 1 / (K - b_f)  (b_f = max tolerated Byzantine clients)
#    - Algorithm: sort descending → binary/root-finding search for Lagrangian
#      threshold γ → apply w_i = clip(v_i - γ, 0, cap_t).
#    - Complexity: O(K log K) — avoids quadratic O(K² · d) pairwise matrix math.
#    - Malicious clients have their aggregation weight forced to exactly 0.0.
#    - Output: final_weights array of shape (K,) — used for global re-aggregation.
#
#  Step 5 ─ Global Re-Aggregation
#    - Apply final_weights to ALL layer parameters (including early layers):
#        global_params = [np.average(layer_stack, axis=0, weights=final_weights)
#                         for layer_stack in zip(*all_client_params)]
#    - Return aggregated parameters + aggregated metrics to Flower server loop.
#
# ═══════════════════════════════════════════════════════════════════════════════
# KEY FUNCTIONS TO IMPLEMENT:
# ═══════════════════════════════════════════════════════════════════════════════
#
#   def extract_final_layer(ndarrays: List[np.ndarray]) -> np.ndarray:
#       """Return the flattened final-layer weight matrix for a single client."""
#
#   def compute_layer_wise_cosine_similarity(final_layers: np.ndarray) -> np.ndarray:
#       """K×K cosine similarity matrix from a (K, D_last) matrix of final layers."""
#
#   def compute_mad_scores(sim_matrix: np.ndarray) -> np.ndarray:
#       """Compute robust Z-scores M_i using MAD. Returns (K,) float array."""
#
#   def project_capped_simplex(v: np.ndarray, cap_t: float) -> np.ndarray:
#       """O(K log K) projection onto sparse unit-capped simplex.
#          Forces severely deviated clients to exactly 0.0 aggregation weight."""
#
#   def temperature_scaled_softmax(scores: np.ndarray, temperature: float) -> np.ndarray:
#       """Numerically stable softmax with temperature scaling (LogSumExp trick)."""
#
#   def update_ema_reputation(self, client_ids, current_scores) -> np.ndarray:
#       """Update persistent EMA trust scores and return the updated array."""
#
# ═══════════════════════════════════════════════════════════════════════════════
# KEY CLASS TO IMPLEMENT:
# ═══════════════════════════════════════════════════════════════════════════════
#
#   class RobustFLIDSStrategy(flwr.server.strategy.Strategy):
#       """
#       Custom Flower Strategy subclassing flwr.server.strategy.Strategy.
#       Injects the 3-part defense pipeline inside aggregate_fit().
#       Maintains persistent state: self.reputation_scores, self.prev_updates.
#       """
#       def __init__(self, config): ...
#       def initialize_parameters(self, client_manager): ...
#       def configure_fit(self, server_round, parameters, client_manager): ...
#       def aggregate_fit(self, server_round, results, failures): ...  # MAIN DEFENSE
#       def configure_evaluate(self, server_round, parameters, client_manager): ...
#       def aggregate_evaluate(self, server_round, results, failures): ...
#       def evaluate(self, server_round, parameters): ...
