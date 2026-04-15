# model.py — PyTorch Multi-Layer Perceptron (MLP) definition for IoT edge clients.
#
# Architecture:
#   - Input layer:  ~78 neurons (CIC-IDS2017 tabular features, after feature selection ~40-50)
#   - Hidden layers: 2–3 fully-connected dense layers with ReLU activation
#   - Output layer: Softmax classifier (Benign vs. Malicious)
#
# Design constraints:
#   - Must remain lightweight (few MB) to run on resource-constrained IoT gateways
#     (e.g., NVIDIA Jetson Nano, Raspberry Pi 4)
#   - No CNN, Transformer, or attention mechanisms — strictly feedforward MLP
#   - Hidden layer sizes and dropout rate are driven by configs/config.yaml
#
# Key class to implement:
#   class MLPClassifier(nn.Module):
#       def __init__(self, input_dim, hidden_dims, output_dim, dropout_rate): ...
#       def forward(self, x): ...
#
# Helper:
#   def get_model_parameters(model) -> List[np.ndarray]:
#       """Serialize PyTorch state_dict to list of NumPy arrays for Flower."""
#
#   def set_model_parameters(model, parameters: List[np.ndarray]) -> None:
#       """Deserialize list of NumPy arrays back into a PyTorch state_dict."""
