# client.py — Flower client class representing a single IoT Edge Gateway.
#
# Responsibilities:
#   - Wraps the PyTorch MLP (from model.py) inside a flwr.client.NumPyClient subclass
#   - Executes local training epochs on the client's private CIC-IDS2017 partition
#   - Returns updated model weights (as NumPy arrays) + training metrics to the server
#   - Does NOT share raw data — only serialized PyTorch weight tensors leave the device
#
# Key class to implement:
#   class FLIDSClient(flwr.client.NumPyClient):
#       def __init__(self, cid, train_loader, val_loader, model, config): ...
#
#       def get_parameters(self, config) -> List[np.ndarray]:
#           """Return current local model weights as NumPy arrays."""
#
#       def fit(self, parameters, config) -> Tuple[List[np.ndarray], int, dict]:
#           """
#           1. Load global weights into local model
#           2. Run local training loop (PyTorch optimizer + CrossEntropyLoss)
#           3. Return updated weights, num_examples, and training metrics dict
#           Note: If this client is flagged as 'poisoned', attacker.py logic is
#                 called here to flip labels before training.
#           """
#
#       def evaluate(self, parameters, config) -> Tuple[float, int, dict]:
#           """Evaluate the global model on the local validation split."""
#
# Factory function:
#   def client_fn(cid: str) -> FLIDSClient:
#       """Instantiate the correct client given a client ID string.
#          Loads the pre-partitioned Non-IID data file for this client
#          from artifacts/data/ using paths from configs/paths.py."""
