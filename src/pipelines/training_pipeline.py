# training_pipeline.py — Orchestrates the complete FL experiment end-to-end.
#
# This is the primary pipeline runner. It wires together all components:
#   1. Loads config from config.yaml
#   2. Preprocesses and partitions CIC-IDS2017 data (calls data_partitioner.py)
#   3. Initializes the global PyTorch MLP model
#   4. Instantiates the custom RobustFLIDSStrategy (from aggregator.py)
#   5. Launches the Flower simulation with N clients (including Byzantine attackers)
#   6. Logs per-round metrics via evaluator.py and logger.py
#   7. Saves model checkpoints to artifacts/models/ every round
#
# Supports Flower simulation mode (single machine, multi-process via Ray).
#
# Key function to implement:
#
#   def run_experiment(config_path: str = "src/configs/config.yaml") -> None:
#       """
#       Full experiment lifecycle:
#         - Parse config
#         - Partition data (or load existing partitions from artifacts/data/)
#         - Define client_fn factory (wires in attacker logic for poisoned clients)
#         - Build strategy (RobustFLIDSStrategy or a baseline from baselines.py)
#         - Call flwr.simulation.start_simulation(
#               client_fn=client_fn,
#               num_clients=config.federated.num_clients,
#               config=flwr.server.ServerConfig(num_rounds=config.federated.num_rounds),
#               strategy=strategy,
#               client_resources={"num_cpus": 1, "num_gpus": 0.0},
#           )
#         - After simulation completes, save final global model checkpoint
#       """
#
#   if __name__ == "__main__":
#       run_experiment()
