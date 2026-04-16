# server.py — Flower server entry point and custom Strategy wiring.
#
# Responsibilities:
#   - Instantiates the custom RobustFLIDSStrategy (from aggregator.py)
#   - Configures the Flower server with the strategy and server-side evaluation function
#   - Provides the initial global model parameters to broadcast in round 0
#   - Defines the on_fit_config_fn to push hyperparameters to clients each round
#   - Defines the evaluate_fn for optional server-side global model evaluation
#
# Key functions / objects:
#   def get_initial_parameters() -> flwr.common.Parameters:
#       """Serialize a freshly initialized MLPClassifier's weights as the
#          starting global model to be broadcast before round 1."""
#
#   def get_fit_config(server_round: int) -> dict:
#       """Return per-round config dict sent to every client before local training.
#          Includes: local_epochs, learning_rate, attacker_ratio (from config.yaml),
#          and whether Byzantine injection should be active this round."""
#
#   def server_evaluate_fn(server_round, parameters, config):
#       """Optional: load a held-out evaluation set on the server side and
#          compute global macro F1-score and Attack Success Rate (ASR) after each round.
#          Results are logged to artifacts/results/ via the custom logger."""
#
# Entry point:
#   if __name__ == "__main__":
#       flwr.server.start_server(
#           server_address=...,
#           config=flwr.server.ServerConfig(num_rounds=...),
#           strategy=RobustFLIDSStrategy(...)
#       )
