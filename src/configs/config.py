# config.py — Configuration loader and dataclass definitions.
#
# Responsibilities:
#   - Load config.yaml from disk and parse into structured Python objects
#   - Provide typed dataclass access to all hyperparameters across the project
#   - Support CLI overrides for experiment sweeps (e.g., different attacker_ratios)
#
# Key functions / classes to implement:
#
#   @dataclass
#   class ModelConfig:
#       input_dim: int
#       hidden_dims: List[int]
#       output_dim: int
#       dropout_rate: float
#
#   @dataclass
#   class FederatedConfig:
#       num_clients: int
#       sample_fraction: float
#       num_rounds: int
#       local_epochs: int
#       learning_rate: float
#       batch_size: int
#
#   @dataclass
#   class AttackConfig: ...
#   @dataclass
#   class DefenseConfig: ...
#   @dataclass
#   class DataConfig: ...
#   @dataclass
#   class ExperimentConfig: ...
#
#   @dataclass
#   class FLIDSConfig:
#       """Master config object combining all sub-configs."""
#       model: ModelConfig
#       federated: FederatedConfig
#       attack: AttackConfig
#       defense: DefenseConfig
#       data: DataConfig
#       experiment: ExperimentConfig
#
#   def load_config(yaml_path: str = "src/configs/config.yaml") -> FLIDSConfig:
#       """Parse YAML file and return structured FLIDSConfig object."""
