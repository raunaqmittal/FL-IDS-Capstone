# app.py — Main entry point to launch the FL-IDS simulation.
#
# Usage:
#   python app.py                          # Run with default config
#   python app.py --config src/configs/config.yaml
#   python app.py --strategy proposed      # proposed | fedavg | trimmed_mean | krum
#   python app.py --attacker_ratio 0.3
#   python app.py --mode evaluate          # Skip training, only run evaluation plots
#
# Responsibilities:
#   - Parse CLI arguments (argparse)
#   - Load FLIDSConfig from specified config path
#   - Override config fields with any CLI arguments provided
#   - Dispatch to the appropriate pipeline:
#       "train"    → src/pipelines/training_pipeline.py::run_experiment()
#       "evaluate" → src/pipelines/evaluation_pipeline.py::run_evaluation()
#       "attack"   → src/pipelines/attack_pipeline.py::run_attack_sweep()
#   - Print welcome banner with experiment parameters before starting
