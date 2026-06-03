import logging
import sys
from pathlib import Path

# Ensure src is in the path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.configs.config import CONFIG
from src.pipelines.training_pipeline import run_experiment
from src.pipelines.attack_pipeline import run_attack_sweep

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("Starting full Phase 2 experiment suite...")
    
    # 1. Sweep for RobustFL across configured attacker ratios (10%, 30%, 50%)
    logging.info("=== Running RobustFL (Variant A) Sweep ===")
    run_attack_sweep(strategy_name="robust")
    
    # 2. Run baselines and SSFG at 30% attacker ratio for comparison
    target_ratio = 0.30
    logging.info(f"=== Running Baselines & SSFG at {target_ratio*100}% Attacker Ratio ===")
    CONFIG["attack"]["attacker_ratio"] = target_ratio
    
    strategies = ["fedavg", "trimmed_mean", "krum", "ssfg"]
    for strategy in strategies:
        logging.info(f"--- Running {strategy} at {target_ratio*100}% ---")
        suffix = f"_{strategy}_ratio_{int(target_ratio * 100):02d}pct"
        run_experiment(results_suffix=suffix, strategy_name=strategy)
        
    logging.info("All experiments completed successfully. You can now run the notebooks in /notebooks to visualize the results.")

if __name__ == "__main__":
    main()
