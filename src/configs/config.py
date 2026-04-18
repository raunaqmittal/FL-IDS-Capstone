from pathlib import Path
import yaml

CONFIG_YAML = Path(__file__).resolve().parent / "config.yaml"

with open(CONFIG_YAML, "r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)
