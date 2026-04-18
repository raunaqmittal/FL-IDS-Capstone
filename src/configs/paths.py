from pathlib import Path

PROJECT_ROOT   = Path(__file__).resolve().parents[2]

ARTIFACTS_DIR  = PROJECT_ROOT / "artifacts"
RAW_DIR        = ARTIFACTS_DIR / "raw"
PREPROCESSED_DIR = ARTIFACTS_DIR / "preprocessed"
DATA_DIR       = ARTIFACTS_DIR / "data"          # Non-IID .npz client partitions
MODELS_DIR     = ARTIFACTS_DIR / "models"
RESULTS_DIR    = ARTIFACTS_DIR / "results"
PLOTS_DIR      = ARTIFACTS_DIR / "plots"


def ensure_dirs() -> None:
    for d in [RAW_DIR, PREPROCESSED_DIR, DATA_DIR, MODELS_DIR, RESULTS_DIR, PLOTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)
