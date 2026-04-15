# paths.py — Centralized path constants for the entire FL-IDS project.
#
# All file system paths are defined here to avoid scattered hardcoded strings.
# Every module imports paths from this single source of truth.
#
# Key constants to define:
#
#   # Project root (auto-detected relative to this file's location)
#   PROJECT_ROOT = Path(__file__).resolve().parents[2]
#
#   # ─── Artifacts ─────────────────────────────────────
#   ARTIFACTS_DIR      = PROJECT_ROOT / "artifacts"
#   MODELS_DIR         = ARTIFACTS_DIR / "models"         # Saved .pth checkpoints per round
#   RESULTS_DIR        = ARTIFACTS_DIR / "results"        # CSV logs (F1, ASR, trust scores)
#   PLOTS_DIR          = ARTIFACTS_DIR / "plots"          # Saved matplotlib/seaborn figures
#   DATA_DIR           = ARTIFACTS_DIR / "data"           # Processed Non-IID .npz partitions
#
#   # ─── Raw Data (user must place CIC-IDS2017 CSVs here) ──
#   RAW_DATA_DIR       = PROJECT_ROOT / "data" / "raw"
#
#   # ─── Source Code ───────────────────────────────────
#   SRC_DIR            = PROJECT_ROOT / "src"
#   COMPONENTS_DIR     = SRC_DIR / "components"
#   CONFIGS_DIR        = SRC_DIR / "configs"
#   CONFIG_YAML_PATH   = CONFIGS_DIR / "config.yaml"
#
#   # ─── Logs ──────────────────────────────────────────
#   LOG_FILE           = RESULTS_DIR / "fl_ids.log"
#
#   def ensure_dirs() -> None:
#       """Create all artifact directories if they don't exist."""
