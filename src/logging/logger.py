# logger.py — Custom logging setup for the FL-IDS project.
#
# Provides a centralized, round-aware logger that writes to both
# console (stdout) and a persistent log file in artifacts/results/.
#
# Design goals:
#   - Every log line includes: timestamp, FL round number, component name, level
#   - Defense decisions (client flagged, trust score changes) are logged at INFO
#   - Mathematical debug data (MAD values, similarity matrices) logged at DEBUG
#   - Round summaries (global F1, ASR, num_flagged) logged at INFO
#
# Key functions / classes to implement:
#
#   def setup_logger(
#       name: str = "FL-IDS",
#       log_level: str = "INFO",
#       log_file: str = "artifacts/results/fl_ids.log"
#   ) -> logging.Logger:
#       """
#       Configure and return a Python logger with:
#         - StreamHandler for console output (colored via optional colorlog)
#         - FileHandler for persistent log file
#         - Custom formatter: [%(asctime)s | Round %(round)s | %(name)s] %(levelname)s: %(message)s
#       """
#
#   class RoundAdapter(logging.LoggerAdapter):
#       """Adapter that automatically injects the current FL round number
#          into every log record's 'extra' dict."""
#       def __init__(self, logger, round_num: int = 0): ...
#       def set_round(self, round_num: int): ...
#
#   def get_logger(name: str = "FL-IDS") -> logging.Logger:
#       """Retrieve the project-wide logger singleton."""
