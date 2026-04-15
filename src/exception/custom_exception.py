# custom_exception.py — Custom exception hierarchy for FL-IDS error handling.
#
# Provides structured, descriptive exceptions for clear debugging
# across the federated pipeline.
#
# Key exception classes to implement:
#
#   class FLIDSBaseException(Exception):
#       """Base exception for all FL-IDS errors. Includes traceback context."""
#
#   class DataLoadingError(FLIDSBaseException):
#       """Raised when CIC-IDS2017 CSV files cannot be found, parsed,
#          or contain unexpected schema (missing columns, NaN rows, etc.)."""
#
#   class DataPartitionError(FLIDSBaseException):
#       """Raised when Dirichlet partitioning produces empty or degenerate
#          client splits (e.g., a client gets 0 samples)."""
#
#   class ModelError(FLIDSBaseException):
#       """Raised on PyTorch model init failures — dimension mismatches,
#          invalid state_dict shapes during weight loading, etc."""
#
#   class AggregationError(FLIDSBaseException):
#       """Raised inside the defense pipeline when mathematical operations
#          fail (e.g., singular similarity matrix, simplex projection
#          convergence failure, division-by-zero in MAD)."""
#
#   class ClientCommunicationError(FLIDSBaseException):
#       """Raised when a Flower client fails to respond, times out,
#          or returns malformed FitRes objects."""
#
#   class ConfigurationError(FLIDSBaseException):
#       """Raised when config.yaml is missing required keys or contains
#          invalid hyperparameter values (e.g., negative learning rate)."""
