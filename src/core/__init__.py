"""
Core utilities and foundational modules for DataAlchemy
"""

from .config import Config
from .exceptions import (
    DataAlchemyError,
    DataValidationError,
    FileOperationError,
    FeatureEngineeringError,
    TransformationError,
    FeatureSelectionError,
    ValidationError,
    ConfigurationError,
    AgentError,
    InsufficientDataError,
    TargetNotFoundError
)
from .logging_config import logger, LogContext, TimedOperation

__all__ = [
    'Config',
    'DataAlchemyError',
    'DataValidationError',
    'FileOperationError',
    'FeatureEngineeringError',
    'TransformationError',
    'FeatureSelectionError',
    'ValidationError',
    'ConfigurationError',
    'AgentError',
    'InsufficientDataError',
    'TargetNotFoundError',
    'logger',
    'LogContext',
    'TimedOperation'
]