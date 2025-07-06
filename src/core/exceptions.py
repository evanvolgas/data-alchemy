"""
Custom exceptions for DataAlchemy
"""
from typing import Optional, Dict, Any


class DataAlchemyError(Exception):
    """Base exception for all DataAlchemy errors"""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class DataValidationError(DataAlchemyError):
    """Raised when data validation fails"""
    pass


class FileOperationError(DataAlchemyError):
    """Raised when file operations fail"""
    pass


class FeatureEngineeringError(DataAlchemyError):
    """Raised when feature engineering fails"""
    pass


class TransformationError(FeatureEngineeringError):
    """Raised when a specific transformation fails"""
    def __init__(self, transformer_name: str, message: str, 
                 details: Optional[Dict[str, Any]] = None):
        super().__init__(message, details)
        self.transformer_name = transformer_name


class FeatureSelectionError(DataAlchemyError):
    """Raised when feature selection fails"""
    pass


class ValidationError(DataAlchemyError):
    """Raised when feature validation fails"""
    pass


class ConfigurationError(DataAlchemyError):
    """Raised when configuration is invalid"""
    pass


class AgentError(DataAlchemyError):
    """Exception for any agent-related errors"""
    def __init__(self, agent_name: str, message: str, 
                 details: Optional[Dict[str, Any]] = None):
        super().__init__(f"{agent_name}: {message}", details)
        self.agent_name = agent_name


class InsufficientDataError(DataAlchemyError):
    """Raised when there's not enough data to perform an operation"""
    def __init__(self, required_rows: int, actual_rows: int, 
                 operation: str = "operation"):
        message = f"Insufficient data for {operation}: required {required_rows} rows, got {actual_rows}"
        super().__init__(message, {
            "required_rows": required_rows,
            "actual_rows": actual_rows,
            "operation": operation
        })


class TargetNotFoundError(DataAlchemyError):
    """Raised when specified target column is not found"""
    def __init__(self, target_column: str, available_columns: list):
        message = f"Target column '{target_column}' not found in data"
        super().__init__(message, {
            "target_column": target_column,
            "available_columns": available_columns
        })