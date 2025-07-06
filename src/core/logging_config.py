"""
Structured logging configuration for DataAlchemy
"""
import sys
import structlog
from pathlib import Path
from typing import Optional
import logging
from .config import Config


def configure_logging(log_file: Optional[Path] = None) -> structlog.BoundLogger:
    """
    Configure structured logging for DataAlchemy
    
    Args:
        log_file: Optional path to log file
        
    Returns:
        Configured logger instance
    """
    # Set up standard logging first
    log_level = getattr(logging, Config.LOG_LEVEL, logging.INFO)
    
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        format="%(message)s",
        level=log_level,
        handlers=handlers
    )
    
    # Configure structlog processors
    processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]
    
    # Add development-friendly console renderer for non-production
    if Config.LOG_LEVEL == "DEBUG":
        processors.append(structlog.dev.ConsoleRenderer())
    else:
        processors.append(structlog.processors.JSONRenderer())
    
    # Configure structlog
    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Return a logger instance
    return structlog.get_logger()


# Global logger instance
logger = configure_logging()


class LogContext:
    """Context manager for adding context to log messages"""
    
    def __init__(self, **kwargs):
        self.context = kwargs
        self.logger = structlog.get_logger()
        self.bound_logger = None
    
    def __enter__(self):
        self.bound_logger = self.logger.bind(**self.context)
        return self.bound_logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.bound_logger.error(
                "operation_failed",
                exc_type=exc_type.__name__,
                exc_message=str(exc_val)
            )
        return False


class TimedOperation:
    """Context manager for timing operations and logging results"""
    
    def __init__(self, operation_name: str, **context):
        self.operation_name = operation_name
        self.context = context
        self.logger = structlog.get_logger()
        self.start_time = None
        self.bound_logger = None
    
    def __enter__(self):
        import time
        self.start_time = time.time()
        self.bound_logger = self.logger.bind(
            operation=self.operation_name,
            **self.context
        )
        self.bound_logger.info(f"{self.operation_name}_started")
        return self.bound_logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        import time
        duration = time.time() - self.start_time
        
        if exc_type is not None:
            self.bound_logger.error(
                f"{self.operation_name}_failed",
                duration_seconds=duration,
                exc_type=exc_type.__name__,
                exc_message=str(exc_val)
            )
        else:
            self.bound_logger.info(
                f"{self.operation_name}_completed",
                duration_seconds=duration
            )
        return False