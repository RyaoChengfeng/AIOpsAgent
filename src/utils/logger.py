"""
Logging utility module
Provides unified logging functionality
"""

import logging
import logging.handlers
import os
from pathlib import Path
from typing import Optional
from config.settings import get_config


def setup_logger(
    name: str = "AIOpsAgent",
    log_file: Optional[str] = None,
    level: str = "INFO",
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Set up a logger.

    Args:
        name: Logger name
        log_file: Path of the log file
        level: Log level
        format_string: Log format string

    Returns:
        A configured logger
    """
    # Load configuration
    if not log_file:
        log_file = get_config('logging.file', 'logs/agent.log')
    if not format_string:
        format_string = get_config(
            'logging.format',
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

    # Create log directory
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    # Avoid adding duplicate handlers
    if logger.handlers:
        return logger

    # Create formatter
    formatter = logging.Formatter(format_string)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (with rotation)
    max_size = get_config('logging.max_size', '10MB')
    backup_count = get_config('logging.backup_count', 5)

    # Parse size string
    size_bytes = _parse_size(max_size)

    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=size_bytes,
        backupCount=backup_count,
        encoding='utf-8'
    )
    file_handler.setLevel(getattr(logging, level.upper()))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def _parse_size(size_str: str) -> int:
    """
    Parse a size string.

    Args:
        size_str: Size string, e.g. '10MB', '1GB'

    Returns:
        Size in bytes
    """
    size_str = size_str.upper().strip()

    if size_str.endswith('KB'):
        return int(float(size_str[:-2]) * 1024)
    elif size_str.endswith('MB'):
        return int(float(size_str[:-2]) * 1024 * 1024)
    elif size_str.endswith('GB'):
        return int(float(size_str[:-2]) * 1024 * 1024 * 1024)
    else:
        # Default to bytes
        return int(size_str)


def get_logger(name: str = "AIOpsAgent") -> logging.Logger:
    """
    Get a logger.

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        # Set default config if no handlers found
        level = get_config('logging.level', 'INFO')
        return setup_logger(name, level=level)
    return logger


class LoggerMixin:
    """Logger mixin class"""

    @property
    def logger(self) -> logging.Logger:
        """Get logger"""
        if not hasattr(self, '_logger'):
            class_name = self.__class__.__name__
            self._logger = get_logger(f"AIOpsAgent.{class_name}")
        return self._logger


# Create default logger
default_logger = setup_logger()


def log_function_call(func):
    """
    Decorator: log function call.

    Args:
        func: Function to decorate

    Returns:
        Decorated function
    """
    def wrapper(*args, **kwargs):
        logger = get_logger()
        func_name = func.__name__
        logger.debug(f"Calling function: {func_name}")

        try:
            result = func(*args, **kwargs)
            logger.debug(f"Function {func_name} executed successfully")
            return result
        except Exception as e:
            logger.error(f"Function {func_name} failed: {e}")
            raise

    return wrapper


def log_execution_time(func):
    """
    Decorator: log function execution time.

    Args:
        func: Function to decorate

    Returns:
        Decorated function
    """
    import time

    def wrapper(*args, **kwargs):
        logger = get_logger()
        func_name = func.__name__

        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.info(f"Function {func_name} execution time: {execution_time:.2f}s")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(
                f"Function {func_name} failed (took {execution_time:.2f}s): {e}"
            )
            raise
    
    return wrapper
