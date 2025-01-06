import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
from typing import Optional
from datetime import datetime

def setup_logger(name: str, log_dir: str = "logs") -> logging.Logger:
    """
    Configure and return a logger with both file and console handlers.
    
    Args:
        name: The logger name, typically __name__
        log_dir: Directory to store log files
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Create logs directory if it doesn't exist
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)

    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )

    # File handler (with rotation)
    file_handler = RotatingFileHandler(
        filename=log_path / f"timelapse_{datetime.now().strftime('%Y%m%d')}.log",
        maxBytes=5 * 1024 * 1024,  # 5MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.DEBUG)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(logging.INFO)

    # Add handlers if they don't exist
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger

# Custom log levels
TRACE = 5
logging.addLevelName(TRACE, "TRACE")

def trace(self, message, *args, **kwargs):
    """Add trace logging level"""
    if self.isEnabledFor(TRACE):
        self._log(TRACE, message, args, **kwargs)

# Add trace method to Logger class
logging.Logger.trace = trace

class LogContext:
    """Context manager for temporary log level changes"""
    
    def __init__(self, logger: logging.Logger, level: int):
        self.logger = logger
        self.level = level
        self.previous_level = logger.level

    def __enter__(self):
        self.logger.setLevel(self.level)
        return self.logger

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.setLevel(self.previous_level)

def get_logger(name: str) -> logging.Logger:
    """Get or create a logger with the given name"""
    return logging.getLogger(name)