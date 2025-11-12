"""
Centralised logging configuration for production reliability and debugging
"""
import logging
import os
from pathlib import Path


def setup_logging(level: str = "INFO") -> None:
    """
    Configure application-wide logging.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))
    
    # Formatter (industry standard)
    formatter = logging.Formatter(
        fmt='[%(asctime)s] %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler
    file_handler = logging.FileHandler(log_dir / "backtest.log")
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
