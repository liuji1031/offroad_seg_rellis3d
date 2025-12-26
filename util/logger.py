"""Logging utilities for the offroad_det_seg_rellis package.

This module provides a centralized logging configuration with proper formatting
including log level, timestamp, filename, and line number.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str,
    level: int = logging.INFO,
    log_file: Optional[str | Path] = None,
    format_string: Optional[str] = None,
) -> logging.Logger:
    """Set up a logger with custom formatting.
    
    Args:
        name: Logger name (typically __name__)
        level: Logging level (default: INFO)
        log_file: Optional file path to write logs to
        format_string: Optional custom format string. If None, uses default format.
        
    Returns:
        Configured logger instance
        
    Default format includes:
        - Log level
        - Date and time
        - Filename
        - Line number
        - Function name
        - Message
    """
    logger = logging.getLogger(name)
    
    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger
    
    logger.setLevel(level)
    
    # Default format: LEVEL - YYYY-MM-DD HH:MM:SS - filename:line - function - message
    if format_string is None:
        format_string = (
            "%(levelname)-8s - %(asctime)s - %(filename)s:%(lineno)d - "
            "%(funcName)s() - %(message)s"
        )
    
    # Date format: YYYY-MM-DD HH:MM:SS
    date_format = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(format_string, datefmt=date_format)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get or create a logger for the given name.
    
    This is a convenience function that sets up a logger with default settings
    if it doesn't exist, or returns the existing logger.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    
    # If logger doesn't have handlers, set it up with defaults
    if not logger.handlers:
        setup_logger(name)
    
    return logger
