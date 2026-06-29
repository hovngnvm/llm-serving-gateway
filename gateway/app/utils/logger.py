"""
Centralized Logger Utility for Enterprise AI Gateway.
Provides standardized ISO-8601 timestamps and non-propagating loggers to eliminate duplicated logs.
"""

import logging
import sys


def get_logger(name: str = "EnterpriseAIGateway", level: int = logging.INFO) -> logging.Logger:
    """Factory function to build and configure a unified console logger."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if not logger.handlers:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S%z"
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger
