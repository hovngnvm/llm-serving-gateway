"""
Unified Logger Utility for Training Pipelines (Re-exports central gateway logger).
"""

from gateway.app.utils.logger import get_logger

__all__ = ["get_logger"]
