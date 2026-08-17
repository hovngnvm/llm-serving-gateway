"""
Root pytest configuration and fixture setup for Enterprise AI Platform.
Ensures project root is added to sys.path for unified module discovery.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

