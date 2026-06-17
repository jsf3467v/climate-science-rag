"""Add src/ and evaluation/ to the path so tests import the real modules.

The scripts run from inside those folders, so they are the import roots. Tests
live at the repo root, so both are added here once for the whole suite.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for name in ("src", "evaluation"):
    path = str(ROOT / name)
    if path not in sys.path:
        sys.path.insert(0, path)
