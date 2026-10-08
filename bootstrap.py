"""Resolve shared research modules without depending on the caller's directory."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent

# Keep legacy module names stable for existing model code while organizing files by method.
# This changes import locations, not the network architecture or checkpoint parameter names.
for folder in (ROOT, ROOT / 'data', ROOT / 'pure_rl', ROOT / 'proposed_model/shared'):
    location = str(folder)
    if location not in sys.path:
        sys.path.insert(0, location)
