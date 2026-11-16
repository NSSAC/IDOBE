"""Lets scripts be run as `python scripts/<name>.py` from the annotation/ folder or anywhere else."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
