"""Root CLI entry point forwarding to training.train."""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.train import build_parser, main

if __name__ == "__main__":
    main()
