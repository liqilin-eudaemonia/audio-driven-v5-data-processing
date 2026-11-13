#!/usr/bin/env python3
"""CLI for per-clip backchannel/face statistics and reproducible selection."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from listen_benchmark.backchannel_face_stats import main

if __name__ == "__main__":
    main()
