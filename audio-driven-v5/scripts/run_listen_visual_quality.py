#!/usr/bin/env python3
"""CPU-only visual quality entry; see releases/listen_visual_quality_v1_20261030/."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from listen_visual_quality.cli import main

if __name__ == "__main__":
    main()
