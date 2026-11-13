#!/usr/bin/env python3
"""Prepare caption evidence, extract open behaviors, then select train subsets."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from listen_richness.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
