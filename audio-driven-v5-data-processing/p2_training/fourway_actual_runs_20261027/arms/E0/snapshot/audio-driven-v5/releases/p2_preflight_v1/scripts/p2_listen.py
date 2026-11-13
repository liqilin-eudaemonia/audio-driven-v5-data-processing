#!/usr/bin/env python3
"""Use this versioned CLI for fast, shared preflight and matching inference/resume."""
import runpy
import sys
from pathlib import Path
RELEASE=Path(__file__).resolve().parents[1]
BASE=RELEASE.parents[1]
sys.path.insert(0,str(RELEASE/'src'))
import p2_listen  # Fix package resolution before the original CLI adds its src path.
if __name__=='__main__':
    runpy.run_path(str(BASE/'scripts/p2_listen.py'),run_name='__main__')
