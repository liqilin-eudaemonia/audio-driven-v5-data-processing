#!/usr/bin/env python3
"""Invoke the deployed inference on node8 GPU 7 from any shared-workspace node."""
from pathlib import Path
import subprocess,sys,shlex
root=Path(__file__).resolve().parent
raise SystemExit(subprocess.call([sys.executable,str(root/'remote.py'),shlex.join(['bash',str(root/'node8.sh'),*sys.argv[1:]])]))
