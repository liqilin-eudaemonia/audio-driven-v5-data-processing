#!/usr/bin/env python3
"""Use torchrun ... scripts/train_p2_listen_distributed.py --arm E0 --train-seed ..."""
import sys
import runpy
from pathlib import Path

if __name__=="__main__":
    sys.argv.insert(1,"train")
    runpy.run_path(str(Path(__file__).with_name("p2_listen.py")),run_name="__main__")
