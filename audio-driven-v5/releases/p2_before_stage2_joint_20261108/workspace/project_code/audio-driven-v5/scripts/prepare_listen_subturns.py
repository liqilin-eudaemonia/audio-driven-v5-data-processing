#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from p3_listen.subturns import prepare, render, render_contexts

if __name__ == "__main__":
    p=argparse.ArgumentParser(description="<=10s subturns and source-track loudness normalization")
    p.add_argument("command",choices=("plan","prepare","render","context"))
    p.add_argument("--workers",type=int,default=4)
    a=p.parse_args()
    if a.workers<1: p.error("workers must be positive")
    if a.command=="render":
        render(a.workers)
        render_contexts(a.workers)
        from repair_listen_preview_audio import repair
        repair(a.workers)
    elif a.command=="context":render_contexts(a.workers)
    else: prepare(a.workers,normalize_audio=a.command=="prepare")
