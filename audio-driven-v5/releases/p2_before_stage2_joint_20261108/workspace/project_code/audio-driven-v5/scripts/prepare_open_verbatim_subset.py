#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from listen_benchmark.open_verbatim_subset import prepare

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['fusion','full','stats','report','output']:p.add_argument('--'+name,required=True)
    p.add_argument('--workers',type=int,default=32);a=p.parse_args()
    prepare(a.fusion,a.full,a.stats,a.report,a.output,a.workers)
