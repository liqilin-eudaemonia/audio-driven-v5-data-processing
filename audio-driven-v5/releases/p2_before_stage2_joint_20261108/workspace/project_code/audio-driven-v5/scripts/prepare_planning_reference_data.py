"""Validate latest manifests/index once before multi-rank planning training."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from p2_planning_sft.reference_data import prepare
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
 p.add_argument('--diagnostic-count',type=int,default=256);p.add_argument('--seed',type=int,default=20261102);a=p.parse_args()
 prepare(a.data,a.output,seed=a.seed,diagnostic_count=a.diagnostic_count)
