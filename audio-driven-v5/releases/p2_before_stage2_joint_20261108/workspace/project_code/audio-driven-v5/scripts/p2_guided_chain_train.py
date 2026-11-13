#!/usr/bin/env python3
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('command',choices=['train']);p.add_argument('--config',required=True);p.add_argument('--parent');p.add_argument('--resume');p.add_argument('--stop-after',type=int);a=p.parse_args()
 from p2_matrix16.config import load
 from p2_matrix16.training import train
 from p2_guided_chain import backend
 result=train(load(a.config,formal=True),parent=a.parent,resume_path=a.resume,stop_after=a.stop_after,backend=backend)
 print(json.dumps(result),flush=True)
