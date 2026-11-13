#!/usr/bin/env python3
"""Use the frozen CPU runtime when an upstream nod runtime differs."""
import argparse,json,sys,importlib.metadata
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from listen_benchmark.j_annotation import read
from listen_benchmark.j_visual_quality_schedule import ensure_nod_quality
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--job',type=Path,required=True);a=p.parse_args()
 c=read(a.config)['visual_quality_gate'];expected=read(Path(c['work'])/'config.json')['contract']['packages']
 assert all(importlib.metadata.version(n)==v for n,v in expected.items()),'Incorrect early screening runtime'
 print(json.dumps(ensure_nod_quality(c,a.job)))
