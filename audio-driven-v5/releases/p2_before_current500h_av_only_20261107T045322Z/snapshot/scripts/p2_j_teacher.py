#!/usr/bin/env python3
"""Feature teacher sidecar; does not mutate the existing 500h training cache."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','worker','node','status','audit'])
    p.add_argument('--work',required=True);p.add_argument('--source');p.add_argument('--ids')
    p.add_argument('--gpu',type=int,default=0);p.add_argument('--batch',type=int,default=16)
    p.add_argument('--producers',type=int,default=4);p.add_argument('--prefetch',type=int,default=32)
    p.add_argument('--per-gpu',type=int,default=2)
    a=p.parse_args()
    if a.command=='prepare':
        from p2_j_teacher.core import prepare_work
        ids=None if a.ids is None else json.loads(Path(a.ids).read_text())
        r=prepare_work(a.source,a.work,ids);r={k:v for k,v in r.items() if k not in ('sample_ids','dependencies','code')}
    else:
        from p2_j_teacher import runner
        if a.command=='worker':r=runner.worker(a.work,a.gpu,a.batch,a.producers,a.prefetch)
        elif a.command=='node':
            import fcntl
            with (Path(a.work)/'node.lock').open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                r=runner.node(a.work,a.batch,a.producers,a.prefetch,a.per_gpu)
        else:r=getattr(runner,a.command)(a.work)
    print(json.dumps(r,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
