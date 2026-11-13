#!/usr/bin/env python3
"""Explicit 2 text x 4 loss x 2 Qwen-stage training matrix."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))

def main():
    from p2_matrix16.config import load,emit
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    emitp=sub.add_parser('emit');emitp.add_argument('--output',default=str(ROOT/'configs/p2_matrix16'))
    for name in ('preflight','train','cache-teacher'):
        x=sub.add_parser(name);x.add_argument('--config',required=True)
        if name=='train':
            g=x.add_mutually_exclusive_group();g.add_argument('--resume');g.add_argument('--parent')
            x.add_argument('--steps',type=int);x.add_argument('--global-batch',type=int)
            x.add_argument('--stop-after',type=int);x.add_argument('--audit-gradients',action='store_true')
        if name=='cache-teacher':
            x.add_argument('--sample-ids',nargs='+');x.add_argument('--rank',type=int,default=0)
            x.add_argument('--shards',type=int,default=1);x.add_argument('--device',default='cuda:0')
    a=p.parse_args()
    if a.command=='emit':result=dict(success=True,experiments=emit(a.output))
    else:
        c=load(a.config)
        if a.command=='preflight':
            from p2_matrix16.training import preflight
            result=preflight(c)
        elif a.command=='cache-teacher':
            from p2_matrix16.teacher import cache_teacher
            result=cache_teacher(c,c['teacher_cache'],a.sample_ids,a.rank,a.shards,a.device)
        else:
            from p2_matrix16.training import train
            result=train(c,resume_path=a.resume,parent=a.parent,steps=a.steps,global_batch=a.global_batch,
                         stop_after=a.stop_after,audit=a.audit_gradients)
    print(json.dumps(result,ensure_ascii=False),flush=True)
    return 2 if result.get('success') is False else 0

if __name__=='__main__':raise SystemExit(main())
