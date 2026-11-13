#!/usr/bin/env python3
"""Configure, probe and launch frozen plan caches on the three specified hosts."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from p2_plan_cluster import cluster as cl


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    for cmd in ('plan','probe','start','smoke'):
        p=sub.add_parser(cmd)
        p.add_argument('--cluster',default=str(cl.DEFAULT))
        if cmd=='probe':p.add_argument('--output',required=True)
        if cmd in ('start','smoke'):p.add_argument('--job',required=True)
    for cmd in ('status','stop'):
        p=sub.add_parser(cmd);p.add_argument('--job',required=True)
    p=sub.add_parser('_probe');p.add_argument('--challenge',required=True);p.add_argument('--node',required=True)
    for cmd in ('_node','_detach'):
        p=sub.add_parser(cmd);p.add_argument('--job',required=True);p.add_argument('--node',required=True)
    p=sub.add_parser('_smoke');p.add_argument('--job',required=True);p.add_argument('--rank',required=True,type=int);p.add_argument('--stage',choices=cl.STAGES,required=True)
    a=parser.parse_args()
    if a.command=='plan':
        c=cl.load(a.cluster);rows=cl.assignments(c)
        result=dict(success=True,shards=len(rows),assignments=rows,stages=c['stages'],work=c['work'],
                    note='Independent cache workers; internal SSH ports, no NCCL/DDP rendezvous')
    elif a.command=='probe':result=cl.probe(cl.load(a.cluster),a.output)
    elif a.command in ('start','smoke'):
        result=cl.start(cl.load(a.cluster),a.job,smoke=a.command=='smoke')
        if a.command=='smoke' and result['success']:
            deadline=time.monotonic()+180
            while True:
                state=cl.status(a.job)
                if state['state']=='succeeded':
                    result=cl.smoke_check(a.job)
                    cl.atomic_json(Path(a.job)/'smoke_validation.json',result);break
                if state['state']=='failed' or time.monotonic()>deadline:
                    Path(a.job,'STOP').touch()
                    result=dict(success=False,error='Smoke failed/timed out',status=state);break
                time.sleep(1)
    elif a.command=='status':result=cl.status(a.job)
    elif a.command=='stop':
        cl.read(Path(a.job)/'job.json')
        Path(a.job,'STOP').touch();result=dict(success=True,stop_requested=True)
    elif a.command=='_probe':result=cl.probe_node(a.challenge,a.node)
    elif a.command=='_node':result=cl.supervise(a.job,a.node)
    elif a.command=='_detach':result=cl.detach(a.job,a.node)
    else:result=cl.smoke_worker(a.job,a.rank,a.stage)
    print(json.dumps(result,ensure_ascii=False),flush=True)
    # Running status is informative rather than a command failure.
    return 0 if result.get('success') or a.command=='status' and result.get('state')=='running' else 2


if __name__=='__main__':raise SystemExit(main())
