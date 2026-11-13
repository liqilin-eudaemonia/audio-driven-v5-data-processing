#!/usr/bin/env python3
"""Prepare/evaluate OmniResponse AV metrics with explicit listener pairs."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('runtime');q.add_argument('--catalog',type=Path,default=ROOT.parent/'OmniResponse/evaluation/autonomous_dev48/catalog.local.json');q.add_argument('--output',type=Path,default=ROOT/'configs/p2_omni_runtime.lock.json')
    q=sub.add_parser('check');q.add_argument('--runtime',type=Path,default=ROOT/'configs/p2_omni_runtime.lock.json')
    q=sub.add_parser('prepare')
    for name in ('manifest','output'):q.add_argument('--'+name,type=Path,required=True)
    q.add_argument('--runtime',type=Path,default=ROOT/'configs/p2_omni_runtime.lock.json')
    q.add_argument('--metrics',nargs='+',choices=['fvd','fd','utmos','sync','text'],default=['fvd','utmos','sync'])
    q.add_argument('--fvd-frames',type=int,choices=[32,256],default=32);q.add_argument('--crop-seed',type=int,default=42)
    for cmd in ('evaluate','worker'):
        q=sub.add_parser(cmd);q.add_argument('--work',type=Path,required=True);q.add_argument('--device',default='cpu')
    q=sub.add_parser('export-generation')
    for name in ('work','gt-targets','output'):q.add_argument('--'+name,type=Path,required=True)
    q=sub.add_parser('sync-worker');q.add_argument('--request',type=Path,required=True)
    q=sub.add_parser('track-worker');q.add_argument('--sync-root',type=Path,required=True)
    q=sub.add_parser('compare');q.add_argument('--left',type=Path,required=True);q.add_argument('--right',type=Path,required=True);q.add_argument('--output',type=Path,required=True)
    args,extra=p.parse_known_args()
    if extra and args.command!='track-worker':p.error('Unknown arguments: '+' '.join(extra))
    from p2_omni_eval.common import read,atomic_json
    if args.command=='runtime':
        from p2_omni_eval.common import build_runtime
        r=build_runtime(args.catalog);atomic_json(args.output,r,immutable=True);result=dict(status='ready',source_files=len(r['source']),model_files=len(r['models']),output=str(args.output))
    elif args.command=='check':
        from p2_omni_eval.common import check_runtime
        check_runtime(read(args.runtime),full=True);result=dict(status='ready')
    elif args.command=='prepare':
        from p2_omni_eval.protocol import prepare
        result=prepare(args.manifest,args.output,args.runtime,args.metrics,dict(fvd_frames=args.fvd_frames,crop_seed=args.crop_seed))
    elif args.command=='export-generation':
        from p2_omni_eval.protocol import export_generation
        result=export_generation(args.work,args.gt_targets,args.output)
    elif args.command in ('evaluate','worker'):
        from p2_omni_eval.pipeline import evaluate,worker
        result=(evaluate if args.command=='evaluate' else worker)(args.work,args.device)
        if args.command=='worker':result={k:v for k,v in result.items() if k not in ('rows','per_seed')}
    elif args.command=='sync-worker':
        from p2_omni_eval.sync import worker
        worker(args.request);return
    elif args.command=='track-worker':
        from p2_omni_eval.sync import track_worker
        track_worker(args.sync_root,extra);return
    else:
        from p2_omni_eval.compare import compare
        result=compare(args.left,args.right,args.output)
    print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
    if result.get('status')=='incomplete':raise SystemExit(2)

if __name__=='__main__':main()
