#!/usr/bin/env python3
"""GLARE four-metric evaluation, independent of the running P2 trainer."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def parser():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('detector-runtime');q.add_argument('--output',type=Path)
    q=sub.add_parser('detect-one')
    for key in ('video','output','protocol'):q.add_argument('--'+key,type=Path,required=True)
    q=sub.add_parser('prepare-pairs');q.add_argument('--manifest',type=Path,required=True);q.add_argument('--output',type=Path,required=True)
    q.add_argument('--detector-runtime',type=Path,default=ROOT/'.eval_models/detector_runtime.json')
    def score_options(q):
        q.add_argument('--detector-python',type=Path,default=ROOT/'.venv-p2-glare-detector/bin/python')
        q.add_argument('--device',default='cpu',help='FID device; independent detectors use CPU')
        q.add_argument('--batch-size',type=int,default=16)
        q.add_argument('--workers',type=int,default=4,help='Independent video detection workers')
    q=sub.add_parser('evaluate');q.add_argument('--work',type=Path,required=True);q.add_argument('--detection-only',action='store_true');score_options(q)
    q=sub.add_parser('prepare-generation')
    for key in ('run','observations','targets','output'):q.add_argument('--'+key,type=Path,required=True)
    q.add_argument('--seeds',type=int,nargs='+',default=[137,251]);q.add_argument('--train-diagnostic',action='store_true',help='Explicit in-training diagnostic; never a held-out score')
    q=sub.add_parser('generate');q.add_argument('--work',type=Path,required=True);q.add_argument('--checkpoint',type=Path,required=True)
    q.add_argument('--device',default='cuda:0');q.add_argument('--shards',type=int,default=1);q.add_argument('--shard-index',type=int,default=0)
    q.add_argument('--wait',action='store_true');q.add_argument('--wait-timeout',type=float,default=86400)
    q=sub.add_parser('export-pairs');q.add_argument('--work',type=Path,required=True);q.add_argument('--output',type=Path,required=True)
    q=sub.add_parser('run',help='Generate on local GPUs, independently detect, then score all planned outputs')
    q.add_argument('--work',type=Path,required=True);q.add_argument('--checkpoint',type=Path,required=True)
    q.add_argument('--devices',nargs='+',default=['cuda:0']);q.add_argument('--wait',action='store_true')
    q.add_argument('--wait-timeout',type=float,default=86400);q.add_argument('--evaluation-output',type=Path,required=True);score_options(q)
    return p


def main():
    a=parser().parse_args()
    from p2_glare_eval.common import read,atomic_json
    if a.command=='detector-runtime':
        from p2_glare_eval.detector import runtime_identity
        result=runtime_identity()
        if a.output:atomic_json(a.output,result,immutable=True)
    elif a.command=='detect-one':
        from p2_glare_eval.detector import detect
        p=read(a.protocol)
        result=detect(a.video,a.output,p['detector_policy'],p['detector_runtime'])
    elif a.command=='prepare-pairs':
        from p2_glare_eval.pipeline import prepare_pairs
        p=prepare_pairs(a.manifest,a.output,a.detector_runtime)
        result=dict(samples=p['sample_count'],seeds=p['seeds'],output=str(a.output))
    elif a.command=='evaluate':
        from p2_glare_eval.pipeline import evaluate
        result=evaluate(a.work,detector_python=a.detector_python,device=a.device,batch_size=a.batch_size,
                        detection_only=a.detection_only,workers=a.workers)
    elif a.command=='prepare-generation':
        from p2_glare_eval.generation import prepare
        result=prepare(a.run,a.observations,a.targets,a.output,a.seeds,train_diagnostic=a.train_diagnostic)
    elif a.command=='generate':
        from p2_glare_eval.generation import generate
        result=generate(a.work,a.checkpoint,device=a.device,shards=a.shards,shard_index=a.shard_index,
                        wait=a.wait,wait_timeout=a.wait_timeout)
    elif a.command=='export-pairs':
        from p2_glare_eval.generation import export_pairs
        result=export_pairs(a.work,a.output)
    else:
        from p2_glare_eval.pipeline import run
        result=run(a.work,a.checkpoint,a.evaluation_output,devices=a.devices,wait=a.wait,wait_timeout=a.wait_timeout,
                   detector_python=a.detector_python,device=a.device,batch_size=a.batch_size,workers=a.workers)
    print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
    if result.get('status') in ('incomplete','scoring_failed') or any(r.get('status')=='failed' for r in result.get('results',[])):
        raise SystemExit(2)


if __name__=='__main__':main()
