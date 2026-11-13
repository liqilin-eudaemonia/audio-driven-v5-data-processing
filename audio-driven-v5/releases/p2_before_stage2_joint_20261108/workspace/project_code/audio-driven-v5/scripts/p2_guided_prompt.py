#!/usr/bin/env python3
"""Prepare verbatim guidance, cache Gemma and run annotation-guided diagnostics."""
import argparse
import json
from pathlib import Path
import runpy
import sys
ROOT = Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser(description=__doc__); commands=parser.add_subparsers(dest='command',required=True)
    prep=commands.add_parser('prepare');prep.add_argument('--observations',required=True);prep.add_argument('--annotations',required=True);prep.add_argument('--output',required=True);prep.add_argument('--max-text-tokens',type=int,choices=[1024,2048,4096],default=4096)
    cache=commands.add_parser('cache-text');cache.add_argument('--work',required=True);cache.add_argument('--run',required=True);cache.add_argument('--sample-id',action='append');cache.add_argument('--device',default='cuda:0')
    gen=commands.add_parser('generate');gen.add_argument('--work',required=True);gen.add_argument('--run',required=True);gen.add_argument('--checkpoint',required=True);gen.add_argument('--sample-id',required=True);gen.add_argument('--output',required=True);gen.add_argument('--seed',type=int,default=137);gen.add_argument('--device',default='cuda:0');gen.add_argument('--restoration');gen.add_argument('--variant',choices=['fixed','guided'],default='guided')
    args=parser.parse_args()
    if hasattr(args,'run'):
        runpy.run_path(str(ROOT/'scripts/p2_eval_generate.py'))['bootstrap'](['--run',args.run])
    sys.path.insert(0,str(ROOT/'src'))  # Avoid this CLI filename shadowing the package; p2_listen is already bootstrapped.
    if args.command=='prepare':
        from p2_guided_prompt.protocol import prepare
        result=prepare(args.observations,args.annotations,args.output,args.max_text_tokens)
    elif args.command=='cache-text':
        from p2_guided_prompt.runtime import cache_text
        # Activate the reused model modules before their first import.
        from p2_listen.config import activate
        activate(json.loads((Path(args.run)/'run.json').read_text())['config'])
        result=cache_text(args.work,args.run,args.sample_id,args.device)
    else:
        from p2_guided_prompt.runtime import generate
        result=generate(args.work,args.run,args.checkpoint,args.sample_id,args.output,seed=args.seed,device=args.device,restoration=args.restoration,variant=args.variant)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    if result.get('success') is False:return 2
    return 0

if __name__=='__main__':raise SystemExit(main())
