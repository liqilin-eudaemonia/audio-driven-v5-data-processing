#!/usr/bin/env python3
"""Opt-in, source-pinned duration-aligned training and isolated timing probes."""
from pathlib import Path
import argparse,json,os,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from p2_duration_execution import install_backend,install_sampler

def profile(folder,backend):
    import torch
    from p2_matrix16 import model
    from p2_listen import performance
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    rank=int(os.environ.get('RANK',0));trace={};iteration=0
    def wrap(owner,name,key):
        original=getattr(owner,name)
        def timed(*args,**kwargs):
            nonlocal trace,iteration
            if key=='data':trace={};iteration+=1
            torch.cuda.synchronize();start=time.perf_counter()
            result=original(*args,**kwargs);torch.cuda.synchronize()
            trace[key]=time.perf_counter()-start
            if key=='data':
                row=args[0].rows[args[1]]
                trace.update(sample_id=row['sample_id'],duration=row['requested_duration_s'])
            if key=='optimizer':
                with (folder/f'rank_{rank}.jsonl').open('a') as f:
                    f.write(json.dumps(dict(iteration=iteration,rank=rank,**trace))+'\n')
            return result
        setattr(owner,name,timed)
    for owner,name,key in [(backend.Dataset,'sample','data'),(model.MatrixModel,'forward','forward'),
        (torch.Tensor,'backward','backward'),(performance.BucketGradientAccumulator,'finish','reduce'),
        (torch.optim.AdamW,'step','optimizer')]:wrap(owner,name,key)

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['train']);p.add_argument('--config',required=True)
    p.add_argument('--parent');p.add_argument('--resume');p.add_argument('--stop-after',type=int);p.add_argument('--profile')
    a=p.parse_args()
    from p2_matrix16.config import load
    from p2_matrix16.training import train
    backend=install_backend();c=load(a.config,formal=True);install_sampler(c)
    if a.profile:profile(a.profile,backend)
    result=train(c,parent=a.parent,resume_path=a.resume,stop_after=a.stop_after,backend=backend)
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
