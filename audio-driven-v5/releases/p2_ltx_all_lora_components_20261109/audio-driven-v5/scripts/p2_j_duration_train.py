"""J S1 training with explicit, checkpoint-bound duration sampling."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from p2_j_duration_execution import install_backend,install_sampler

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['train','preflight'])
    parser.add_argument('--config',required=True);parser.add_argument('--resume');parser.add_argument('--stop-after',type=int)
    parser.add_argument('--processes',type=int,default=32)
    args=parser.parse_args()
    from p2_matrix16.config import load
    from p2_matrix16.training import train
    c=load(args.config,formal=True);backend=install_backend();install_sampler(c)
    if args.command=='preflight':
        result=backend.preflight(c,processes=args.processes)
        print(json.dumps(dict(samples=result['samples'],output=c['output'],training_started=False)),flush=True)
    else:print(json.dumps(train(c,resume_path=args.resume,stop_after=args.stop_after,backend=backend)),flush=True)
