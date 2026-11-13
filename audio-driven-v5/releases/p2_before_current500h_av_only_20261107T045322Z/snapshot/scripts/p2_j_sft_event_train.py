"""Preflight or train AV+Event using frozen planning-SFT hidden states."""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))

if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['preflight','train'])
    p.add_argument('--config',required=True);p.add_argument('--resume');p.add_argument('--stop-after',type=int)
    p.add_argument('--processes',type=int,default=16);a=p.parse_args()
    from p2_matrix16.config import load
    from p2_j_duration_execution import install_sampler
    from p2_matrix16.training import train
    import p2_j_sft_event_backend as backend
    c=load(a.config,formal=True);install_sampler(c)
    if a.command=='preflight':
        result=backend.preflight(c,processes=a.processes)
        print(json.dumps(dict(samples=result['samples'],training_started=False)))
    else:print(json.dumps(train(c,resume_path=a.resume,stop_after=a.stop_after,backend=backend)))
