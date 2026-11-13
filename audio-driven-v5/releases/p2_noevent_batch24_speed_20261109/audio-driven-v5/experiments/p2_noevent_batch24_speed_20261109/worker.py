from pathlib import Path
import sys,os,argparse,json
W=Path(__file__).resolve().parent;ROOT=W.parents[1];sys.path[:0]=[str(W),str(ROOT/'src'),str(ROOT/'scripts')]
p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--resume',required=True);a=p.parse_args()
import torch
torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
from p2_whisper_av.bootstrap import activate_ltx
activate_ltx()
import batch_runtime as R
from p2_listen.contracts import read,sha
from gate import write,record
c=R.validate(read(a.config));ready=read(W/'READY.json');assert ready['source']==R.source()and ready['config_sha256']==sha(a.config)
rank=int(os.environ['RANK']);assert int(os.environ['WORLD_SIZE'])==24
write(Path(c['output'])/'rank_logs'/f'worker_{rank:02d}.json',dict(**record(),host=__import__('socket').gethostname(),rank=rank,world=24,physical_gpu=list(map(int,os.environ['CUDA_VISIBLE_DEVICES'].split(',')))[int(os.environ['LOCAL_RANK'])]))
print(json.dumps(dict(phase='loading',rank=rank,batch=24)),flush=True)
print(json.dumps(R.load().train(c,Path(a.resume))),flush=True)
