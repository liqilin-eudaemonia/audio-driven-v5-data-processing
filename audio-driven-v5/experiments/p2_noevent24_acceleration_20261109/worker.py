from pathlib import Path
import os,sys,argparse,json
W=Path(__file__).resolve().parent;ROOT=W.parents[1]
sys.path[:0]=[str(W),str(ROOT/'src'),str(ROOT/'scripts')]
p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--resume',required=True);a=p.parse_args()
import torch
torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
from p2_whisper_av.bootstrap import activate_ltx
activate_ltx()
import adapt_runtime as R
from p2_listen.contracts import read,sha
from p2_ltx_all_lora_noevent.config import load
c=load(a.config);ready=read(W/'ADOPTION_READY.json')
assert ready['source']==R.source()and sha(a.config)==ready['config_sha256']
assert int(os.environ['WORLD_SIZE'])==24
from gate import record,write
rank=int(os.environ['RANK']);physical=list(map(int,os.environ['CUDA_VISIBLE_DEVICES'].split(',')))[int(os.environ['LOCAL_RANK'])]
write(Path(c['output'])/'rank_logs'/f'worker_{rank:02d}.json',dict(**record(),rank=rank,world=24,physical_gpu=physical,command=sys.argv))
print(json.dumps(R.load().train(c,Path(a.resume))),flush=True)
