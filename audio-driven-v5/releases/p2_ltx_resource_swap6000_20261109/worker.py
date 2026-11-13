"""Bind the assigned CUDA device before importing the unchanged native trainer."""
from pathlib import Path
import argparse,importlib,json,os,socket,sys,time
WORK=Path(__file__).resolve().parent;ROOT=WORK.parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(WORK)]

def main():
 p=argparse.ArgumentParser();p.add_argument('--arm',choices=['full','noevent'],required=True);p.add_argument('--config',required=True);p.add_argument('--resume',required=True);a=p.parse_args()
 from gate import write,record,read
 ready=read(WORK/'CODE_READY.json')
 import hashlib
 assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==ready['source']['worker.py']
 assert hashlib.sha256(Path(a.config).read_bytes()).hexdigest()==ready['targets'][a.arm]['config_sha256']
 import torch
 torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
 from p2_whisper_av.bootstrap import activate_ltx
 activate_ltx()
 package='p2_ltx_all_lora'if a.arm=='full'else'p2_ltx_all_lora_noevent'
 c=importlib.import_module(package+'.config').load(a.config)
 world=int(os.environ['WORLD_SIZE']);assert world==(1 if a.arm=='full'else 31)
 if world==1:
  assert os.environ['CUDA_VISIBLE_DEVICES']=='0'
  write(Path(c['output'])/'CONTROLLER.json',dict(**record(),host=socket.gethostname(),physical_gpu=0,world=1,global_batch=31,command=sys.argv))
 result=importlib.import_module(package+'.training').train(c,Path(a.resume))
 print(json.dumps(result),flush=True)
if __name__=='__main__':main()
