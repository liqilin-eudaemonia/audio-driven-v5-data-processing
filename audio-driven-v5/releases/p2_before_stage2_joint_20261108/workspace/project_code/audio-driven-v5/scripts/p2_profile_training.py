"""Temporary 24-rank timing probe, never used by the production queue."""
from pathlib import Path
import sys,os,time,json,runpy
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import torch
from p2_legacy_qwen_lora import backend
from p2_listen import performance
from p2_matrix16 import model,qwen
config=Path(sys.argv[sys.argv.index('--config')+1]).resolve()
if os.environ.get('P2_PROFILE_FAST')=='1':
 from p2_legacy_execution import install
 install(config.parent.parent/'execution.json')
# Activate pinned native dependencies before importing the AV adapter.
backend.data_config(json.loads(config.read_text()))
from p2_listen import runtime
folder=config.parent.parent/'timings';folder.mkdir(exist_ok=True)
rank=int(os.environ['RANK']);trace={};counter=0;last_finish=0

def timed(owner,name,key):
 original=getattr(owner,name)
 def wrapped(*args,**kwargs):
  global trace,counter,last_finish
  if key=='data':trace={};counter+=1
  torch.cuda.synchronize();start=time.perf_counter()
  if key=='optimizer':trace['norm_logging_and_clip']=start-last_finish
  result=original(*args,**kwargs);torch.cuda.synchronize();trace[key]=time.perf_counter()-start
  if key=='reduce':last_finish=time.perf_counter()
  if key=='optimizer':
   with (folder/f'rank_{rank}.jsonl').open('a') as f:f.write(json.dumps(dict(iteration=counter,rank=rank,**trace))+'\n')
  return result
 setattr(owner,name,wrapped)
for owner,name,key in [(backend.Dataset,'sample','data'),(qwen,'online_condition','qwen'),(runtime,'av_loss','ltx_forward'),(model.MatrixModel,'forward','forward'),(torch.Tensor,'backward','backward'),(performance.BucketGradientAccumulator,'add','accumulate'),(performance.BucketGradientAccumulator,'finish','reduce'),(torch.optim.AdamW,'step','optimizer')]:timed(owner,name,key)
runpy.run_path(str(ROOT/'scripts/p2_legacy_qwen_lora.py'),run_name='__main__')
