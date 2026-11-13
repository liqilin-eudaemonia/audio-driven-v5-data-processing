import sys,os,time,json,traceback,hashlib
from pathlib import Path
root=Path('/private/liqi_workspace/project_code/audio-driven-v5')
sys.path.insert(0,str(root/'src'))
rank=int(sys.argv[1]);gpu=int(sys.argv[2]);folder=Path(__file__).parent
os.environ['CUDA_VISIBLE_DEVICES']=str(gpu)
os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2'
def status(**value):
 value.update(rank=rank,gpu=gpu,pid=os.getpid(),at=time.time());p=folder/f'state-{rank:02d}.json'
 temp=p.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(p)
status(state='starting')
try:
 import torch
 torch.set_num_threads(2);torch.set_num_interop_threads(1)
 torch.cuda.set_per_process_memory_fraction(24*2**30/torch.cuda.get_device_properties(0).total_memory)
 from p2_matrix16.config import load
 from p2_matrix16.teacher import cache_teacher
 c=load(root/'configs/p2_matrix16/guided_av_event_feature_s1.json')
 for attempt in (1,2):
  status(state='running',attempt=attempt)
  report=cache_teacher(c,c['teacher_cache'],rank=rank,shards=24,device='cuda:0')
  if report['success']:break
 status(state='succeeded' if report['success'] else 'failed',attempt=attempt,samples=report['samples'],failed=sum(not x['success'] for x in report['results']))
 sys.exit(0 if report['success'] else 2)
except Exception as exc:
 status(state='failed',error=str(exc));traceback.print_exc();sys.exit(2)
