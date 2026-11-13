import os,json,time,datetime
from pathlib import Path
import torch
import torch.distributed as d
local=int(os.environ['LOCAL_RANK']);rank=int(os.environ['RANK']);torch.cuda.set_device(local)
torch.cuda.set_per_process_memory_fraction(3*2**30/torch.cuda.get_device_properties(local).total_memory,local)
d.init_process_group('nccl',device_id=torch.device('cuda',local),timeout=datetime.timedelta(seconds=45))
try:
 results=[]
 for mb in [1,32]:
  x=torch.ones(mb*1024*1024//8,device='cuda',dtype=torch.float64)
  start=time.monotonic()
  for i in range(5):
   x.fill_(rank+1);d.all_reduce(x)
   assert bool((x==300).all())
  torch.cuda.synchronize();results.append(dict(mb=mb,seconds=time.monotonic()-start))
 if rank==0:Path(os.environ['PROBE_RESULT']).write_text(json.dumps(dict(success=True,world=d.get_world_size(),hca=os.environ['NCCL_IB_HCA'],results=results,concurrent_training=True,not_end_to_end_speed_evidence=True)))
finally:d.destroy_process_group()
