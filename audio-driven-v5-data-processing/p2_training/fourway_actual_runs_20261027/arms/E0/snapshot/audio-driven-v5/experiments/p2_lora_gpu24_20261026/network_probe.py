import os,time,json
from pathlib import Path
import torch
import torch.distributed as dist
rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);local=int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(local);dist.init_process_group('nccl');records=[]
for mib in [4,32,416]:
 x=torch.empty(mib*1024*1024//8,device='cuda',dtype=torch.float64)
 times=[]
 for i in range(12):
  x.fill_(rank+1);torch.cuda.synchronize();dist.barrier();start=time.monotonic();dist.all_reduce(x);torch.cuda.synchronize();times.append(time.monotonic()-start)
  assert torch.all(x==(world*(world+1)//2)), 'Collective corruption'
 record=dict(mib=mib,seconds=sum(times[2:])/10,rank=rank,world=world);records.append(record)
 if rank==0:print(record,flush=True)
p=Path(os.environ['PROBE_OUTPUT']);p.mkdir(exist_ok=True,parents=True);(p/f'rank{rank}.json').write_text(json.dumps(records,indent=2));dist.destroy_process_group()
