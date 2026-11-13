import argparse,datetime,json,os,socket,subprocess
from pathlib import Path
import torch
import torch.distributed as dist
from p2_listen.contracts import atomic_json
p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--resume');p.add_argument('--smoke',action='store_true');p.add_argument('--output');p.add_argument('--stop-after',type=int);a=p.parse_args()
config=json.loads(Path(a.config).read_text());rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK']);world=int(os.environ['WORLD_SIZE']);assert world==32
assert not a.smoke and a.output is None
assert config['training']['micro_batch']==8 and config['training']['gradient_accumulation']==1
torch.set_num_threads(config.get('cpu_threads',2));torch.cuda.set_device(local)
dist.init_process_group('nccl',timeout=datetime.timedelta(seconds=300))
x=torch.tensor(float(rank),device=f'cuda:{local}');dist.all_reduce(x);assert x.item()==496.0
uuid=subprocess.check_output(['nvidia-smi','-i',str(local),'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
atomic_json(Path(config['output'])/'communication'/f'rank-{rank:02d}.json',dict(rank=rank,local_rank=local,world_size=world,host=socket.gethostname(),gpu_uuid=uuid,nccl_sum=x.item(),pid=os.getpid()))
from p2_planning_sft.train import train
train(a.config,a.resume,a.smoke,a.output,a.stop_after)
