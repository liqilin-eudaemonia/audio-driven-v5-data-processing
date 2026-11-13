from pathlib import Path
import sys,os,argparse,time,json
ROOT=Path('/private/liqi_workspace/project_code/audio-driven-v5');sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]

def main():
 import torch,torch.distributed as dist
 from datetime import timedelta
 from p2_listen.contracts import atomic_json
 p=argparse.ArgumentParser();p.add_argument('command');p.add_argument('--config');p.add_argument('--work',type=Path);p.add_argument('--checkpoint');p.add_argument('--steps',type=int);p.add_argument('--start-step',type=int);a=p.parse_args()
 local=int(os.environ['LOCAL_RANK']);rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);torch.set_num_threads(2);torch.cuda.set_device(local)
 dist.init_process_group('nccl',device_id=torch.device(f'cuda:{local}'),timeout=timedelta(seconds=90))
 try:
  rows=[]
  for mib in [1,32,512]:
   x=torch.empty(mib*1024*1024//8,device=f'cuda:{local}',dtype=torch.float64);times=[]
   for i in range(4):
    x.fill_(rank+1);dist.barrier(device_ids=[local]);torch.cuda.synchronize();begin=time.monotonic();dist.all_reduce(x);torch.cuda.synchronize();elapsed=time.monotonic()-begin
    if not bool((x==world*(world+1)//2).all()):raise ValueError('Incorrect FP64 SUM')
    t=torch.tensor(elapsed,device=x.device);dist.all_reduce(t,op=dist.ReduceOp.MAX);times.append(float(t))
   rows.append(dict(mib=mib,seconds=times))
   del x
  if rank==0:atomic_json(a.work/'SMOKE_ACCEPTED.json',dict(success=True,world=world,hca=os.environ.get('NCCL_IB_HCA'),cross_nic=os.environ.get('NCCL_CROSS_NIC'),measurements=rows))
 finally:dist.destroy_process_group()

def node_main():
    import subprocess
    from p2_listen.contracts import read
    p=argparse.ArgumentParser();p.add_argument('command');p.add_argument('--config',required=True);p.add_argument('--work',type=Path,required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--node-index',type=int,required=True);p.add_argument('--start-step',type=int,required=True);p.add_argument('--steps',type=int,default=8)
    a=p.parse_args();c=read(a.config);cluster=read(Path(c['output'])/'cluster.json');n=cluster['nodes'][a.node_index];off=sum(len(x['gpus']) for x in cluster['nodes'][:a.node_index]);entry=dict(gpus=n['gpus'],ranks=list(range(off,off+len(n['gpus']))));processes=[];streams=[]
    assert [x['name'] for x in cluster['nodes']]==['node12','node8','node4','node11']
    assert [x['gpus'] for x in cluster['nodes']]==[list(range(8))]*3+[list(range(1,8))]
    a.work.mkdir(parents=True,exist_ok=True)
    try:
        for local,rank in enumerate(entry['ranks']):
            env=dict(os.environ,RANK=str(rank),LOCAL_RANK=str(local),WORLD_SIZE='31',LOCAL_WORLD_SIZE=str(len(entry['gpus'])),MASTER_ADDR=cluster['nodes'][0]['host'],MASTER_PORT='29957')
            log=(a.work/f'rank{rank:02d}.log').open('a');streams.append(log)
            processes.append(subprocess.Popen([cluster['python'],__file__,'train','--config',a.config,'--work',str(a.work),'--checkpoint',a.checkpoint,'--start-step',str(a.start_step),'--steps',str(a.steps)],env=env,stdout=log,stderr=subprocess.STDOUT))
        while any(p.poll() is None for p in processes):
            if any(p.poll() not in (None,0) for p in processes):raise RuntimeError('Benchmark rank failed')
            time.sleep(2)
        if any(p.returncode for p in processes):raise RuntimeError('Benchmark failed')
    finally:
        for p in processes:
            if p.poll() is None:p.terminate()
        for p in processes:
            try:p.wait(timeout=15)
            except subprocess.TimeoutExpired:p.kill();p.wait()
        for f in streams:f.close()


if __name__=='__main__':
    if sys.argv[1]=='node':node_main()
    else:main()

