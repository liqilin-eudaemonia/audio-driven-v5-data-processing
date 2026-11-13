from pathlib import Path
import sys,os,argparse,subprocess,time,json
W=Path(__file__).resolve().parent;sys.path.insert(0,str(W));import runtime28 as R
p=argparse.ArgumentParser();p.add_argument('mode',choices=['node','train']);p.add_argument('--index',type=int);p.add_argument('--resume');a=p.parse_args()
c=R.validate(R.read(R.RUN/'config.json'));cl=R.read(R.RUN/'cluster.json')
if a.mode=='train':
 import torch
 torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
 T=R.activate();rank=int(os.environ['RANK'])
 R.atomic_json(R.RUN/'rank_logs'/f'worker_{rank:02d}.json',dict(pid=os.getpid(),rank=rank,world=28,host=__import__('socket').gethostname(),visible=os.environ['CUDA_VISIBLE_DEVICES'],local=int(os.environ['LOCAL_RANK'])))
 print(json.dumps(T.train(c,Path(a.resume)if a.resume else None)),flush=True)
else:
 n=cl['nodes'][a.index];mask=','.join(map(str,n['gpus']));assert os.environ['CUDA_VISIBLE_DEVICES']==mask
 off=sum(len(x['gpus'])for x in cl['nodes'][:a.index]);children=[];streams=[]
 try:
  for local,gpu in enumerate(n['gpus']):
   rank=off+local;env=dict(os.environ,RANK=str(rank),LOCAL_RANK=str(local),WORLD_SIZE='28',LOCAL_WORLD_SIZE=str(len(n['gpus'])),MASTER_ADDR=cl['nodes'][0]['host'],MASTER_PORT=str(cl['master_port']))
   f=(R.RUN/'rank_logs'/f'rank_{rank:02d}.log').open('a');streams.append(f)
   cmd=[cl['python'],'-u',str(Path(__file__)),'train']
   if a.resume:cmd+=['--resume',a.resume]
   children.append(subprocess.Popen(cmd,env=env,stdout=f,stderr=subprocess.STDOUT))
  while any(p.poll()is None for p in children):
   if any(p.poll()not in(None,0)for p in children):raise RuntimeError('Joint28 worker failed')
   time.sleep(1)
  assert all(p.returncode==0 for p in children)
 finally:
  for p in children:
   if p.poll()is None:p.terminate()
  for p in children:
   try:p.wait(timeout=20)
   except subprocess.TimeoutExpired:p.kill();p.wait()
  for f in streams:f.close()
