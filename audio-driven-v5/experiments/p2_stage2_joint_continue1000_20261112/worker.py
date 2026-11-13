from pathlib import Path
import sys,os,argparse,subprocess,time,json
W=Path(__file__).resolve().parent;sys.path.insert(0,str(W));import runtime as R
p=argparse.ArgumentParser();p.add_argument('mode',choices=['node','train']);p.add_argument('--index',type=int);p.add_argument('--resume',required=True);a=p.parse_args()
c=R.validate(R.read(R.RUN/'config.json'));cl=R.read(R.RUN/'cluster.json')
if a.mode=='train':
 import torch
 torch.cuda.set_device(int(os.environ['LOCAL_RANK']));T=R.activate();rank=int(os.environ['RANK'])
 R.atomic_json(R.RUN/'rank_logs'/f'worker_{rank:02d}.json',dict(pid=os.getpid(),rank=rank,world=28,host=__import__('socket').gethostname(),visible=os.environ['CUDA_VISIBLE_DEVICES'],local=int(os.environ['LOCAL_RANK'])))
 print(json.dumps(T.train(c,Path(a.resume))),flush=True)
else:
 n=cl['nodes'][a.index];assert os.environ['CUDA_VISIBLE_DEVICES']==','.join(map(str,n['gpus']))
 off=sum(len(x['gpus'])for x in cl['nodes'][:a.index]);children=[];streams=[]
 try:
  for local,gpu in enumerate(n['gpus']):
   rank=off+local;env=dict(os.environ,RANK=str(rank),LOCAL_RANK=str(local),WORLD_SIZE='28',LOCAL_WORLD_SIZE=str(len(n['gpus'])),MASTER_ADDR=cl['nodes'][0]['host'],MASTER_PORT=str(cl['master_port']))
   f=(R.RUN/'rank_logs'/f'rank_{rank:02d}.log').open('a');streams.append(f)
   children.append(subprocess.Popen([cl['python'],'-u',str(Path(__file__)),'train','--resume',a.resume],env=env,stdout=f,stderr=subprocess.STDOUT))
  while any(p.poll()is None for p in children):
   if any(p.poll()not in(None,0)for p in children):raise RuntimeError('Continuation worker failed')
   time.sleep(1)
  assert all(p.returncode==0 for p in children)
 finally:
  for p in children:
   if p.poll()is None:p.terminate()
  for p in children:
   try:p.wait(timeout=20)
   except subprocess.TimeoutExpired:p.kill();p.wait()
  for f in streams:f.close()
