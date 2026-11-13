from pathlib import Path
import sys,os,argparse,time,subprocess,json
W=Path(__file__).resolve().parent;ROOT=W.parents[1];sys.path[:0]=[str(W),str(ROOT/'src'),str(ROOT/'scripts')]
p=argparse.ArgumentParser();p.add_argument('mode',choices=['node','train']);p.add_argument('--index',type=int);p.add_argument('--config',required=True);p.add_argument('--resume',required=True);a=p.parse_args()
if a.mode=='node':
 c=json.loads(Path(a.config).read_text());out=Path(c['output']);cl=json.loads((out/'cluster.json').read_text());n=cl['nodes'][a.index];offset=sum(len(x['gpus'])for x in cl['nodes'][:a.index]);children=[];streams=[]
 try:
  for local,gpu in enumerate(n['gpus']):
   rank=offset+local;env=dict(os.environ,RANK=str(rank),LOCAL_RANK=str(local),WORLD_SIZE='28',LOCAL_WORLD_SIZE=str(len(n['gpus'])),MASTER_ADDR=cl['nodes'][0]['host'],MASTER_PORT=str(cl['master_port']),P2_TRAIN_MEMORY_GIB='108'if n['name']=='node11'else'134')
   f=(out/'rank_logs'/f'rank_{rank:02d}.log').open('a');streams.append(f);children.append(subprocess.Popen([cl['python'],'-u',str(Path(__file__)),'train','--config',a.config,'--resume',a.resume],env=env,stdout=f,stderr=subprocess.STDOUT))
  while any(x.poll()is None for x in children):
   if any(x.poll()not in(None,0)for x in children):raise RuntimeError('Worker failed; inspect rank logs')
   time.sleep(.5)
  assert all(x.returncode==0 for x in children)
 finally:
  for x in children:
   if x.poll()is None:x.terminate()
  for x in children:
   try:x.wait(timeout=15)
   except subprocess.TimeoutExpired:x.kill();x.wait()
  for f in streams:f.close()
else:
 import torch
 torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
 from p2_whisper_av.bootstrap import activate_ltx
 activate_ltx()
 import runtime28 as R
 from p2_listen.contracts import read,sha,atomic_json
 from p2_ltx_all_lora_noevent.config import load
 from gate import record
 c=R.validate(read(a.config));ready=read(W/'READY.json');assert ready['source']==R.source()and ready['config_sha256']==sha(a.config)
 rank=int(os.environ['RANK']);assert int(os.environ['WORLD_SIZE'])==28
 physical=list(map(int,os.environ['CUDA_VISIBLE_DEVICES'].split(',')))[int(os.environ['LOCAL_RANK'])]
 atomic_json(Path(c['output'])/'rank_logs'/f'worker_{rank:02d}.json',dict(**record(),host=__import__('socket').gethostname(),rank=rank,world=28,physical_gpu=physical))
 print(json.dumps(dict(phase='loading',rank=rank,world=28)),flush=True)
 print(json.dumps(R.load().train(c,Path(a.resume))),flush=True)
