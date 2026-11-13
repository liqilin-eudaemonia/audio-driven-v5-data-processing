from pathlib import Path
import sys,os,json,time,subprocess,argparse,fcntl,threading,traceback
W=Path(__file__).resolve().parent
sys.path.insert(0,str(W))
import runtime as R
from p2_listen.contracts import read,sha,atomic_json,digest
CL=read(R.NOEVENT/'cluster.json');PY=CL['python']
def cluster(arm):
 import copy
 c=copy.deepcopy(CL);c['master_port']={'no_event':29981,'no_feature':29973,'no_whisper':29974}[arm];return c
def resources():
 from listen_cluster import transport
 from concurrent.futures import ThreadPoolExecutor
 def one(n):
  cmd=['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits']
  for attempt in range(6):
   p=subprocess.run(transport(CL,n,cmd),capture_output=True,text=True,timeout=40)
   if p.returncode==0:break
   time.sleep(2)
  if p.returncode:raise RuntimeError(n['name']+p.stderr)
  vals={int(l.split(',')[0]):int(l.split(',')[1]) for l in p.stdout.splitlines()}
  return dict(node=n['name'],selected={g:vals[g] for g in n['gpus']},idle=all(vals[g]<500 for g in n['gpus']))
 with ThreadPoolExecutor(4)as e:result=list(e.map(one,CL['nodes']))
 atomic_json(W/'RESOURCES.json',result);return all(x['idle'] for x in result)
def prepare():
 import torch
 torch.set_num_threads(2)
 for arm in ['no_feature','no_whisper']:
  out=R.output(arm);out.mkdir(exist_ok=True);(out/'rank_logs').mkdir(exist_ok=True)
  from p2_ltx_all_lora_components.config import expected
  c=expected(out,arm);c['batch28_ablation']={'world':28,'reference_samples':186000};c['training'].update(global_batch=28,steps=10151)
  R.validate(c);atomic_json(out/'config.json',c,immutable=True);atomic_json(out/'cluster.json',cluster(arm),immutable=True)
  m=R.load();v=m.preflight(c)
  assert v['samples']==284218
 atomic_json(W/'PREPARED.json',dict(success=True,targets=R.TARGETS,world=28,batch=28,reserved='node11:4-7',source=R.source()))
def node(arm,index,resume):
 c=cluster(arm);n=c['nodes'][index];out=R.output(arm);off=sum(len(x['gpus'])for x in c['nodes'][:index]);children=[];streams=[]
 try:
  for local,gpu in enumerate(n['gpus']):
   rank=off+local;env=dict(os.environ,RANK=str(rank),LOCAL_RANK=str(local),WORLD_SIZE='28',LOCAL_WORLD_SIZE=str(len(n['gpus'])),MASTER_ADDR=c['nodes'][0]['host'],MASTER_PORT=str(c['master_port']),P2_TRAIN_MEMORY_GIB='108' if n['name']=='node11' else '134')
   f=(out/'rank_logs'/f'rank_{rank:02d}.log').open('a');streams.append(f)
   cmd=[PY,'-u',str(Path(__file__)),'train','--arm',arm]
   if resume:cmd+=['--resume',resume]
   children.append(subprocess.Popen(cmd,env=env,stdout=f,stderr=subprocess.STDOUT))
  while any(p.poll() is None for p in children):
   if any(p.poll() not in (None,0) for p in children):raise RuntimeError('Rank failed: '+arm)
   time.sleep(1)
  assert all(p.returncode==0 for p in children)
 finally:
  for p in children:
   if p.poll() is None:p.terminate()
  for p in children:
   try:p.wait(timeout=20)
   except subprocess.TimeoutExpired:p.kill();p.wait()
  for f in streams:f.close()
def train(arm,resume):
 import torch
 torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
 from p2_whisper_av.bootstrap import activate_ltx
 activate_ltx();out=R.output(arm);c=read(out/'config.json')
 if arm=='no_event':m=R.noevent_runtime().load()
 else:m=R.load()
 atomic_json(out/'rank_logs'/f"placement_{int(os.environ['RANK']):02d}.json",dict(pid=os.getpid(),host=__import__('socket').gethostname(),rank=int(os.environ['RANK']),visible=os.environ['CUDA_VISIBLE_DEVICES'],local=int(os.environ['LOCAL_RANK']),target=R.TARGETS[arm]))
 print(m.train(c,Path(resume) if resume else None,R.TARGETS[arm]),flush=True)
def monitor(done):
 while not done.wait(3600):
  rows={a:read(R.output(a)/'status.json') for a in R.TARGETS if (R.output(a)/'status.json').exists()}
  with (W/'hourly.jsonl').open('a')as f:f.write(json.dumps(dict(at=time.time(),statuses=rows))+'\n')
def run():
 from listen_cluster import launch
 done=threading.Event();threading.Thread(target=monitor,args=(done,),daemon=True).start()
 with (W/'queue.lock').open('a')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  try:
   for arm in R.TARGETS:
    out=R.output(arm);target=R.TARGETS[arm]
    if (W/(arm+'_ACCEPTED.json')).exists():continue
    if (W/'STOP').exists():return
    if not resources():raise RuntimeError('Selected training GPUs occupied; other tasks preserved')
    if arm=='no_event':
     resume=str(out/'step_005818.pt');assert read(Path(resume).with_suffix('.complete.json'))['step']==5818
     if (out/'STOP').exists(): (out/'STOP').rename(W/('noevent_previous_STOP_'+str(time.time_ns())))
    else:resume=None
    if (out/'STOP').exists():raise RuntimeError('Arm STOP set')
    c=cluster(arm);commands=[]
    for i,n in enumerate(c['nodes']):
     cmd=['env','CUDA_VISIBLE_DEVICES='+','.join(map(str,n['gpus'])),'PYTHONUNBUFFERED=1',*[k+'='+str(v)for k,v in c['network'].items()],PY,'-u',str(Path(__file__)),'node','--arm',arm,'--index',str(i)]
     if resume:cmd+=['--resume',resume]
     commands.append(cmd)
    atomic_json(W/'STATUS.json',dict(stage='training',arm=arm,target=target,resume=resume,at=time.time()))
    launch(c,commands,W/(arm+'_nodes_'+str(time.time_ns())))
    assert read(out/'status.json')['step']==target
    path=out/f'step_{target:06d}.pt'
    if arm=='no_event':cp=R.noevent_runtime().load().checkpoint
    else:cp=R.load().checkpoint
    v=cp.checked(path);assert v['step']==target and len(v['rank_rng'])==28
    del v
    atomic_json(W/(arm+'_ACCEPTED.json'),dict(success=True,step=target,checkpoint=str(path),sha256=read(path.with_suffix('.complete.json'))['sha256'],at=time.time()))
   atomic_json(W/'COMPLETE.json',dict(success=True,at=time.time(),targets=R.TARGETS))
  except BaseException as e:
   atomic_json(W/'FAILURE.json',dict(error=repr(e),traceback=traceback.format_exc(),at=time.time()));raise
  finally:done.set()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run','node','train']);p.add_argument('--arm',choices=list(R.TARGETS));p.add_argument('--index',type=int);p.add_argument('--resume');a=p.parse_args()
 if a.action=='node':node(a.arm,a.index,a.resume)
 elif a.action=='train':train(a.arm,a.resume)
 else:globals()[a.action]()
