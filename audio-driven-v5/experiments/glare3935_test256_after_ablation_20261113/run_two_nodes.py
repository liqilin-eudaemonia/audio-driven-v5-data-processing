"""Dispatch all24 logical shards over node12/node11's16 GPUs; native workers unchanged."""
import collections,fcntl,os,signal,subprocess,sys,time,traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import run as trial
m=trial.m;BASE=trial.BASE;WORK=trial.WORK;read,save=trial.read,trial.save
UPSTREAM=trial.ROOT/'experiments/p2_ablation12_test64_qwen7500_24gpu_20261113'

def occupied_by_ablation():
 if (UPSTREAM/'COMPLETE.json').exists():return set()
 state=read(UPSTREAM/'status.json')
 return {(j['node'],j['gpu']) for j in state.get('active',[])}

def slots():
 c=read(BASE/'cluster.json')
 return [(rank,n,g) for rank,(n,g) in enumerate((n,g) for n in c['nodes'] for g in n['gpus']) if n['name'] in ['node12','node11']]

def choose_job(shards,generated,scored,busy):
 # Shard IDs remain stable for caches and metrics; they are no longer tied to a GPU.
 for slot,ids in enumerate(shards):
  if slot not in busy and not ids.issubset(generated):return 'generate',slot
 for slot,ids in enumerate(shards):
  if slot not in busy and slot not in scored and ids.issubset(generated):return 'score',slot
 return None

def run_step():
 active={};attempts=collections.Counter();failures=collections.Counter();p=read(WORK/'protocol.json');started=time.time()
 shards=[{s['id'] for s in read(WORK/'blocks'/f'{i:03d}.json')} for i in range(24)]
 nodes=[n for n in read(BASE/'cluster.json')['nodes'] if n['name'] in ['node12','node11']]
 try:
  while not (BASE/'STOP').exists():
   for rank,j in list(active.items()):
    rc=j['proc'].poll()
    if rc is None:continue
    j['proc'].stdin.close();del active[rank];task=(j['stage'],j['slot'])
    if rc:
     failures[task]+=1
     if failures[task]>=3:raise RuntimeError('Repeated worker failure: '+j['log'])
     if j['stage']=='score':
      folder=WORK/'metrics'/f"{j['slot']:03d}"
      if folder.exists() and not (folder/'SUCCESS.json').exists():folder.rename(folder.with_name(folder.name+f'.failed.{time.time_ns()}'))
   generated={s['id'] for s in p['samples'] if (WORK/'seamless/generated'/s['id']/'complete.json').exists()}
   scored={i for i in range(24) if (WORK/'metrics'/f'{i:03d}/SUCCESS.json').exists()}
   if len(generated)==256 and len(scored)==24 and not active:break
   with ThreadPoolExecutor(2) as pool:reports=dict(zip([n['name'] for n in nodes],pool.map(m.probe,nodes)))
   reserved=occupied_by_ablation()
   for rank,n,g in slots():
    if rank in active or (n['name'],g) in reserved or reports[n['name']]['gpus'][g]['free']<48*1024:continue
    assert reports[n['name']]['gpus'][g]['uuid']==read(BASE/'ACTIVATED.json')['uuids'][str(rank)]
    job=choose_job(shards,generated,scored,{j['slot'] for j in active.values()})
    if job:
     attempts[job]+=1;active[rank]=m.launch(job[0],rank,job[1])
   save(WORK/'status.json',dict(at=time.time(),step=3935,generated=len(generated),total=256,scored_shards=len(scored),total_shards=24,active=[{k:v for k,v in j.items() if k!='proc'} for j in active.values()],elapsed=time.time()-started,physical_gpus=16,nodes=['node12','node11'],excluded_nodes=['node8'],scheduler='any idle authorized GPU may claim any remaining shard'))
   time.sleep(5)
  if (BASE/'STOP').exists():return False
  with (WORK/'merge.log').open('a') as log:
   env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH='/private/liqi_workspace/.cache/paired_video_metrics/python',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='2')
   subprocess.run([str(m.METRIC_PYTHON),str(trial.SELF),'merge','--step','3935'],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
  save(WORK/'status.json',dict(at=time.time(),stage='complete',step=3935,generated=256,total=256,scored_shards=24,active=[],physical_gpus=16,nodes=['node12','node11']));return True
 finally:
  for j in active.values():
   if j['proc'].stdin:j['proc'].stdin.close()

def run():
 trial.verify();signal.signal(signal.SIGTERM,lambda *_:sys.exit(143))
 with (BASE/'controller.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);save(BASE/'CONTROLLER.json',dict(pid=os.getpid(),at=time.time(),entry=str(Path(__file__).resolve()),nodes=['node12','node11'],gpus=16))
  nodes=[n for n in read(BASE/'cluster.json')['nodes'] if n['name'] in ['node12','node11']]
  with ThreadPoolExecutor(2) as pool:reports=dict(zip([n['name'] for n in nodes],pool.map(m.probe,nodes)))
  save(BASE/'ACTIVATED.json',dict(at=time.time(),uuids={str(rank):reports[n['name']]['gpus'][g]['uuid'] for rank,n,g in slots()},reports=reports,nodes=['node12','node11'],physical_gpus=16))
  if not (WORK/'COMPLETE.json').exists() and not run_step():return
  save(BASE/'COMPLETE.json',dict(success=True,steps=[3935],at=time.time(),samples=256,seed=137,nodes=['node12','node11'],physical_gpus=16))
if __name__=='__main__':
 try:run()
 except BaseException:
  save(BASE/'FAILURE_two_nodes.json',dict(at=time.time(),traceback=traceback.format_exc()));raise
