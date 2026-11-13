"""Prepare only by default; explicit run exports all four stages before scoring."""
import argparse,collections,fcntl,os,signal,subprocess,time,traceback
from concurrent.futures import ThreadPoolExecutor
from common import *
SELF=Path(__file__).resolve()

def probe(n):
 from listen_cluster import transport
 code="import subprocess,json;print(json.dumps(dict(ips=subprocess.check_output(['hostname','-I'],text=True).split(),gpus=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.free,memory.total,utilization.gpu','--format=csv,noheader,nounits'],text=True))))"
 try:
  r=subprocess.run(transport(read(BASE/'cluster.json'),n,[str(PYTHON),'-c',code]),capture_output=True,text=True,timeout=25)
  if r.returncode:raise RuntimeError(r.stderr[-300:])
  v=json.loads(r.stdout);assert n['host'] in v['ips']
  return dict(at=time.time(),gpus={int(a[0]):dict(uuid=a[1].strip(),used=float(a[2]),free=float(a[3]),total=float(a[4]),util=float(a[5])) for line in v['gpus'].splitlines() if (a:=line.split(','))})
 except Exception as e:return dict(at=time.time(),error=repr(e))

def resources():
 nodes=read(BASE/'cluster.json')['nodes']
 with ThreadPoolExecutor(3) as pool:return dict(zip([n['name'] for n in nodes],pool.map(probe,nodes)))

def activate():
 report=resources();busy=[];uuids={}
 for s in slots():
  r=report[s['node']['name']]
  if 'error' in r:busy.append(dict(node=s['node']['name'],error=r['error']));continue
  g=r['gpus'][s['gpu']];uuids[str(s['rank'])]=g['uuid']
  if g['used']>2048:busy.append(dict(node=s['node']['name'],gpu=s['gpu'],used_mib=g['used']))
 save(BASE/'RESOURCE_CHECK.json',dict(at=time.time(),ready24=not busy,busy=busy,reports=report))
 if busy:raise RuntimeError('Not all24 authorized GPUs are free; see RESOURCE_CHECK.json. No GPU workers started; no other tasks stopped.')
 save(BASE/'ACTIVATED.json',dict(at=time.time(),uuids=uuids,nodes=['node12','node8','node11'],gpus=24))

def launch(dataset,seed,stage,rank,shard=0):
 from listen_cluster import transport
 s=slots()[rank];w=work(dataset,seed)
 env=dict(CUDA_VISIBLE_DEVICES=str(s['gpu']),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',PYTORCH_ALLOC_CONF='expandable_segments:True',PYTHONPATH='/private/liqi_workspace/.cache/paired_video_metrics/python' if stage=='metric' else '')
 if stage=='metric':cmd=[str(METRIC_PYTHON),'-u',str(ROOT/'scripts/p2_pipelined_testsets_eval.py'),'metrics','--work',str(w),'--rank',str(shard),'--arms',ARM]
 else:cmd=[str(PYTHON),'-u',str(SELF),'worker','--dataset',dataset,'--seed',str(seed),'--stage',stage,'--rank',str(rank),'--shard',str(shard)]
 command=[str(PYTHON),str(GUARD),'--guard','env',*[k+'='+v for k,v in env.items()],*cmd]
 log=w/'logs'/f'{stage}_{rank}_{shard}_{time.time_ns()}.log';log.parent.mkdir(exist_ok=True)
 with log.open('ab',buffering=0) as f:p=subprocess.Popen(transport(read(BASE/'cluster.json'),s['node'],command),stdin=subprocess.PIPE,stdout=f,stderr=subprocess.STDOUT)
 rec=dict(pid=p.pid,at=time.time(),dataset=dataset,seed=seed,stage=stage,rank=rank,shard=shard,node=s['node']['name'],gpu=s['gpu'],log=str(log))
 save(w/'processes'/f'{stage}_{rank}_{shard}.json',rec);return dict(proc=p,**rec)

def close(active):
 for j in active.values():
  if j['proc'].stdin:j['proc'].stdin.close()
 end=time.time()+10
 while time.time()<end and any(j['proc'].poll() is None for j in active.values()):time.sleep(.2)
 for j in active.values():
  if j['proc'].poll() is None:j['proc'].terminate()

def finish_cache(dataset):
 from p2_testsets_eval.runtime import ready
 cd=conditions(dataset);p=read(cd/'protocol.json');items=read(dataset_work(dataset)/'shared_items.json');truncated=[]
 for x in items:
  f=cd/'conditions'/x['key'];assert ready(f/'qwen') and ready(f/'text') and ready(cd/'whisper'/x['key'])
  plan=read(f/'qwen/plan.json');assert plan['adapter']==p['planner_adapter_identity'] and not plan['target_description_access']
  if plan.get('planner_truncated'):truncated.append(x['key'])
 save(cd/'CACHE_COMPLETE.json',dict(success=True,at=time.time(),samples=len(items),adapter=p['planner_adapter_identity'],truncated_ids=truncated,raw_truncations_retained=True))

def stage_loop(dataset,seed,phase):
 from workers import publish,audit_inference
 w=work(dataset,seed);cd=conditions(dataset);p=read(w/'protocol.json');items=read(dataset_work(dataset)/'shared_items.json');index=CompletionIndex(w,items)
 active={};failures=collections.Counter();started=time.time()
 try:
  while not stopped(dataset,seed):
   for rank,j in list(active.items()):
    rc=j['proc'].poll()
    if rc is None:continue
    j['proc'].stdin.close();del active[rank]
    task=(j['stage'],j['shard'] if j['stage'] in ['cache','metric'] else rank)
    if rc:
     failures[task]+=1
     if failures[task]>=3:raise RuntimeError('Worker failed3 times: '+j['log'])
   pending_cache=[i for i in range(24) if not cache_ready(dataset,i)] if phase=='cache' else []
   remaining,backlog=index.refresh() if phase=='infer' else (0,0)
   scored={i for i in range(p['workers']) if (f:=w/'progress/metrics'/f'{ARM}_{i:02d}.json').exists() and read(f).get('status')=='complete'} if phase=='score' else set()
   complete=not active and ((phase=='cache' and not pending_cache) or (phase=='infer' and len(index.done)==len(items)) or (phase=='score' and len(scored)==p['workers']))
   state=dict(at=time.time(),dataset=dataset,seed=seed,phase=phase,generated=len(index.done) if phase=='infer' else None,latent=len(index.latent) if phase=='infer' else None,total=len(items),cache_shards_remaining=len(pending_cache),scored_shards=len(scored) if phase=='score' else None,elapsed=time.time()-started,active=[{k:v for k,v in j.items() if k!='proc'} for j in active.values()])
   save(BASE/'status.json',state);save(w/'status.json',state)
   if complete:break
   reports=resources()
   for s in slots():
    rank=s['rank'];rep=reports.get(s['node']['name'],{})
    if rank in active or 'gpus' not in rep:continue
    g=rep['gpus'][s['gpu']];assert g['uuid']==read(BASE/'ACTIVATED.json')['uuids'][str(rank)]
    free=g['free']/1024;stage=None;shard=0
    if phase=='cache':
     busy={j['shard'] for j in active.values()};todo=[i for i in pending_cache if i not in busy]
     if todo and free>=CAPS['cache']+6:stage='cache';shard=todo[0]
    elif phase=='infer':stage=choose_inference(list(active.values()),remaining,backlog,free)
    else:
     busy={j['shard'] for j in active.values()};todo=[i for i in range(p['workers']) if i not in scored and i not in busy]
     if todo and free>=CAPS['metric']+6:stage='metric';shard=todo[0];publish(dataset,seed,shard)
    if stage:active[rank]=launch(dataset,seed,stage,rank,shard)
   time.sleep(10)
  if stopped(dataset,seed):return False
  if phase=='cache':finish_cache(dataset)
  elif phase=='infer':audit_inference(dataset,seed)
  else:
   log=w/'merge.log';env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='2',PYTHONPATH='/private/liqi_workspace/.cache/paired_video_metrics/python')
   with log.open('a') as f:subprocess.run([str(METRIC_PYTHON),str(ROOT/'scripts/p2_testsets_eval.py'),'merge','--work',str(w)],env=env,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
   r=read(w/'RESULTS.json')[ARM+'/'+dataset];assert r['requested_pairs']==r['audio']['requested_pairs']==COUNTS[dataset]
   save(w/'SCORING_COMPLETE.json',dict(success=True,at=time.time(),samples=COUNTS[dataset],coverage=r['coverage'],audio_errors=r['audio']['pairs_with_errors'],native_status=r['status']))
  return True
 finally:close(active)

def run_queue(mode):
 verify();signal.signal(signal.SIGTERM,lambda *_:sys.exit(143))
 marker='INFERENCE_COMPLETE.json' if mode=='infer' else 'SCORING_COMPLETE.json'
 if (BASE/marker).exists():return
 if mode=='score':assert (BASE/'INFERENCE_COMPLETE.json').exists(),'Finish all four inference/export stages before manual scoring'
 with (BASE/'controller.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);save(BASE/'CONTROLLER.json',dict(at=time.time(),pid=os.getpid(),mode=mode))
  if stopped():return
  activate()
  for dataset,seed in ORDER:
   if stopped(dataset,seed):return
   if (work(dataset,seed)/marker).exists():continue
   if mode=='infer':
    if not (conditions(dataset)/'CACHE_COMPLETE.json').exists() and not stage_loop(dataset,seed,'cache'):return
    if not stage_loop(dataset,seed,'infer'):return
   elif not stage_loop(dataset,seed,'score'):return
  save(BASE/marker,dict(success=True,at=time.time(),stages=[dict(dataset=d,seed=s,samples=COUNTS[d]) for d,s in ORDER],total_outputs=2*sum(COUNTS.values()),dual_seed_aggregation=False))
  save(BASE/'status.json',dict(at=time.time(),stage='all_inference_and_exports_complete' if mode=='infer' else 'all_single_seed_scores_complete',active=[],automatic_next_action=False))

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','check','resources','run','score','worker']);ap.add_argument('--dataset',choices=list(COUNTS),default='seamless');ap.add_argument('--seed',type=int,choices=SEEDS,default=137);ap.add_argument('--stage',choices=['cache','render','decode']);ap.add_argument('--rank',type=int,default=0);ap.add_argument('--shard',type=int,default=0);a=ap.parse_args()
 try:
  if a.action=='prepare':
   from prepare import prepare
   print(json.dumps(prepare()['scope'],ensure_ascii=False))
  elif a.action=='check':print(json.dumps(verify()['scope'],ensure_ascii=False))
  elif a.action=='resources':print(json.dumps(resources(),ensure_ascii=False))
  elif a.action in ['run','score']:run_queue('infer' if a.action=='run' else 'score')
  else:
   from workers import worker
   worker(a.dataset,a.seed,a.stage,a.rank,a.shard)
 except BaseException:
  save(BASE/f'FAILURE_{a.action}_{a.dataset}_{a.seed}_{a.stage}_{a.rank}_{a.shard}.json',dict(at=time.time(),traceback=traceback.format_exc()));raise
