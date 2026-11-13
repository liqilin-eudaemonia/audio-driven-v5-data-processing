"""Scheduling-only migration; all frozen workers, weights, inputs and scores reused."""
from run import *

def per_checkpoint_limit(state):
 pending=sum(v['remaining']>0 and not v['merged'] for v in state.values())
 return max(2,20//max(1,pending))

def queue():
 from p2_progressive_eval_20261108 import ResourcePoller
 verify();assert '172.29.252.37' in subprocess.check_output(['hostname','-I'],text=True).split();assert (WORK/'ACTIVATED.json').exists()
 signal.signal(signal.SIGTERM,lambda *_:sys.exit(143))
 with (WORK/'controller.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);save(WORK/'CONTROLLER.json',dict(at=time.time(),pid=os.getpid(),host=socket.gethostname(),entry=str(Path(__file__).resolve()),scheduler='dynamic-per-checkpoint-share-v2'))
  poller=ResourcePoller(config(),probe,interval=10);active={};attempts=collections.Counter();failures=collections.Counter();merges={};published=set();next_update=0
  def start(stage,rank,key='',q=0,shard=0):
   task=(stage,key,q,shard) if stage in ['cache','metric'] else (stage,rank,key)
   attempts[task]+=1;active[task]=launch(stage,rank,key,q,shard,attempts[task])
  try:
   while not stopped():
    for task,j in list(active.items()):
     rc=j['process'].poll()
     if rc is None:continue
     if j['process'].stdin:j['process'].stdin.close()
     del active[task]
     if rc:
      failures[task]+=1
      if failures[task]>=(6 if rc==255 else 3):raise RuntimeError('Repeated worker error:'+j['log'])
    for key,p in list(merges.items()):
     if p.poll() is None:continue
     del merges[key]
     if p.returncode:raise RuntimeError('Merge failed:'+key)
    if (WORK/'INTEGRITY_FAILED.json').exists():raise RuntimeError('Checkpoint integrity failed')
    state=snapshot();reports=poller.snapshot();used={j['rank'] for j in active.values()};started=collections.Counter()
    for s in sorted(slots(),key=lambda s:({'node12':0,'node11':1,'node8':2}[s['node']],s['gpu'])):
     rank=s['rank'];rep=reports.get(s['node'],{})
     if rank in used or started[s['node']]>=2 or 'gpus' not in rep:continue
     g=rep['gpus'][str(s['gpu'])]
     free=g['free']/1024
     if free<44:continue
     pending_cache=[(q,i) for q in QWENS for i in range(4) if not (cache_dir(q)/f'CACHE_COMPLETE_{i}.json').exists() and ('cache','',q,i) not in active]
     counts=collections.Counter(j['key'] for j in active.values() if j['stage']=='render')
     candidates=[k for k,v in state.items() if not v['reused'] and v['cache_ready'] and v['remaining']>counts[k] and counts[k]<per_checkpoint_limit(state) and free>=102 and (WORK/'integrity'/f'{k}.json').exists()]
     chosen=max(candidates,key=lambda k:(-counts[k],state[k]['remaining'])) if candidates else None
     score=None
     for k,v in state.items():
      if v['reused'] or v['merged']:continue
      for i in range(4):
       if i not in v['scores'] and ('metric',k,0,i) not in active and all(x['key'] in v['done'] for x in read(child(k)/'shards'/f'{i:02d}.json')):score=(k,i);break
      if score:break
     backlog=sum(len(v['latent']-v['done']) for v in state.values());decoders=sum(j['stage']=='decode' for j in active.values());scorers=sum(j['stage']=='metric' for j in active.values())
     if pending_cache:q,i=pending_cache[0];start('cache',rank,q=q,shard=i)
     elif score and (scorers<4 or chosen is None):
      k,i=score
      if score not in published:publish(k,i);published.add(score)
      start('metric',rank,k,shard=i)
     elif backlog>max(0,decoders*8) and (decoders<4 or chosen is None):start('decode',rank)
     elif chosen:start('render',rank,chosen)
     elif backlog>decoders:start('decode',rank)
     else:continue
     used.add(rank);started[s['node']]+=1
    for key,v in state.items():
     if not v['reused'] and not v['merged'] and len(v['scores'])==4 and key not in merges:
      env=dict(os.environ,**environment('merge',0));env['CUDA_VISIBLE_DEVICES']='';log=WORK/'logs'/f'merge_{key}.log'
      with log.open('a') as f:merges[key]=subprocess.Popen([str(METRIC_PYTHON),str(ROOT/'scripts/p2_testsets_eval.py'),'merge','--work',str(child(key))],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    finished=all(v['merged'] for v in state.values()) and not active and not merges
    if time.time()>=next_update or finished:
     info=dict(at=time.time(),stage='complete_with_coverage_report' if finished else 'cache_render_decode_score',generated=sum(len(v['done']) for v in state.values()),latent=sum(len(v['latent']) for v in state.values()),total=768,reused=0,new=768,completed_combinations=[k for k,v in state.items() if v['merged']],active_gpus=len(used),max_gpus=24,per_checkpoint_worker_limit=per_checkpoint_limit(state),scheduler='dynamic-per-checkpoint-share-v2',active=[{k:v for k,v in j.items() if k!='process'} for j in active.values()]);save(WORK/'status.json',info);summaries=results();next_update=time.time()+15
    if finished:
     assert len(summaries)==12
     for v in summaries.values():
      r=v['scores'];assert all(r['coverage'][k]['pairs']==64 for k in ['psnr','ssim','lpips','fid','fvd']) and r['audio']['valid_pairs']==64 and r['audio']['pairs_with_errors']==0
     save(WORK/'COMPLETE.json',dict(success=True,at=time.time(),outputs=768,combinations=12,physical_gpus=24,nodes=['node12','node8','node11'],metrics_coverage='See per-combination coverage; partial emotion/deltaResp possible'));return
    time.sleep(3)
  finally:
   poller.close()
   for j in active.values():
    if j['process'].stdin:j['process'].stdin.close()
   for p in merges.values():
    if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)

if __name__=='__main__':
 try:queue()
 except BaseException:
  save(WORK/'FAILURE_dynamic_scheduler.json',dict(at=time.time(),traceback=traceback.format_exc()));raise
