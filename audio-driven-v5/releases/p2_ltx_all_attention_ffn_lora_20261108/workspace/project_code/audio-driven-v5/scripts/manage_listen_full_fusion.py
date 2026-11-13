"""CPU pipeline controller and GPU handoff supervisor, with durable process ownership."""
import argparse,concurrent.futures,hashlib,json,os,signal,socket,subprocess,sys,time,traceback,shlex
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from listen_benchmark.full_fusion import read,atomic,lock,prepare_shard,fuse_shard,ready_raw,file_sha

def alive(pid):
 try:return Path(f'/proc/{pid}/stat').read_text().split(')')[-1].split()[0]!='Z'
 except FileNotFoundError:return False

def owned(pid,run,needle):
 try:
  cmd=Path(f'/proc/{pid}/cmdline').read_bytes().decode()
  return str(run) in cmd and needle in cmd and os.getpgid(pid)==pid
 except (FileNotFoundError,ProcessLookupError):return False

def gpu_info():
 value=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=20)
 return {int(z[0]):dict(memory_used_mib=float(z[1]),memory_total_mib=float(z[2]),gpu_util=float(z[3])) for line in value.splitlines() if (z:=line.split(','))}

def audit_fused_shard(run,shard,state):
 """Verify immutable shard files in parallel; global ID checking stays central."""
 from listen_benchmark.full_fusion import jsonl
 run=Path(run)
 for path,meta in state['files'].items():
  if file_sha(run/path)!=meta['sha256']:raise AssertionError(f'Final checksum mismatch: {shard}/{path}')
 ids=[row['sample_id'] for row in jsonl(run/f'labels/{shard}.jsonl')]
 if len(ids)!=state['counts']['samples']:raise AssertionError(f'Final shard count mismatch: {shard}')
 if len(ids)!=len(set(ids)):raise AssertionError(f'Repeated shard sample ID: {shard}')
 return shard,ids

def node_loop(run,node):
 cfg=read(run/'config.json');raw=Path(cfg['raw_run']);jobs=cfg['nodes'][node]['jobs'];folder=run/'nodes'/node;folder.mkdir(parents=True,exist_ok=True);children={};retry={};terminating={}
 with lock(folder/'supervisor.lock') as acquired:
  if not acquired:raise RuntimeError('Node supervisor already running')
  for j in jobs:
   gpu=j['gpu'];rec=read(folder/f'g{gpu}.json')
   if rec.get('pid') and alive(rec['pid']):
    if not owned(rec['pid'],run,'run_listen_full_fusion_semantic.py'):raise RuntimeError('Foreign process at recorded PID')
    children[gpu]=rec['pid']
  while not (run/'STOP').exists() and not (run/'COMPLETE.json').exists():
   now=time.time();info={g:v for g,v in gpu_info().items() if g in {j['gpu'] for j in jobs}};reasons={};states={};raw_summary=read(raw/'status.json');raw_branches=raw_summary.get('branches',{})
   all_ingested=(run/'INGEST_COMPLETE.json').exists();semantic_left=sum(not (run/f'state/semantic/{p.stem}.json').exists() for p in (run/'state/ingest').glob('*.json'))
   for j in jobs:
    gpu=j['gpu'];worker=f'{node}-g{gpu}';pid=children.get(gpu)
    if pid and alive(pid):
     state=read(run/f'workers/{worker}.json');states[str(gpu)]=state
     # Stalls terminate only our own verified process group, then resume its shard.
     launched=read(folder/f'g{gpu}.json').get('utc',now)
     last_heartbeat=max(launched,state.get('heartbeat',launched) if state.get('pid')==pid else launched)
     if (gpu in terminating or now-last_heartbeat>cfg.get('worker_stall_s',1800)) and owned(pid,run,'run_listen_full_fusion_semantic.py'):
      if gpu not in terminating:
       os.killpg(pid,signal.SIGTERM);terminating[gpu]=now
       atomic(folder/f'alert-g{gpu}.json',dict(utc=now,reason='fusion worker stale; restart scheduled',pid=pid))
      elif now-terminating[gpu]>60:os.killpg(pid,signal.SIGKILL)
     elif state.get('pid')==pid and state.get('status') in ('inference','shard_complete'):
      alert=folder/f'alert-g{gpu}.json'
      if alert.exists() and not read(alert).get('resolved_at'):atomic(alert,dict(read(alert),resolved_at=now))
     continue
    if pid:
     try:os.waitpid(pid,os.WNOHANG)
     except ChildProcessError:pass
     state=read(run/f'workers/{worker}.json');children.pop(gpu,None);terminating.pop(gpu,None)
     if state.get('status')!='finished':
      attempt,retry_at=retry.get(gpu,(0,0));attempt+=1;retry[gpu]=(attempt,now+min(300,10*2**min(attempt,5)));atomic(folder/f'alert-g{gpu}.json',dict(utc=now,reason='fusion worker exited; backoff/restart',attempt=attempt))
    if all_ingested and not semantic_left:reasons[str(gpu)]='semantic_queue_complete';continue
    if not semantic_left:reasons[str(gpu)]='waiting_for_ready_ingest';continue
    if now<retry.get(gpu,(0,0))[1]:reasons[str(gpu)]='restart_backoff';continue
    rawrec=read(raw/f'nodes/{node}/g{gpu}.json');rawpid=rawrec.get('pid');w=read(raw/f'workers/{worker}.json')
    if rawpid and alive(rawpid):reasons[str(gpu)]='raw_worker_still_alive';continue
    if w.get('status')!='finished_available_shards':reasons[str(gpu)]='raw_worker_not_confirmed_finished';continue
    branch=w.get('branch');bs=raw_branches.get(branch,{})
    all_raw_branch_done=bool(bs) and bs.get('shards_done')==bs.get('total_shards')
    if branch in ('audio','asr') and not all_raw_branch_done:reasons[str(gpu)]='raw_fast_branch_not_globally_finished';continue
    # Existing raw monitor may reassign a just-finished fast branch to visual work.
    allocation=read(raw/'allocations.json').get(worker,branch)
    if allocation!=branch:reasons[str(gpu)]='raw_handoff_pending';continue
    visual_left=raw_branches.get('visual',{}).get('total_shards',1024)-raw_branches.get('visual',{}).get('shards_done',0)
    if branch in ('audio','asr') and visual_left>24:reasons[str(gpu)]='raw_visual_has_priority';continue
    if info[gpu]['memory_used_mib']>256:reasons[str(gpu)]='GPU_context_not_released_or_other_job';continue
    # Slot lock prevents duplicate integration launch after manager restart.
    with lock(folder/f'launch-g{gpu}.lock') as slot:
     if not slot:continue
     env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',VLLM_WORKER_MULTIPROC_METHOD='spawn',LD_LIBRARY_PATH=cfg['vllm_library_path'])
     cmd=[cfg['python'],str(ROOT/'scripts/run_listen_full_fusion_semantic.py'),'--run',str(run),'--worker',worker,'--rank',str(j['rank']),'--batch',str(cfg['semantic_batch']),'--memory',str(cfg['semantic_memory'])]
     logpath=folder/f'g{gpu}-{int(now)}.log'
     with logpath.open('a') as log:child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
     children[gpu]=child.pid;atomic(folder/f'g{gpu}.json',dict(pid=child.pid,command=cmd,log=str(logpath),utc=now,from_raw_pid=rawpid))
     atomic(run/f'handoffs/{worker}.json',dict(utc=now,raw_pid=rawpid,new_pid=child.pid,branch='text_semantic',gpu=gpu,node=node));reasons[str(gpu)]='launched'
   atomic(folder/'status.json',dict(node=node,host=socket.gethostname(),pid=os.getpid(),heartbeat=now,gpus=info,children=children,workers=states,waiting=reasons,semantic_ready_shards=semantic_left,alerts=[read(p) for p in folder.glob('alert-g*.json') if not read(p).get('resolved_at')]))
   time.sleep(cfg.get('poll_seconds',10))
  atomic(folder/'finished.json',dict(utc=time.time(),reason='stop' if (run/'STOP').exists() else 'complete'))

def ensure_node_watchdogs(run,cfg):
    """Restart a missing remote integration watchdog; shared lock prevents duplicates."""
    for node,settings in cfg['nodes'].items():
        marker=run/f'watchdogs/node-{node}.json';state=read(marker)
        if time.time()-state.get('heartbeat',0)<90:continue
        cmd=[cfg['manager_python'],str(ROOT/'scripts/manage_listen_full_fusion.py'),'--run',str(run),'--role','node','--node',node,'--watchdog']
        remote_script="import subprocess; from pathlib import Path; p=Path("+repr(str(run/f'watchdogs/bootstrap-{node}.log'))+"); f=p.open('a'); c=subprocess.Popen("+repr(cmd)+",stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True); print(c.pid)"
        command=['ssh','-i',cfg['ssh_key'],'-o','BatchMode=yes','-o','ConnectTimeout=8','root@'+settings['host'],shlex.join([cfg['manager_python'],'-c',remote_script])]
        try:
            result=subprocess.run(command,capture_output=True,text=True,timeout=20)
            if result.returncode:raise RuntimeError(result.stderr[-1500:])
            atomic(run/f'watchdogs/bootstrap-{node}.json',dict(utc=time.time(),pid=result.stdout.strip()))
        except Exception as exc:atomic(run/f'alerts/watchdog-{node}.json',dict(utc=time.time(),error=repr(exc)))

def controller_loop(run):
 cfg=read(run/'config.json');raw=Path(cfg['raw_run']);expected=sorted(read(raw/'queue/index.json')['shards']);expected_samples=read(raw/'queue/index.json')['counts']['queued'];preview=Path(cfg['preview']);preview.mkdir(parents=True,exist_ok=True);pending={};attempts={};last_scan=0
 with lock(run/'controller.lock') as acquired:
  if not acquired:raise RuntimeError('Controller already running')
  with concurrent.futures.ProcessPoolExecutor(max_workers=cfg['cpu_workers']) as pool:
   while not (run/'STOP').exists():
    now=time.time()
    if now-last_scan>60:
     ensure_node_watchdogs(run,cfg);last_scan=now
    for future,(stage,shard) in list(pending.items()):
     if not future.done():continue
     pending.pop(future)
     try:
      future.result();attempts.pop((stage,shard),None)
      alert=run/f'alerts/{stage}-{shard}.json'
      if alert.exists():atomic(alert,dict(read(alert),resolved_at=now))
     except Exception as e:
      n=attempts.get((stage,shard),(0,0))[0]+1;attempts[(stage,shard)]=(n,now+min(300,15*n));atomic(run/f'alerts/{stage}-{shard}.json',dict(utc=now,attempt=n,error=repr(e),traceback=traceback.format_exc()[-6000:]))
    ingest={p.stem:read(p) for p in (run/'state/ingest').glob('*.json')};semantic={p.stem:read(p) for p in (run/'state/semantic').glob('*.json')};fusion={p.stem:read(p) for p in (run/'state/fusion').glob('*.json')}
    active=set(pending.values());tasks=[('fusion',s) for s in expected if s in semantic and s not in fusion]+[('ingest',s) for s in expected if s not in ingest]
    for stage,shard in tasks:
     if len(pending)>=cfg['cpu_workers']:break
     if (stage,shard) in active or now<attempts.get((stage,shard),(0,0))[1]:continue
     if stage=='ingest' and not ready_raw(raw,shard):continue
     future=pool.submit(prepare_shard if stage=='ingest' else fuse_shard,str(run),shard);pending[future]=(stage,shard);active.add((stage,shard))
    counts=Counter();perclass=Counter()
    for x in fusion.values():counts.update(x['counts']);perclass.update(x['per_class'])
    prep=Counter()
    for x in ingest.values():prep.update(x['counts'])
    if len(ingest)==len(expected) and not (run/'INGEST_COMPLETE.json').exists():
     if prep['samples']!=expected_samples:raise AssertionError('Ingest total sample mismatch')
     atomic(run/'INGEST_COMPLETE.json',dict(utc=now,shards=len(ingest),samples=prep['samples']))
    nodes={p.parent.name:read(p) for p in (run/'nodes').glob('*/status.json')};workers={p.stem:read(p) for p in (run/'workers').glob('*.json')}
    summary=dict(run_id=run.name,heartbeat=now,pid=os.getpid(),raw_run=str(raw),expected_samples=expected_samples,expected_shards=len(expected),ingested_shards=len(ingest),semantic_done_shards=len(semantic),fused_shards=len(fusion),prepared_counts=dict(prep),output_counts=dict(counts),per_class=dict(perclass),semantic_job_references=sum(x['unique_semantic_jobs'] for x in ingest.values()),semantic_failed_references=sum(x['failed'] for x in semantic.values()),active_cpu_tasks=list(pending.values()),alerts=[read(p) for p in (run/'alerts').glob('*.json') if not read(p).get('resolved_at')],nodes=nodes,workers=workers,raw_status_file=str(raw/'status.json'),raw_must_be_preserved=True,training_admitted=False)
    atomic(run/'status.json',summary);atomic(preview/'status.json',summary)
    with (run/'history.jsonl').open('a') as f:f.write(json.dumps({k:summary[k] for k in ['heartbeat','ingested_shards','semantic_done_shards','fused_shards','output_counts']})+'\n')
    if len(fusion)==len(expected):
     if counts['samples']!=expected_samples:raise AssertionError('Final sample count mismatch')
     # Audit every shard checksum and globally unique sample ID before declaring completion.
     seen=set()
     audited=0
     audits=[pool.submit(audit_fused_shard,str(run),shard,fusion[shard]) for shard in expected]
     for future in concurrent.futures.as_completed(audits):
      shard,ids=future.result()
      for sid in ids:
       if sid in seen:raise AssertionError('Repeated global sample ID')
       seen.add(sid)
      audited+=1
      summary.update(heartbeat=time.time(),final_audit_shard=shard,final_audit_done_shards=audited,final_audit_workers=cfg['cpu_workers'])
      atomic(run/'status.json',summary);atomic(preview/'status.json',summary)
     if len(seen)!=expected_samples:raise AssertionError('Final unique sample count mismatch')
     index=dict(schema='listen-full-fusion-index-v1',samples=expected_samples,shards=[dict(fusion[s],shard=s,labels=f'labels/{s}.jsonl',evidence=f'evidence/{s}.jsonl',issues=f'issues/{s}.jsonl') for s in expected],raw_run=str(raw),training_admitted=False,completion_status='completed_with_issues' if counts['samples_with_raw_issues'] or counts['semantic_failed_events'] or counts['unlocalized_asr_units'] else 'completed_candidates')
     atomic(run/'index.json',index);atomic(run/'COMPLETE.json',dict(utc=time.time(),index_sha256=file_sha(run/'index.json'),samples=expected_samples,completion_status=index['completion_status']));return
    time.sleep(cfg.get('poll_seconds',10))

def watchdog(run,role,node):
 cfg=read(run/'config.json');identity=role+('-'+node if node else '');folder=run/'watchdogs';folder.mkdir(parents=True,exist_ok=True)
 with lock(folder/(identity+'.lock')) as acquired:
  if not acquired:raise RuntimeError('Watchdog already running')
  marker=folder/(identity+'.json');previous=read(marker);child=None;pid=previous.get('child_pid');attempt=previous.get('restarts',0)
  if pid and alive(pid) and not owned(pid,run,'manage_listen_full_fusion.py'):raise RuntimeError('Foreign child PID')
  terminate_at=None;launched_at=time.time()
  while not (run/'STOP').exists() and not (run/'COMPLETE.json').exists():
   heartbeat=read(run/'status.json' if role=='controller' else run/f'nodes/{node}/status.json')
   if pid and alive(pid) and time.time()-launched_at>120 and time.time()-heartbeat.get('heartbeat',time.time())>1200 and owned(pid,run,'manage_listen_full_fusion.py'):
    if terminate_at is None:os.killpg(pid,signal.SIGTERM);terminate_at=time.time()
    elif time.time()-terminate_at>60:os.killpg(pid,signal.SIGKILL)
   if not pid or not alive(pid):
    if child is not None:child.wait()
    cmd=[sys.executable,str(Path(__file__).resolve()),'--run',str(run),'--role',role]+(['--node',node] if node else [])
    with (folder/(identity+f'-{int(time.time())}.log')).open('a') as log:child=subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    pid=child.pid;attempt+=1;terminate_at=None;launched_at=time.time()
   atomic(marker,dict(pid=os.getpid(),child_pid=pid,role=role,node=node,heartbeat=time.time(),restarts=attempt-1,command_scope=str(run)))
   time.sleep(10)
  atomic(marker,dict(pid=os.getpid(),child_pid=pid,heartbeat=time.time(),status='stopped' if (run/'STOP').exists() else 'complete'))

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--role',choices=['controller','node'],required=True);p.add_argument('--node',default='');p.add_argument('--watchdog',action='store_true');a=p.parse_args();run=a.run.resolve()
 if a.watchdog:watchdog(run,a.role,a.node)
 elif a.role=='controller':controller_loop(run)
 else:node_loop(run,a.node)
if __name__=='__main__':main()
