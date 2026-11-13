#!/usr/bin/env python3
"""Bounded health window; restores only this pipeline's owned processes."""
import argparse,concurrent.futures,json,os,socket,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from listen_benchmark.j_annotation import read,write,acquire
KEY='/private/ssh_keys/root_dedicated-developjob-lq-dev-machine-1nkpf_ed25519'
REMOTE=r'''
import os,sys,time,signal,socket,subprocess,json
from pathlib import Path
sys.path.insert(0,ROOT+'/scripts');sys.path.insert(0,ROOT+'/src')
from listen_benchmark.j_annotation import read,write,acquire
from listen_j_annotation_supervisor import alive,identity,spawn
from listen_j_omni import environment,complete
def read_heartbeat(path):
 last_error=None
 for attempt in range(3):
  try:
   h=read(path)
   for field in ('pid','stage','completed_clips','heartbeat'):
    if field not in h:raise ValueError('missing heartbeat field: '+field)
   return h
  except (OSError,ValueError,TypeError) as error:
   last_error=error
   if attempt<2:time.sleep(.05*(attempt+1))
 raise RuntimeError('heartbeat observation unavailable: '+str(path)+': '+repr(last_error))
def read_process_record(path):
 # Mutable metadata may be briefly unobservable on shared storage. Avoid a
 # separate exists() check and re-read before publishing a missing process.
 record=None;last_error=None
 for attempt in range(3):
  try:
   record=read(path);last_error=None
   if alive(record):return record
  except (OSError,ValueError) as error:last_error=error
  if attempt<2:time.sleep(.1)
 if last_error is not None and not isinstance(last_error,FileNotFoundError):
  raise RuntimeError('Process metadata observation unavailable: '+str(path)+': '+repr(last_error))
 return record
def watchdog_status(folder,command,env,log_name,allow_restore):
 # A transient metadata miss is not proof that the process exited. Retry the
 # identity read, then honor the owner's lock and any live pending launch.
 record=None;last_error=None
 for attempt in range(3):
  try:
   record=read(folder/'watchdog.json');last_error=None
   if alive(record):return record,None,dict(alive=True,reason='identity_match',read_attempts=attempt+1,pid=record.get('pid'),start_ticks=record.get('start_ticks'))
  except (OSError,ValueError) as error:last_error=error
  if attempt<2:time.sleep(.1)
 observation=dict(alive=False,reason='identity_not_confirmed',read_error=repr(last_error) if last_error else None,observed_identity=record)
 if last_error is not None and not isinstance(last_error,FileNotFoundError):
  observation['reason']='metadata_unavailable';return record,None,observation
 recovery=acquire(folder/'window-watchdog-recovery.lock',False)
 if recovery is None:
  observation['reason']='another_recovery_probe';return record,None,observation
 try:
  try:record=read(folder/'watchdog.json')
  except FileNotFoundError:record=None
  if alive(record):return record,None,dict(alive=True,reason='identity_revalidated',pid=record.get('pid'),start_ticks=record.get('start_ticks'))
  pending_path=folder/'window-watchdog-pending.json'
  try:pending=read(pending_path)
  except FileNotFoundError:pending=None
  if alive(pending):
   observation.update(reason='pending_launch_alive',pending=pending);return record,None,observation
  ownership=acquire(folder/'watchdog.lock',False)
  if ownership is None:
   observation['reason']='owner_lock_held';return record,None,observation
  ownership.close()
  if not allow_restore:
   observation['reason']='restoration_disabled';return record,None,observation
  child=spawn(folder,log_name,command,env);write(pending_path,child)
  observation.update(reason='restoration_launched',pending=child)
  return record,child,observation
 finally:recovery.close()
work=Path(WORK);nw=work/'nodes'/NODE;cfg=read(work/'config.json');assert socket.gethostname()==cfg['nodes'][NODE]['hostname']
heartbeats={gpu:read_heartbeat(work/'workers'/f'{NODE}-gpu-{gpu}.json') for gpu in cfg['nodes'][NODE]['gpu_ids']}
statepath=nw/'window_guard_state.json';previous=read(statepath) if statepath.exists() else {'suspects':{},'pending':{}}
actions=[];now=time.time()
stop=(nw/'STOP_SUPERVISION').exists() or complete(work)
watch,child,watch_observation=watchdog_status(nw,['/private/software/conda/envs/vllm/bin/python',ROOT+'/scripts/listen_j_omni.py','watchdog','--work',WORK,'--node',NODE],environment(),'window-watchdog-recovery',not stop)
if child:actions.append(dict(action='restore_watchdog',child=child))
s=read_process_record(nw/'supervisor.json') or {}
raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=15)
dev={int(x.split(',')[0]):list(map(float,x.split(',')[1:])) for x in raw.strip().splitlines()}
workers=[]
for gpu in cfg['nodes'][NODE]['gpu_ids']:
 key=str(gpu);child=s.get('children',{}).get('gpu-'+key);h=heartbeats[gpu]
 live=alive(child);matching=bool(child and h.get('pid')==child.get('pid'))
 progress=h.get('last_progress_at',h.get('started_at',now));age=now-progress
 token=[child.get('pid'),child.get('start_ticks'),progress,h.get('completed_clips'),h.get('held_clips'),h.get('benchmark_phase'),h.get('benchmark_clips')] if child else None
 suspect=previous['suspects'].get(key,{})
 suspect_count=suspect.get('count',0)+1 if suspect.get('token')==token else 1
 stalled=(live and matching and age>900 and h.get('stage') not in ('loading_model','waiting','stopped','complete') and dev[gpu][1]<=5)
 previous['suspects'][key]=dict(token=token,count=suspect_count if stalled else 0)
 pending=previous['pending'].get(key)
 if pending and now-pending['at']>120 and matching:
  if matching and (progress>pending['progress'] or token!=pending.get('progress_token',token)):
   previous['pending'].pop(key,None)
  else:
   killed=[]
   for rec in reversed(pending['owned_tree']):
    if alive(rec):os.kill(rec['pid'],signal.SIGKILL);killed.append(rec['pid'])
   if killed:actions.append(dict(action='kill_confirmed_stalled_owned_tree',gpu=gpu,pids=killed,evidence=pending))
   previous['pending'].pop(key,None)
 if not stop and stalled and suspect_count>=3 and not pending:
  cmd=Path(f"/proc/{child['pid']}/cmdline").read_bytes().decode().replace('\x00',' ')
  assert 'listen_j_omni.py worker' in cmd and WORK in cmd
  # Snapshot exact descendants and start ticks before signaling; PID reuse is
  # checked again before any escalation. Other jobs are never selected by GPU.
  pairs={}
  for line in subprocess.check_output(['ps','-eo','pid=,ppid='],text=True).splitlines():
   p,pp=map(int,line.split());pairs[p]=pp
  owned=[child['pid']]
  for _ in range(12):
   extra=[p for p,pp in pairs.items() if pp in owned and p not in owned]
   if not extra:break
   owned+=extra
  tree=[dict(pid=p,start_ticks=identity(p)) for p in owned]
  record=dict(at=now,progress=progress,progress_token=token,owned_tree=tree,worker=h,gpu=dev[gpu])
  previous['pending'][key]=record;os.kill(child['pid'],signal.SIGTERM)
  actions.append(dict(action='terminate_confirmed_stalled_worker',gpu=gpu,evidence=record))
 workers.append(dict(gpu=gpu,pid=child.get('pid') if child else None,alive=live,matching_heartbeat=matching,stage=h.get('stage'),completed=h.get('completed_clips',0),progress_age_s=age,heartbeat_age_s=now-h.get('heartbeat',now),memory_mib=dev[gpu][0],gpu_util=dev[gpu][1],execution_revision=h.get('execution_revision'),last_error=(h.get('last_error') or '')[-350:]))
previous['at']=now;write(statepath,previous)
quality=None
quality_workers=cfg['nodes'][NODE].get('visual_quality_workers')
if quality_workers and cfg.get('visual_quality_gate'):
 qw=Path(cfg['visual_quality_gate']['work']);qn=qw/'nodes'/NODE
 qcstate=read_process_record(qn/'controller.json')
 qwstate,child,qwatch_observation=watchdog_status(qn,[cfg['visual_quality_gate']['runtime_python'],ROOT+'/scripts/listen_j_visual_gate_cluster.py','watchdog','--work',str(qw),'--node',NODE,'--workers',str(quality_workers)],dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',GLOG_minloglevel='2'),'window-quality-recovery',not (qw/'STOP').exists() and not (qn/'STOP').exists())
 if child:actions.append(dict(action='restore_quality_watchdog',child=child))
 quality=dict(watchdog_alive=qwatch_observation['alive'],watchdog_observation=qwatch_observation,controller_alive=alive(qcstate),configured_workers=quality_workers,state=qcstate)
cpu=dict(x.split() for x in Path('/sys/fs/cgroup/cpu.stat').read_text().splitlines())
print(__import__('json').dumps(dict(node=NODE,at=now,hostname=socket.gethostname(),watchdog_alive=watch_observation['alive'],watchdog_observation=watch_observation,supervisor_alive=alive(s),supervisor_error=s.get('error'),blocked_gpus=s.get('blocked_gpus',{}),workers=workers,cpu_usage_usec=int(cpu['usage_usec']),visual_quality=quality,actions=actions,stop_requested=stop)))
'''
def probe(work,node,host):
 code=f'ROOT={str(ROOT)!r}\nWORK={str(work)!r}\nNODE={node!r}\n'+REMOTE
 key=read(work/'config.json')['nodes'][node].get('ssh_key',KEY)
 command=['ssh','-i',key,'-o','BatchMode=yes','-o','ConnectTimeout=8','-o','ServerAliveInterval=10','-o','ServerAliveCountMax=2','-o','ControlMaster=auto','-o','ControlPersist=600','-o','ControlPath=/tmp/jomni-guard-%C','root@'+host,'python3 -']
 if socket.gethostname()==read(work/'config.json')['nodes'][node]['hostname']:command=[sys.executable,'-']
 for attempt in range(3):
  try:
   r=subprocess.run(command,input=code,text=True,capture_output=True,timeout=40)
   if r.returncode:raise RuntimeError(r.stderr[-1500:] or r.stdout[-1500:])
   return json.loads(r.stdout.strip().splitlines()[-1])
  except (subprocess.TimeoutExpired,RuntimeError):
   if attempt==2:raise
   time.sleep(2*(attempt+1))
def main(work,start,end,folder=None):
 folder=folder or work/'monitoring_10h';folder.mkdir(parents=True,exist_ok=True);owner=acquire(folder/'leader.lock',False)
 if owner is None:return
 plan=folder/'plan.json'
 if plan.exists():p=read(plan);assert p['start']==start and p['end']==end
 else:write(plan,dict(start=start,end=end,report_interval_s=1800,health_poll_s=30,scope='configured nodes and GPUs; restore owned watchdogs and confirmed stalled workers only'))
 from importlib.util import spec_from_file_location,module_from_spec
 spec=spec_from_file_location('supervisor',ROOT/'scripts/listen_j_annotation_supervisor.py');m=module_from_spec(spec);spec.loader.exec_module(m)
 state=dict(pid=os.getpid(),start_ticks=m.identity(os.getpid()),started_at=time.time(),start=start,end=end,completed_reports=0)
 last=read(folder/'state.json') if (folder/'state.json').exists() else {};reported=set(last.get('reported_slots',[]));cfg=read(work/'config.json')
 state['health_journal_started_at']=last.get('health_journal_started_at',time.time())
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
  while True:
   t=time.time();cfg=read(work/'config.json');nodes={};fs={pool.submit(probe,work,n,c['host']):n for n,c in cfg['nodes'].items()}
   for f,n in fs.items():
    try:nodes[n]=f.result()
    except Exception as e:nodes[n]=dict(node=n,at=time.time(),observation_error=repr(e))
   q=read(work/'queues/caption/state.json');report=dict(at=time.time(),nodes=nodes,resolved_clips=q['completed_clips'],failed_sources=len(q['failed']))
   if cfg.get('visual_quality_backfill_enabled'):
    from listen_benchmark.j_visual_backfill import refill
    try:report['visual_quality_refill']=refill(cfg)
    except Exception as exc:report['visual_quality_refill']=dict(status='error',error=repr(exc))
   # Preserve observations and recovery actions between the half-hour reports.
   # latest.json alone overwrites this evidence on the next health poll.
   with (folder/'health.jsonl').open('a') as journal:
    journal.write(json.dumps(report,ensure_ascii=False,allow_nan=False)+'\n')
   write(folder/'latest.json',report)
   slot=min(20,max(0,int((t-start)//1800)))
   if slot not in reported or t>=end:
    write(folder/f'report-{slot:02d}.json',report);reported.add(slot)
   state.update(heartbeat=time.time(),reported_slots=sorted(reported),completed_reports=len(reported),last_resolved_clips=q['completed_clips'],stage='complete' if t>=end else 'monitoring')
   write(folder/'state.json',state)
   if t>=end:write(folder/'complete.json',dict(at=time.time(),start=start,end=end,report_slots=sorted(reported)));break
   time.sleep(min(30,max(0,end-time.time())))
 owner.close()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True);p.add_argument('--start',type=float,required=True);p.add_argument('--end',type=float,required=True);p.add_argument('--folder',type=Path);a=p.parse_args();main(a.work.resolve(),a.start,a.end,a.folder)
