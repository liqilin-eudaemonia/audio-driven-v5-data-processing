"""Nine-hour hourly audits and scoped recovery for the authorized GPU7 evals."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from p2_testsets_eval.protocol import read,save,sha,PYTHON
from listen_cluster import transport
import p2_all_lora_followup_queue_20261109 as Q

WORK=ROOT/'experiments/p2_ltx_all_lora_eval_watch9h_20261109'
UPSTREAM=Q.UPSTREAM
CHAIN=Q.WORK
STEPS=[5000,5500,7000,7500]
RUNS=[UPSTREAM]+[ROOT/f'experiments/p2_ltx_all_lora{s}_seamless200_sft2752_gpu7_20261109' for s in STEPS[1:]]
MODES={
 'upstream':dict(work=UPSTREAM,script=ROOT/'scripts/p2_all_lora5000_seamless_gpu7_20261109.py',args=['queue','--work',str(UPSTREAM)],lock='controller.lock',runs=[UPSTREAM]),
 'chain':dict(work=CHAIN,script=ROOT/'scripts/p2_all_lora_followup_queue_20261109.py',args=[],lock='queue.lock',runs=RUNS[1:])}

REMOTE=r'''
import json,os,signal,subprocess,sys,time
from pathlib import Path
request=json.loads(sys.argv[1]);works=request['works']
def own(pid):
 p=Path('/proc')/str(pid)
 try:
  cmd=p.joinpath('cmdline').read_bytes().decode().replace(chr(0),' ')
  env=p.joinpath('environ').read_bytes().split(bytes([0]));cuda=next((e.split(b'=',1)[1].decode()for e in env if e.startswith(b'CUDA_VISIBLE_DEVICES=')),None)
  allowed=any(s in cmd for s in ['p2_all_lora5000_seamless_gpu7_20261109.py','p2_all_lora_followups_gpu7_20261109.py','p2_pipelined_testsets_eval.py','p2_pipelined_video_metrics.py'])
  work=next((w for w in works if w in cmd),None)
  stat=p.joinpath('stat').read_text().rsplit(')',1)[1].split()
  return dict(pid=pid,owned=bool(allowed and work and cuda=='7'),work=work,cuda=cuda,start_ticks=stat[19],state=stat[0],cmd=cmd)
 except (FileNotFoundError,ProcessLookupError):return dict(pid=pid,owned=False,exited=True)
if request.get('kill'):
 expected=request['kill'];now=own(expected['pid'])
 if now.get('owned')and now.get('work')==expected['work']and now.get('start_ticks')==expected['start_ticks']:
  os.kill(now['pid'],signal.SIGTERM);print(json.dumps(dict(terminated=now['pid'],verified=now)))
 else:print(json.dumps(dict(terminated=None,reason='identity_changed_or_not_owned',current=now)))
 sys.exit(0)
def run(cmd):
 p=subprocess.run(cmd,capture_output=True,text=True,timeout=12)
 if p.returncode:raise RuntimeError(p.stderr[-500:])
 return p.stdout
raw=run(['nvidia-smi','-i','7','--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu,clocks.sm,power.draw','--format=csv,noheader,nounits'])
v=[x.strip()for x in raw.strip().split(',')]
gpu=dict(index=int(v[0]),uuid=v[1],used_mib=float(v[2]),total_mib=float(v[3]),util=float(v[4]),sm_mhz=float(v[5]),watts=float(v[6]))
apps=[]
for line in run(['nvidia-smi','-i','7','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader,nounits']).splitlines():
 fields=[x.strip()for x in line.split(',')]
 if len(fields)!=3 or fields[0]!=gpu['uuid']:continue
 identity=own(int(fields[1]));identity['used_mib']=fields[2];apps.append(identity)
print(json.dumps(dict(at=time.time(),gpu=gpu,apps=apps)))
'''


def lock_free(path):
 path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('a')as f:
  try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:return False
 return True


def process_identity(pid,script):
 p=Path('/proc')/str(pid)
 try:
  cmd=p.joinpath('cmdline').read_bytes().decode().replace(chr(0),' ')
  state=p.joinpath('stat').read_text().rsplit(')',1)[1].split()[0]
  return dict(pid=pid,live=state!='Z' and str(script) in cmd,cmd=cmd,state=state)
 except (FileNotFoundError,ProcessLookupError):return dict(pid=pid,live=False)


def controller(mode):
 info=MODES[mode];f=info['work']/'CONTROLLER.json'
 return process_identity(read(f)['pid'],info['script']) if f.exists()else dict(pid=None,live=False)


def remote(request=None):
 cfg=read(UPSTREAM/'cluster.json');node=cfg['nodes'][0]
 payload=dict(works=[str(p) for p in RUNS]);payload.update(request or {})
 p=subprocess.run(transport(cfg,node,['/private/software/conda/bin/python','-c',REMOTE,json.dumps(payload)]),capture_output=True,text=True,timeout=25)
 if p.returncode:raise ConnectionError(p.stderr[-1000:])
 return json.loads(p.stdout)


def done(work,step):
 try:return Q.completed(work,step)
 except Exception as exc:return dict(error=repr(exc))


def run_state(work,step,detailed=False):
 d=dict(step=step,work=str(work),complete=done(work,step),stopped=(work/'STOP').exists())
 d['status']=read(work/'status.json')if(work/'status.json').exists()else {}
 d['failure']=read(work/'FAILURE.json')if(work/'FAILURE.json').exists()else None
 if detailed:
  rows=[]
  for f in (work/'latents').glob('*/*/*/metadata.json'):
   m=read(f);rows.append(dict(at=f.stat().st_mtime,seconds=m['seconds'],frames=m['geometry']['native_frames']))
  rows.sort(key=lambda x:x['at']);d['generated']=len(rows)
  media=list((work/'generated').glob('*/*/*/complete.json'));d['decoded']=len(media)
  if rows:
   d['last_generation_at']=rows[-1]['at'];d['recent_seconds_per_frame']=[x['seconds']/x['frames']for x in rows[-20:]]
   d['mean_recent_seconds']=statistics.mean(x['seconds']for x in rows[-10:])
  if step==5000 and len(rows)>=22:d['baseline_seconds_per_frame']=statistics.median(x['seconds']/x['frames']for x in rows[2:22])
  d['latest_media_at']=max([f.stat().st_mtime for f in media],default=None)
 return d


def next_work(states):
 return next((d for d in states if d['complete'] is not True),None)


def slowdown(samples,baseline):
 if len(samples)<20 or not baseline:return False
 return all(statistics.median(part)>baseline*1.5 for part in (samples[:10],samples[-10:]))


def audit(detailed=True):
 states=[run_state(w,s,detailed)for w,s in zip(RUNS,STEPS)]
 return dict(at=time.time(),runs=states,active=next_work(states),controllers={m:controller(m)for m in MODES},
             chain_status=read(CHAIN/'status.json')if(CHAIN/'status.json').exists()else {})


def archive_failure(work,destination):
 f=work/'FAILURE.json'
 if f.exists():
  destination.mkdir(parents=True,exist_ok=True);f.replace(destination/'FAILURE.json')


def recover(mode,reason,limit_end,state):
 info=MODES[mode];work=info['work'];now=time.time()
 if now>=limit_end or (WORK/'STOP').exists()or(work/'STOP').exists():return dict(skipped='deadline_or_user_stop')
 if any((p/'STOP').exists()for p in info['runs']):return dict(skipped='explicit_stage_stop')
 previous=state.get('last_repair',{}).get(mode,0)
 if now-previous<900:return dict(skipped='15minute_recovery_cooldown')
 diagnostic=WORK/'repairs'/str(time.time_ns());diagnostic.mkdir(parents=True)
 # Observation failure is never proof that a process died.
 before=remote();save(diagnostic/'remote_before.json',before);ident=controller(mode)
 if ident['live']:
  if reason not in ('confirmed_stall','confirmed_slowdown','missed_transition'):return dict(skipped='controller_still_live')
  verified=process_identity(ident['pid'],info['script'])
  if not verified['live']:return dict(skipped='controller_changed_before_signal')
  save(diagnostic/'controller_terminated.json',verified);os.kill(ident['pid'],signal.SIGTERM)
 # Closing our controllers' stdin ownership stops their guarded workers. Recheck
 # cleanup, and SIGTERM only a still-identical orphan on CUDA_VISIBLE_DEVICES=7.
 for attempt in range(8):
  view=remote();owned=[a for a in view['apps']if a.get('owned')and a.get('work')in [str(x)for x in info['runs']]]
  if not owned and lock_free(work/info['lock']):break
  if attempt>=3:
   for a in owned:save(diagnostic/f'orphan_{a["pid"]}_{attempt}.json',remote(dict(kill=a)))
  time.sleep(5)
 else:return dict(skipped='owned_worker_cleanup_unconfirmed',diagnostic=str(diagnostic))
 if controller(mode)['live']or not lock_free(work/info['lock']):return dict(skipped='another_controller_owns_queue')
 if (work/'STOP').exists()or(WORK/'STOP').exists():return dict(skipped='user_stop_during_cleanup')
 archive_failure(work,diagnostic/'controller')
 for p in info['runs']:archive_failure(p,diagnostic/p.name)
 cmd=['env','PYTHONPATH=','CUDA_VISIBLE_DEVICES=','OMP_NUM_THREADS=2','OPENBLAS_NUM_THREADS=1',str(PYTHON),'-u',str(info['script']),*info['args']]
 with (work/('monitor_recovery_'+str(time.time_ns())+'.log')).open('ab',buffering=0)as log:
  process=subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 action=dict(at=time.time(),mode=mode,reason=reason,pid=process.pid,diagnostic=str(diagnostic),completed_outputs_preserved=True,physical_gpus=[7])
 save(diagnostic/'RECOVERY.json',action);state.setdefault('last_repair',{})[mode]=time.time()
 return action


def confirm_stall(active):
 # Three distinct current observations, not three reads of one stale file.
 work=Path(active['work']);counts=None;samples=[]
 for i in range(3):
  view=remote();d=read(work/'status.json')if(work/'status.json').exists()else {}
  current=tuple(d.get(k,0)for k in ('latents','decoded','scored_shards'))
  if counts is not None and current!=counts:return False,samples
  counts=current;samples.append(dict(at=time.time(),counts=current,remote=view))
  if i<2:time.sleep(10)
 # Resource saturation alone is not a hang. This function is called only after
 # >=15min with unchanged completion counters, exceeding any normal clip cost.
 return True,samples


def health(state,config,full=False):
 result=audit(full);active=result['active'];actions=[]
 if active is None:
  if full:result['resource']=remote()
  return dict(**result,actions=actions,all_evaluations_complete=True)
 mode='upstream'if active['step']==5000 else'chain'
 now=time.time();key=str(active['step']);c=active['status'];counts=tuple(c.get(k,0)for k in ('latents','decoded','scored_shards'))
 old=state.setdefault('progress',{}).get(key)
 controller_pid=result['controllers'][mode]['pid']
 if not old or old['counts']!=list(counts) or old.get('controller_pid')!=controller_pid:
  state['progress'][key]=dict(counts=list(counts),changed_at=now,controller_pid=controller_pid)
 last=state['progress'][key]['changed_at'];result['unchanged_seconds']=now-last
 if active['stopped']or(CHAIN/'STOP').exists():return dict(**result,actions=actions,explicit_stop_respected=True)
 if not result['controllers'][mode]['live'] and lock_free(MODES[mode]['work']/MODES[mode]['lock']):
  actions.append(recover(mode,'controller_exited',config['end_at'],state))
 elif now-last>(1800 if counts[0]==200 and counts[1]==200 else 900):
  okay,proof=confirm_stall(active);result['stall_confirmation']=proof
  if okay:
   last_resource=proof[-1]['remote'];foreign=[a for a in last_resource['apps']if not a.get('owned')]
   owned=[a for a in last_resource['apps']if a.get('owned')]
   free=last_resource['gpu']['total_mib']-last_resource['gpu']['used_mib']
   if foreign and not owned and free<80*1024:
    result['resource_contention']='GPU7 occupied by unrelated task; not terminated'
   elif mode=='chain'and result['chain_status'].get('stage')=='cpu_preparation':
    result['cpu_preparation_in_progress']=True
   else:actions.append(recover(mode,'confirmed_stall',config['end_at'],state))
 if full:
  try:result['resource']=remote()
  except Exception as exc:result['observation_error']=repr(exc);return dict(**result,actions=actions)
  base=result['runs'][0].get('baseline_seconds_per_frame')or state.get('baseline_seconds_per_frame')
  if base:state['baseline_seconds_per_frame']=base
  recent=active.get('recent_seconds_per_frame',[])
  result['baseline_seconds_per_frame']=base;result['sustained_slowdown']=slowdown(recent,base)
  if result['sustained_slowdown']and not actions:
   # One bounded reset per stage, only when no foreign GPU7 task is competing;
   # ongoing completion is preserved. Never repeatedly reset an inherently
   # slower/shared device or kill an unrelated application.
   foreign=[a for a in result['resource']['apps']if not a.get('owned')]
   result['foreign_gpu7_tasks']=foreign
   if not foreign and key not in state.setdefault('slowdown_reset_steps',[]):
    actions.append(recover(mode,'confirmed_slowdown',config['end_at'],state));state['slowdown_reset_steps'].append(key)
 return dict(**result,actions=actions)


def main():
 WORK.mkdir(parents=True,exist_ok=True)
 with(WORK/'watcher.lock').open('a')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);config=read(WORK/'CONFIG.json')
  assert config['end_at']-config['start_at']==9*3600 and config['physical_gpus']==[7]
  for p,digest in config['code'].items():assert sha(p)==digest
  save(WORK/'CONTROLLER.json',dict(pid=os.getpid(),at=time.time(),end_at=config['end_at']))
  state=read(WORK/'STATE.json')if(WORK/'STATE.json').exists()else {}
  while not(WORK/'STOP').exists():
   now=time.time();due=[i for i in range(10)if now>=config['start_at']+3600*i and not(WORK/f'hour_{i:02d}.json').exists()]
   try:
    report=health(state,config,bool(due))
    save(WORK/'LATEST.json',report);save(WORK/'STATE.json',state)
    if 'observation_error'not in report:
     for i in due:
      report=dict(report,hour=i,scheduled_at=config['start_at']+3600*i,sample_delay_seconds=report['at']-(config['start_at']+3600*i))
      save(WORK/f'hour_{i:02d}.json',report)
    with(WORK/'health.jsonl').open('a')as f:f.write(json.dumps(report)+'\n')
   except Exception as exc:
    report=dict(at=time.time(),observation_error=repr(exc),traceback=traceback.format_exc(),no_restart_from_observation_failure=True)
    save(WORK/'OBSERVATION_ERROR.json',report)
   if time.time()>=config['end_at']:
    reports=[i for i in range(10)if(WORK/f'hour_{i:02d}.json').exists()]
    save(WORK/'WINDOW_COMPLETE.json',dict(success=len(reports)==10,at=time.time(),start_at=config['start_at'],end_at=config['end_at'],hourly_reports=len(reports),missing=[i for i in range(10)if i not in reports],evaluation_tasks_left_running=True))
    return
   time.sleep(min(60,max(1,config['end_at']-time.time())))
  save(WORK/'STOPPED.json',dict(at=time.time(),evaluation_tasks_not_stopped=True))

if __name__=='__main__':main()
