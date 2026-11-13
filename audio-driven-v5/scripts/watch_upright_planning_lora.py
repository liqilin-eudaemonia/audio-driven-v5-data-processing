#!/usr/bin/env python3
"""User-authorized first 40-minute check, then hourly checks until LoRA finishes.

Local PID/birth/command proof and continuing progress govern recovery. A query
failure or an idle GPU sample alone never authorizes terminating live work.
"""
import argparse
import collections
import datetime
import fcntl
import hashlib
import json
import math
import os
import re
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time
import traceback

ROOT=Path('/private/liqi_workspace/project_code/audio-driven-v5')
WORK=ROOT/'experiments/p2_planning_sft_upright_3b_gpu7_20261105'
FILTER=ROOT/'experiments/listen_j_upright_response_sft_20261105'
MON=WORK/'monitor'
UUID='GPU-d5cb2cac-0f7a-e5f8-1870-c2efcae7aedc'
PY=str(ROOT.parent/'Omni-LTX-stage2/.venv/bin/python')
FPY='/private/software/conda/envs/vllm/bin/python'
CHECKPOINT_FILES=('adapter_model.safetensors','adapter_config.json','optimizer.pt','scheduler.pt','trainer_state.json','rng_state.pth')

def read(p,default=None):
 try:return json.loads(Path(p).read_bytes())
 except FileNotFoundError:
  if default is not None:return default
  raise

def save(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+f'.{os.getpid()}.tmp');t.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n');t.replace(p)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def ensure(ok,msg):
 if not ok:raise ValueError(msg)
def event(kind,**kw):
 r=dict(event=kind,at=time.time(),**kw)
 with (MON/'events.jsonl').open('a') as f:f.write(json.dumps(r,ensure_ascii=False)+'\n')
 print(json.dumps(r,ensure_ascii=False),flush=True)

def process(pid):
 try:
  p=Path('/proc',str(pid));s=(p/'stat').read_text().rsplit(')',1)[1].split();args=[a.decode() for a in (p/'cmdline').read_bytes().split(b'\0') if a]
  return dict(pid=int(pid),state=s[0],ppid=int(s[1]),pgid=int(s[2]),start_ticks=s[19],args=args,alive=s[0] not in ('Z','X'),wchan=(p/'wchan').read_text().strip())
 except (FileNotFoundError,ProcessLookupError):return None

def owned(record):
 if not record or not record.get('pid'):return None
 p=process(record['pid'])
 if not p or not p['alive']:return None
 if record.get('start_ticks') is not None and str(record['start_ticks'])!=p['start_ticks']:return None
 cmd=record.get('command',record.get('cmd'))
 if isinstance(cmd,list) and p['args']!=cmd:return None
 # Every managed command must be bound to one of these two exact runs.
 if not any(str(WORK) in a or str(FILTER) in a or a==str(ROOT/'scripts/run_j_upright_response_sft.py') for a in p['args']):return None
 return p

def gpu():
 try:
  fields=[x.strip() for x in subprocess.check_output(['nvidia-smi','-i','7','--query-gpu=uuid,memory.used,utilization.gpu,power.draw','--format=csv,noheader,nounits'],text=True,timeout=12).split(',')]
  ensure(fields[0]==UUID,'Unexpected physical node/GPU')
  return dict(ok=True,uuid=fields[0],memory_mib=int(fields[1]),utilization=int(fields[2]),power_w=float(fields[3]),at=time.time())
 except (OSError,ValueError,subprocess.SubprocessError) as exc:return dict(ok=False,error=repr(exc),at=time.time())

def metrics():
 p=WORK/'training/step_metrics.jsonl'
 if not p.exists():return []
 with p.open('rb') as f:
  off=max(0,p.stat().st_size-32000);f.seek(off)
  if off:f.readline()
  rows=f.readlines()
 result=[]
 for raw in rows:
  try:result.append(json.loads(raw))
  except json.JSONDecodeError:continue
 return result

def losses():
 p=WORK/'logs/training.log'
 if not p.exists():return dict(observed=False,finite=True,latest=[])
 with p.open('rb') as f:f.seek(max(0,p.stat().st_size-524288));text=f.read().decode(errors='replace')
 vals=[]
 for match in re.finditer(r"['\"](?:loss|grad_norm)['\"]\s*:\s*([^,}]+)",text):
  try:vals.append(float(match.group(1)))
  except ValueError:continue
 return dict(observed=bool(vals),finite=all(math.isfinite(v) for v in vals),latest=vals[-8:])

def snapshot():
 queue_record=read(WORK/'QUEUE_LAUNCH.json',{});train_record=read(WORK/'training_process.json',{});smoke_record=read(WORK/'GPU_smoke_process.json',{})
 fil_record=read(FILTER/'LAUNCH.json',{});fs=read(FILTER/'RUN_STATUS.json',{});part='voiced' if 'voiced' in fs.get('stage','') else 'strict';fp=read(FILTER/part/'process.json',{})
 rows=metrics();last=rows[-1] if rows else {};now=time.time();qs=read(WORK/'QUEUE_STATUS.json',{})
 return dict(at=now,hostname=socket.gethostname(),gpu=gpu(),loss_health=losses(),queue=qs,queue_record=queue_record,queue_process=owned(queue_record),training_record=train_record,training_process=owned(train_record),smoke_record=smoke_record,smoke_process=owned(smoke_record),filter_record=fil_record,filter_process=owned(fil_record),filter_worker_record=fp,filter_worker=owned(fp),filter_status=fs,filter_complete=(FILTER/'COMPLETE.json').exists(),filter_error=read(FILTER/'CONTROLLER_ERROR.json',{}),queue_error=read(WORK/'QUEUE_ERROR.json',{}),training_complete=(WORK/'COMPLETE.json').exists(),step=last.get('step',0),last_metrics=last,recent_step_count=len(rows),recent_mean_seconds=sum(r['step_seconds'] for r in rows[-100:])/len(rows[-100:]) if rows else None,recent_mean_wall_seconds=sum(r['wall_since_previous_step_s'] for r in rows[-100:])/len(rows[-100:]) if rows else None,metrics_age_s=now-last['time'] if last else None,free_disk_bytes=shutil.disk_usage(WORK).free)

def frozen_ok():
 p=read(WORK/'PREPARED.json');ensure(sha(WORK/'config.json')==p['config_sha256'],'Frozen training config changed')
 for path,h in p['code'].items():ensure(sha(path)==h,'Frozen training code changed: '+path)
 return True

def latest_checkpoint(cache=None):
 cache={} if cache is None else cache;rejected=[];out=WORK/'training'
 for p in sorted(out.glob('checkpoint-*'),key=lambda p:int(p.name.split('-')[-1]),reverse=True):
  try:
   for name in CHECKPOINT_FILES:ensure((p/name).is_file() and (p/name).stat().st_size>0,'Missing/empty '+name)
   state=read(p/'trainer_state.json');step=int(p.name.split('-')[-1]);ensure(state['global_step']==step,'Checkpoint step differs')
   ensure(all(math.isfinite(float(r[k])) for r in state.get('log_history',[]) for k in ('loss','grad_norm') if k in r),'Checkpoint has nonfinite loss/gradient')
   config=read(p/'adapter_config.json');ensure(config['r']==32 and config['lora_alpha']==64,'Wrong LoRA checkpoint')
   files={}
   for name in CHECKPOINT_FILES:
    st=(p/name).stat();key=str(p/name)+':'+str(st.st_size)+':'+str(st.st_mtime_ns)
    if key not in cache:
     value=sha(p/name);after=(p/name).stat();ensure((st.st_size,st.st_mtime_ns)==(after.st_size,after.st_mtime_ns),'Checkpoint is still changing');cache[key]=value
    files[name]=dict(bytes=st.st_size,mtime_ns=st.st_mtime_ns,sha256=cache[key])
   return dict(valid=True,path=str(p),step=step,files=files,rejected_newer=rejected)
  except (OSError,ValueError,KeyError) as exc:rejected.append(dict(path=str(p),error=repr(exc)))
 return dict(valid=False,rejected_newer=rejected)

def env(gpu_mask=''):
 return dict(os.environ,CUDA_VISIBLE_DEVICES=gpu_mask,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',PYTHONPATH=str(WORK/'vendor')+':'+str(ROOT/'src')+':'+str(ROOT/'scripts'),PYTHONUNBUFFERED='1')

def spawn(command,log,record,child_env=None):
 with Path(log).open('a') as f:p=subprocess.Popen(command,cwd=ROOT,env=child_env or env(),stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 info=process(p.pid);ensure(info is not None,'Spawned process vanished')
 rec=dict(pid=p.pid,start_ticks=info['start_ticks'],command=command,started_at=time.time(),node='node8',host='172.29.104.237',gpu=7)
 save(record,rec);return rec

def archive(p,folder):
 p=Path(p)
 if p.exists():folder.mkdir(parents=True,exist_ok=True);p.rename(folder/p.name)

def other_live_inputs():
 # Adopted native descendants can outlive a terminated controller. Never
 # duplicate a live child merely because its parent's status stopped updating.
 result=[];needles={str(WORK/'vendor/train_entry.py'),str(WORK/'vendor/queue_driver.py')}
 for p in Path('/proc').iterdir():
  if not p.name.isdigit():continue
  try:args=set(x.decode() for x in (p/'cmdline').read_bytes().split(b'\0') if x)
  except (FileNotFoundError,ProcessLookupError,PermissionError):continue
  if not needles.intersection(args):continue
  info=process(int(p.name))
  if info and info['alive']:result.append(info)
 return result

def stop_verified_group(record):
 p=owned(record);ensure(p is not None and p['pgid']==p['pid'],'Owned process group identity not proven')
 old=record.copy();os.killpg(p['pgid'],signal.SIGTERM)
 for _ in range(10):
  if owned(old) is None:return
  time.sleep(1)
 p=owned(old)
 if p and p['pgid']==p['pid']:os.killpg(p['pgid'],signal.SIGKILL)

def repair(s,history,cache,attempts):
 if (MON/'STOP').exists() or s['training_complete']:return None
 g=s['gpu'];now=s['at']
 if not g['ok']:return dict(action='defer_observation_unavailable')
 if attempts and now-attempts[-1]<60:return None
 if len([t for t in attempts if now-t<7200])>=3:return dict(action='needs_attention_recovery_limit')
 # Restart only a proven-dead filter parent with no surviving filter worker.
 if not s['filter_complete']:
  if s['filter_process'] or s['filter_worker']:return None
  if g['memory_mib']>1024:return dict(action='defer_gpu_not_released')
  folder=MON/'recoveries'/str(time.time_ns());folder.mkdir(parents=True);save(folder/'before.json',s)
  archive(FILTER/'CONTROLLER_ERROR.json',folder);cmd=[FPY,str(FILTER/'vendor/controller.py'),'control']
  rec=spawn(cmd,folder/'filter_restart.log',FILTER/'LAUNCH.json',dict(env(),LD_PRELOAD='/private/software/conda/envs/vllm/lib/libstdc++.so.6'));attempts.append(now);return dict(action='verified_dead_filter_restarted',process=rec,at=now)
 train=s['training_process'];smoke=s['smoke_process']
 if train:
  # Two minutes of observed zero activity AND ten minutes of no committed
  # steps, after startup, are required; in-flight checkpoint writes exempt it.
  recent=[r for r in history if now-r['at']<165 and r['gpu']['ok']]
  zeros=len(recent)>=5 and now-recent[0]['at']>=120 and all(r['gpu']['utilization']==0 for r in recent)
  checkpoint_writing=any(now-p.stat().st_mtime<180 for p in (WORK/'training').glob('checkpoint-*/*') if p.is_file())
  age=s['metrics_age_s'] if s['metrics_age_s'] is not None else now-s['training_record'].get('started_at',now)
  if age>600 and zeros and not checkpoint_writing:
   folder=MON/'recoveries'/str(time.time_ns());folder.mkdir(parents=True);save(folder/'confirmed_stall.json',s);stop_verified_group(s['training_record']);return dict(action='confirmed_owned_training_stall_stopped_for_checkpoint_resume',at=now,evidence=str(folder))
  return None
 if smoke:return None
 if s['queue_process']:return None
 # A missing parent can leave a working child. No restart while any exact
 # native queue/train command is still live, even if the GPU sample is idle.
 live=other_live_inputs()
 if live:return dict(action='defer_live_owned_child',pids=[p['pid'] for p in live])
 if g['memory_mib']>1024:return dict(action='defer_gpu_not_released')
 frozen_ok();folder=MON/'recoveries'/str(time.time_ns());folder.mkdir(parents=True);save(folder/'before.json',s)
 ckpt=latest_checkpoint(cache)
 if (WORK/'training/run.json').exists() and not (WORK/'training/COMPLETE.json').exists() and not ckpt['valid']:
  # Preserve every unsaved artifact. Native fresh-run replay is allowed only
  # when there is no complete checkpoint to recover and no owned process lives.
  archive(WORK/'training',folder);save(folder/'fresh_replay.json',dict(reason='No complete checkpoint exists; preserve interrupted run and replay same seed/data/config',lost_unsaved_steps=s['step']))
 if (WORK/'smoke/run.json').exists() and not (WORK/'SMOKE_PASSED.json').exists():archive(WORK/'smoke',folder)
 archive(WORK/'QUEUE_ERROR.json',folder)
 rec=spawn([PY,str(WORK/'vendor/queue_driver.py'),'control'],folder/'queue_restart.log',WORK/'QUEUE_LAUNCH.json');attempts.append(now)
 return dict(action='verified_dead_queue_restarted',process=rec,resume_checkpoint=ckpt.get('path'),at=now)

def full_check(s,number,due,previous,cache):
 report=dict(number=number,due_at=due,at=time.time(),snapshot=s,code_config_intact=frozen_ok(),checkpoint=latest_checkpoint(cache),net_steps_since_previous=s['step']-previous.get('step',0),previous_report_at=previous.get('at'))
 ga=read(WORK/'training/gradient_audit.json',{});report['gradient_audit']=ga
 if ga:ensure(ga['finite'] and ga['norm']>0 and not ga['frozen_gradient'],'Gradient audit failed')
 ensure(s['loss_health']['finite'],'Recent loss or gradient is nonfinite')
 if (WORK/'DATA_READY.json').exists():
  data=read(WORK/'DATA_READY.json');report['expected_steps']=data['total_optimizer_steps'];mean=s.get('recent_mean_wall_seconds') or s['recent_mean_seconds']
  if mean:report['rough_remaining_hours']=max(0,data['total_optimizer_steps']-s['step'])*mean/3600
 report['training_smoothly_running']=bool(s['training_process'] and s['step']>=2 and s['metrics_age_s'] is not None and s['metrics_age_s']<300 and s['gpu']['ok'] and ga and s['loss_health']['observed'] and s['loss_health']['finite'])
 save(MON/'reports'/f'check-{number:04d}.json',report);save(MON/'latest_full_check.json',report)
 event('scheduled_full_check',number=number,due_at=due,step=s['step'],training_smoothly_running=report['training_smoothly_running'],checkpoint_step=report['checkpoint'].get('step'))
 return report

def final_audit(s,cache):
 ensure((WORK/'COMPLETE.json').exists(),'Queue final marker missing');done=read(WORK/'COMPLETE.json');train=WORK/'training';td=read(train/'COMPLETE.json');data=read(WORK/'DATA_READY.json')
 ensure(done['status']=='complete' and td['success'] and not td['smoke'] and td['validation_enabled'] is False,'Formal training did not finish')
 ensure(td['global_step']==data['total_optimizer_steps'],'Not a full training epoch')
 ensure(sha(train/'COMPLETE.json')==done['training_complete_sha256'] and sha(WORK/'data_seal.json')==done['data_seal_sha256'],'Final receipt SHA mismatch')
 ensure((train/'final_adapter/adapter_model.safetensors').is_file(),'Final adapter missing');frozen_ok();ck=latest_checkpoint(cache);ensure(ck['valid'],'No complete restartable checkpoint')
 ensure(not s['training_process'] and not s['queue_process'],'Final writers are still active')
 result=dict(passed=True,global_step=td['global_step'],expected_steps=data['total_optimizer_steps'],training_complete_sha256=sha(train/'COMPLETE.json'),final_adapter_sha256=sha(train/'final_adapter/adapter_model.safetensors'),checkpoint=ck,reports=len(list((MON/'reports').glob('check-*.json'))),at=time.time())
 save(MON/'COMPLETE.json',result);event('monitor_complete',global_step=td['global_step']);return result

def supervise():
 MON.mkdir(exist_ok=True);(MON/'reports').mkdir(exist_ok=True)
 with (MON/'monitor.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  info=process(os.getpid());save(MON/'process.json',dict(pid=os.getpid(),start_ticks=info['start_ticks'],command=[sys.executable,str(Path(__file__).resolve()),'supervise'],started_at=time.time(),host=socket.gethostname()))
  cfg=read(MON/'schedule.json');state=read(MON/'state.json',{});number=state.get('full_checks',0);due=state.get('next_full_check_at',cfg['first_check_at']);cache=state.get('checkpoint_hash_cache',{});history=[];attempts=state.get('recovery_attempts',[]);last_full=read(MON/'latest_full_check.json',{});previous=dict(step=last_full.get('snapshot',{}).get('step',0),at=last_full.get('at'));last_action=None
  while not (MON/'STOP').exists():
   try:
    s=snapshot();history.append(s);history=[r for r in history if s['at']-r['at']<180]
    action=repair(s,history,cache,attempts)
    if action and (action.get('action')!=last_action or action.get('process')):event('recovery_action',**action)
    last_action=action.get('action') if action else None
    if s['at']>=due:
     number+=1;report=full_check(s,number,due,previous,cache);previous=dict(step=s['step'],at=s['at'])
     due=due+3600 if report['training_smoothly_running'] else s['at']+300
    if s['training_complete'] and not s['training_process'] and not s['queue_process']:
     final_audit(s,cache);save(MON/'state.json',dict(stage='complete',at=time.time(),full_checks=number));return
    save(MON/'state.json',dict(stage='monitoring',at=time.time(),full_checks=number,next_full_check_at=due,first_check_at=cfg['first_check_at'],recovery_attempts=attempts,checkpoint_hash_cache=cache,latest_snapshot=s))
   except Exception as exc:
    error=dict(error=repr(exc),traceback=traceback.format_exc(),at=time.time());save(MON/'LAST_ERROR.json',error);event('inspection_error',error=repr(exc))
   time.sleep(30)
  save(MON/'state.json',dict(stage='stopped_by_explicit_STOP_file',at=time.time(),full_checks=number))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['snapshot','supervise']);a=p.parse_args()
 if a.action=='snapshot':print(json.dumps(snapshot(),ensure_ascii=False))
 else:supervise()
