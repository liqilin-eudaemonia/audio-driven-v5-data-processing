"""Authorized four-stage6000 sequence, isolated from frozen training sources."""
from pathlib import Path
import concurrent.futures, fcntl, json, math, os, statistics, subprocess, sys, time, traceback
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
PARENT=ROOT/'experiments/p2_current500h_av_whisper31_20261107'
ARMS=('av_event','no_whisper','no_feature','no_event')
TARGET=6000
from p2_listen.contracts import read,sha,atomic_json,digest

def alive(record):
 try:
  parts=Path('/proc',str(record['pid']),'stat').read_text().split()
  return parts[2]!='Z' and parts[21]==str(record['start_ticks'])
 except FileNotFoundError:return False

def emit(stage,**extra):
 value=dict(stage=stage,at=time.time(),controller_pid=os.getpid(),**extra)
 atomic_json(HERE/'SEQUENCE_STATE.json',value)
 with (HERE/'sequence.events.jsonl').open('a')as f:f.write(json.dumps(value)+'\n')
 print(json.dumps(value),flush=True)

def cancelled():
 if (HERE/'QUEUE_STOP').exists():raise RuntimeError('Explicit queue stop preserved')

def boundary(step):
 if step>TARGET:raise RuntimeError('Training advanced beyond6000; preserve weights and block handoff')
 return step==TARGET

def arm_command(arm,logs):
 return [sys.executable,str(ROOT/'scripts'/('p2_current500h_av_event_only.py' if arm=='av_event' else 'p2_current500h_component_ablations.py')),
         'launch','--config',str(HERE/arm/'config.json'),'--stop-after',str(TARGET),'--logs',str(logs)]

def validate_ready():
 from p2_whisper_av.bootstrap import activate_ltx
 activate_ltx()
 from p2_current_component_ablation.config import load,source_identity
 from p2_current_component_ablation.data import data_identity
 from p2_current_component_ablation.training import check_preflight
 accepted=read(HERE/'REVIEW_ACCEPTED.json')
 if not accepted['success'] or sha(HERE/'SOURCE_IDENTITY.json')!=accepted['source_identity_sha256']:raise ValueError('Review identity changed')
 source=source_identity()
 if source!=read(HERE/'SOURCE_IDENTITY.json'):raise ValueError('Frozen training source changed')
 for arm in ARMS[1:]:
  folder=HERE/arm;c=load(folder/'config.json');receipt=accepted['arms'][arm]
  if sha(folder/'config.json')!=receipt['config_sha256']or sha(folder/'preflight.json')!=receipt['preflight_sha256']:raise ValueError('Arm acceptance changed')
  check_preflight(c,read(folder/'preflight.json'),source,data_identity(c))
  if (folder/'STOP').exists():raise RuntimeError('Explicit arm STOP preserved: '+arm)


 from p2_current_av_event_only.config import load as event_load,source_identity as event_source
 from p2_current_av_event_only.data import data_identity as event_data
 from p2_current_av_event_only.training import check_preflight as event_check
 folder=HERE/'av_event';c=event_load(folder/'config.json');receipt=read(folder/'ACCEPTED.json')
 if not receipt['success'] or receipt['source']!=event_source() or receipt['config_sha256']!=sha(folder/'config.json') or receipt['preflight_sha256']!=sha(folder/'preflight.json'):raise ValueError('AV+Event acceptance changed')
 event_check(c,read(folder/'preflight.json'),event_source(),event_data(c))
 if (folder/'STOP').exists():raise RuntimeError('Explicit AV+Event STOP preserved')

def verified_state(value,identity):
 if value['step']!=TARGET or value['identity']!=identity or identity['world']!=31 or len(value['rank_rng'])!=31:
  raise ValueError('6000 full checkpoint/run/world/RNG mismatch')
 steps={int(s['step'])for s in value['optimizer']['state'].values()if 'step'in s}
 if steps!={TARGET}:raise ValueError('6000 Adam state missing or stale')

def verify_checkpoint(folder,kind):
 # Separate CPU process releases the complete model/Adam mapping before launch.
 command=[sys.executable,str(Path(__file__).resolve()),'verify',str(folder),kind]
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
 result=subprocess.run(command,env=env,capture_output=True,text=True,timeout=1800)
 (folder/'SEQUENCE_CHECKPOINT_VERIFY.log').write_text(result.stdout+result.stderr)
 if result.returncode:raise RuntimeError('Full checkpoint verification failed: '+str(folder))
 return read(folder/'SEQUENCE_CHECKPOINT6000_VERIFIED.json')

def verify_main(folder,kind):
 import torch
 torch.set_num_threads(2)
 if kind=='parent':from p2_current_av_whisper.checkpoint import checked
 elif kind=='av_event':from p2_current_av_event_only.checkpoint import checked
 else:from p2_current_component_ablation.checkpoint import checked
 from p2_listen.training import rng_state,restore_rng
 p=folder/'step_006000.pt';v=checked(p);identity=read(folder/'run.json');verified_state(v,identity)
 original=rng_state()
 for r in v['rank_rng']:restore_rng(r)
 restore_rng(original)
 receipt=dict(success=True,at=time.time(),step=TARGET,path=str(p),sha256=read(p.with_suffix('.complete.json'))['sha256'],
              bytes=p.stat().st_size,world=31,full_model_adam_rng=True,all_rank_rng_restorable=True,cuda_initialized=torch.cuda.is_initialized())
 if receipt['cuda_initialized']:raise RuntimeError('Checkpoint verifier must be CPU only')
 atomic_json(folder/'SEQUENCE_CHECKPOINT6000_VERIFIED.json',receipt)

def workers_exited(folder):
 from listen_cluster import transport
 cluster=read(folder/'cluster.json')
 def one(node):
  try:
   workers=read(folder/'rank_logs'/(node['name']+'_workers.json'))
   if workers['gpus']!=node['gpus']or len(workers['pids'])!=len(node['gpus']):raise ValueError('Unexpected rank ownership')
   code="""import json,sys
from pathlib import Path
pids=json.loads(sys.argv[1]);active=[]
for pid in pids:
 try:
  s=Path('/proc',str(pid),'stat').read_text().split()
  cmd=Path('/proc',str(pid),'cmdline').read_bytes().decode().replace(chr(0),' ')
  if s[2]!='Z' and sys.argv[2] in cmd:active.append(pid)
 except FileNotFoundError:pass
print(json.dumps(dict(active=active)))
"""
   r=subprocess.run(transport(cluster,node,[cluster['python'],'-c',code,json.dumps(workers['pids']),str(folder/'config.json')]),capture_output=True,text=True,timeout=30)
   if r.returncode:raise RuntimeError(r.stderr[-500:])
   value=json.loads(r.stdout);return dict(node=node['name'],exited=not value['active'],**value)
  except Exception as e:return dict(node=node['name'],exited=False,error=repr(e))
 with concurrent.futures.ThreadPoolExecutor(4)as pool:reports=list(pool.map(one,cluster['nodes']))
 result=dict(at=time.time(),all31_exited=all(r['exited']for r in reports),nodes=reports)
 atomic_json(folder/'SEQUENCE_WORKERS_EXIT.json',result)
 return result['all31_exited']

def await_exit(folder,controller=None):
 while True:
  cancelled()
  if read(folder/'status.json')['step']!=TARGET:raise ValueError('Expected exact6000 before handoff')
  if (controller is None or not alive(controller))and workers_exited(folder):return
  emit('waiting_previous_processes_exit',work=str(folder));time.sleep(5)

def wait_idle(folder):
 import p2_current500h_component_ablations as cli
 consecutive=0
 while consecutive<2:
  cancelled();snapshot=cli.launcher().selected_resources(read(folder/'cluster.json'))
  atomic_json(HERE/'SEQUENCE_RESOURCES.json',snapshot)
  consecutive=consecutive+1 if snapshot['selected_all_idle']else 0
  emit('waiting_selected31_idle',next_arm=folder.name,consecutive_idle=consecutive)
  if consecutive<2:time.sleep(10)

def metrics(folder):
 path=folder/'metrics.jsonl'
 if not path.exists():return None
 with path.open('rb')as f:f.seek(max(0,path.stat().st_size-150000));raw=f.read().splitlines()
 rows=[]
 for line in raw:
  try:rows.append(json.loads(line))
  except ValueError:pass
 if not rows:return None
 row=rows[-1];weights=read(folder/'config.json')['loss_weights']
 expected=row['av']+weights['event']*row['event']+sum(weights[m+'_feature']*row['feature_'+m]for m in ('audio','video'))
 healthy=all(math.isfinite(row[k])for k in ('av','event','feature_video','feature_audio','weighted_total'))and abs(expected-row['weighted_total'])<1e-8
 report=dict(at=time.time(),step=row['step'],loss_valid=healthy,mean_last30_seconds=statistics.mean(r['seconds']for r in rows[-30:]),
             metrics_age_seconds=time.time()-path.stat().st_mtime,peak_reserved_gib=max(r['peak_reserved_gib']for r in rows[-30:]))
 atomic_json(folder/'SEQUENCE_PERFORMANCE.json',report);return report

def run():
 with (HERE/'sequence.lock').open('a')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  atomic_json(HERE/'SEQUENCE_CONTROLLER.json',dict(pid=os.getpid(),start_ticks=Path('/proc',str(os.getpid()),'stat').read_text().split()[21],at=time.time()))
  validate_ready();cancelled()
  emit('armed_waiting_current6000',current=str(PARENT),order=list(ARMS),target_step=TARGET,world=31,reserved_gpu='node11:7')
  while not boundary(read(PARENT/'status.json')['step']):
   cancelled()
   if (PARENT/'STOP').exists():raise RuntimeError('Current training stopped before6000; preserve explicit STOP')
   if not alive(read(PARENT/'CONTROLLER.json')):raise RuntimeError('Current training controller exited before6000')
   time.sleep(.05)
  # status is published BEFORE the scheduled500-step full save. All ranks wait
  # for rank0 checkpoint completion before checking STOP at the next update.
  if not (PARENT/'STOP').exists():atomic_json(PARENT/'STOP',dict(owner='user-component-ablation-sequence6000',target_step=TARGET,at=time.time()))
  emit('waiting_current6000_save_and_exit')
  await_exit(PARENT,read(PARENT/'CONTROLLER.json'))
  receipt=verify_checkpoint(PARENT,'parent');emit('current6000_verified',receipt=receipt)
  for arm in ARMS:
   cancelled();validate_ready();folder=HERE/arm
   if (folder/'run.json').exists():
    if read(folder/'status.json')['step']!=TARGET:raise RuntimeError('Existing unfinished arm requires explicit strict resume: '+arm)
    await_exit(folder);verify_checkpoint(folder,arm);continue
   wait_idle(folder);cancelled();validate_ready()
   logs=folder/('sequence_launch_'+str(int(time.time())))
   command=arm_command(arm,logs);atomic_json(folder/'SEQUENCE_LAUNCH_COMMAND.json',command)
   emit('launching_fresh_arm',arm=arm,stop_after=TARGET,independent_initialization=True)
   with (folder/'sequence_launch.log').open('a')as log:
    p=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
    atomic_json(folder/'SEQUENCE_LAUNCH_PROCESS.json',dict(pid=p.pid,start_ticks=Path('/proc',str(p.pid),'stat').read_text().split()[21]))
    while p.poll()is None:
     # Queue cancellation prevents further launches; never kills live training.
     report=metrics(folder)
     emit('training',arm=arm,performance=report)
     time.sleep(15)
    if p.returncode:raise RuntimeError('Arm failed; block all later arms: '+arm)
   await_exit(folder);receipt=verify_checkpoint(folder,arm)
   atomic_json(folder/'SEQUENCE_FINISHED.json',dict(success=True,at=time.time(),receipt=receipt))
   emit('arm6000_verified',arm=arm,receipt=receipt)
  emit('all_three_finished',target_step=TARGET)
  atomic_json(HERE/'SEQUENCE_FINISHED.json',dict(success=True,at=time.time(),order=list(ARMS),step_each=TARGET))

if __name__=='__main__':
 if len(sys.argv)>1 and sys.argv[1]=='verify':verify_main(Path(sys.argv[2]),sys.argv[3])
 else:
  try:run()
  except Exception:
   atomic_json(HERE/'SEQUENCE_ERROR.json',dict(at=time.time(),traceback=traceback.format_exc()))
   emit('blocked');raise
