"""Scoped recovery for the two explicitly authorized31-rank ablations."""
from pathlib import Path
import os,sys,time,json,signal,subprocess,fcntl,concurrent.futures,runpy,traceback
ROOT=Path('/private/liqi_workspace/project_code/audio-driven-v5');WORK=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from p2_listen.contracts import read,sha,atomic_json
from listen_cluster import transport,launch
G=runpy.run_path(str(WORK/'guardian.py'),run_name='recovery_observations')
F=G['F'];W=G['W'];alive=G['alive'];probe=G['probe'];tail=G['tail']

def cancelled(*,finishing=False):
 # The deadline stops new interventions. Once owned ranks were terminated,
 # finish restoring the existing training even if the observation window ends.
 # Explicit training STOP is checked separately and always takes precedence.
 return not finishing and ((WORK/'STOP').exists()or time.time()>=read(WORK/'protocol.json')['deadline'])

def explicit_stop(out):
 if not (out/'STOP').exists():return False
 v=read(out/'STOP')
 # Only the scheduled6000 transition may be rewound if that save failed.
 return not(out==F and v.get('owner')=='user-av-feature6000-to-av-whisper31' and not (out/'step_006000.complete.json').exists())

def validate(out):
 from p2_whisper_av.bootstrap import activate_ltx
 activate_ltx()
 import p2_current500h_av_feature as feature
 import p2_current500h_av_whisper as whisper
 runtime=feature if out==F else whisper
 runtime.load(out/'config.json');c=read(out/'config.json');run=read(out/'run.json')
 G['contract'](out)
 if c['training']['global_batch']!=31 or c['training']['save_every']!=500 or c['training']['gradient_checkpointing']!='all':raise ValueError('Stable training contract changed')
 runtime.layout31(read(out/'cluster.json'))
 from p2_current_av_feature.data import data_identity as fid
 from p2_current_av_whisper.data import data_identity as wid
 if run['data_identity']!=(fid(c)if out==F else wid(c)):raise ValueError('Frozen data/cache changed')
 return runtime,c,run

def snapshot_valid(snap,cluster):
 expected={n['name']:n['gpus']for n in cluster['nodes']}
 return (len(snap['nodes'])==len(expected) and {n['node']for n in snap['nodes']}==set(expected)
  and all(not n.get('error') and not n.get('errors') and sorted(c['gpu']for c in n.get('cards',[]))==expected[n['node']]for n in snap['nodes']))

def idle(snap,cluster):
 return snapshot_valid(snap,cluster)and all(not n['workers']and not n.get('foreign')and all(c['used_mib']<2048 and c['util']<10 for c in n['cards'])for n in snap['nodes'])

def latest_checked(out,run):
 from p2_current_av_feature import checkpoint as fcp
 from p2_current_av_whisper import checkpoint as wcp
 cp=fcp if out==F else wcp
 rejected=[]
 for marker in reversed(sorted(out.glob('step_*.complete.json'))):
  p=marker.with_suffix('').with_suffix('.pt')
  try:
   v=cp.checked(p)
   if v['identity']!=run or len(v['rank_rng'])!=31:raise ValueError('checkpoint run/RNG identity mismatch')
   return p,v['step'],rejected
  except Exception as e:rejected.append(dict(path=str(p),error=repr(e)))
 raise RuntimeError('No valid full checkpoint available: '+str(rejected))

KILL=r'''from pathlib import Path
import os,signal,json
x=json.loads(PAYLOAD);results=[]
for w in x['workers']:
 p=Path('/proc',str(w['pid']))
 try:
  a=[v.decode()for v in (p/'cmdline').read_bytes().split(b'\0')if v]
  st=(p/'stat').read_text().rsplit(')',1)[1].split()
  e=dict(v.split('=',1)for v in (p/'environ').read_bytes().decode().split('\0')if '='in v)
  if (st[19]!=w['start_ticks']or x['script']not in a or x['config']not in a or 'train'not in a
      or e.get('RANK')!=str(w['rank'])or e.get('WORLD_SIZE')!='31'or e.get('CUDA_VISIBLE_DEVICES')!=x['mask']):raise RuntimeError('process identity changed')
  os.kill(w['pid'],signal.SIGTERM);results.append(dict(pid=w['pid'],terminated=True))
 except ProcessLookupError:results.append(dict(pid=w['pid'],already_exited=True))
 except FileNotFoundError:results.append(dict(pid=w['pid'],already_exited=True))
print(json.dumps(results))
'''

def terminate(out,snap,cluster):
 def one(n):
  observed=next(x for x in snap['nodes']if x['node']==n['name'])
  x=dict(workers=observed['workers'],script=cluster['training_script'],config=str(out/'config.json'),mask=','.join(map(str,n['gpus'])))
  r=subprocess.run(transport(cluster,n,[cluster['python'],'-c',KILL.replace('PAYLOAD',repr(json.dumps(x)))]),capture_output=True,text=True,timeout=45)
  if r.returncode:raise RuntimeError(n['name']+': '+r.stderr[-1000:])
  return dict(node=n['name'],result=json.loads(r.stdout))
 with concurrent.futures.ThreadPoolExecutor(4)as pool:return list(pool.map(one,cluster['nodes']))

def stop_controller(out,record):
 if not alive(record):return
 argv=Path('/proc',str(record['pid']),'cmdline').read_bytes().split(b'\0')
 if str(out/'coordinator.py').encode()not in argv and not(str(WORK/'recovery.py').encode()in argv and str(out).encode()in argv):
  raise ValueError('Controller identity does not belong to this experiment')
 os.kill(record['pid'],signal.SIGTERM)


def recover(out,reason):
 if out not in (F,W):raise ValueError('Out of authorized scope')
 if cancelled()or explicit_stop(out):raise RuntimeError('User stop/deadline preserved')
 if out==F and (out/'step_006000.complete.json').exists():return dict(skipped='6000 save exists; handoff owns next action')
 with (WORK/'recovery.lock').open('a')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  runtime,c,run=validate(out);cluster=read(out/'cluster.json')
  incident=WORK/f'incident_{int(time.time())}';incident.mkdir()
  cp,step,rejected=latest_checked(out,run)
  atomic_json(incident/'CHECKPOINT_VERIFIED.json',dict(path=str(cp),step=step,rejected=rejected,sha256=sha(cp)))
  # All read/verification can take time; re-evaluate progress before signalling.
  rows=tail(out);age=time.time()-(out/'metrics.jsonl').stat().st_mtime if rows else 9999
  grace=600 if rows and rows[-1]['step']%500==0 and not(out/f"step_{rows[-1]['step']:06d}.complete.json").exists()else 180
  if reason not in ('sustained_slowdown','confirmed_dead')and age<=grace:return dict(skipped='Training advanced during diagnosis')
  before=probe(out);atomic_json(incident/'before.json',before)
  if not snapshot_valid(before,cluster):raise RuntimeError('Node/GPU observation incomplete; do not terminate or launch')
  if reason=='confirmed_dead' and (alive(read(out/'CONTROLLER.json'))or before['healthy31']and age<=grace):
   return dict(skipped='Controller/live worker evidence no longer confirms failure')
  if reason=='sustained_slowdown':
   if any(n.get('foreign')for n in before['nodes']):return dict(skipped='Foreign workload contention; keep healthy training running',incident=str(incident))
   # Require fresh speed evidence captured by the observer, not an old alert.
   h=read(WORK/'status.json')
   if h['active']!=out.name or h.get('duration_adjusted_speed_ratio',0)<1.8 or time.time()-h['at']>60:return dict(skipped='Slowdown no longer confirmed')
  if cancelled()or explicit_stop(out):raise RuntimeError('User stop/deadline preserved')
  if reason not in ('sustained_slowdown','confirmed_dead')and time.time()-(out/'metrics.jsonl').stat().st_mtime<=grace:
   return dict(skipped='Training advanced during process query')
  atomic_json(incident/'termination.json',terminate(out,before,cluster))
  controller=read(out/'CONTROLLER.json');until=time.time()+150
  while True:
   check=probe(out)
   if not snapshot_valid(check,cluster):raise RuntimeError('Observation failed during cleanup; do not launch')
   if not any(n['workers']for n in check['nodes']):break
   if time.time()>until:raise RuntimeError('Owned workers have not exited; do not duplicate launch')
   time.sleep(5)
  if alive(controller):
   until=time.time()+20
   while alive(controller)and time.time()<until:time.sleep(1)
   stop_controller(out,controller)
  if alive(controller):
   until=time.time()+20
   while alive(controller)and time.time()<until:time.sleep(1)
   if alive(controller):raise RuntimeError('Original controller remains alive')
  if (out/'STOP').exists():
   if explicit_stop(out):raise RuntimeError('Preserve user STOP')
   (out/'STOP').rename(incident/'scheduled6000_stop_before_rewind.json')
  until=time.time()+180
  while True:
   check=probe(out);atomic_json(incident/'after.json',check)
   if idle(check,cluster):break
   if not snapshot_valid(check,cluster):raise RuntimeError('Incomplete GPU query; do not launch')
   if time.time()>until:raise RuntimeError('Selected31 GPUs not free; foreign tasks remain untouched')
   time.sleep(5)
  if cancelled(finishing=True)or explicit_stop(out):raise RuntimeError('User training STOP preserved')
  validate(out)
  if (out/'QUEUE_ERROR.json').exists():(out/'QUEUE_ERROR.json').rename(incident/'previous_queue_error.json')
  with (incident/'launcher.log').open('ab')as log:
   child=subprocess.Popen([cluster['python'],'-u',str(Path(__file__).resolve()),'launch',str(out),str(cp),str(incident/'launch')],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1'))
  record=dict(pid=child.pid,start_ticks=Path('/proc',str(child.pid),'stat').read_text().rsplit(')',1)[1].split()[19],at=time.time(),work=str(out),resume=str(cp),step=step,reason=reason,incident=str(incident))
  atomic_json(incident/'RESTART.json',record);atomic_json(WORK/'CURRENT_RECOVERY_LAUNCHER.json',record)
  return dict(restarted=True,**record)


def relaunch_queue():
 # Re-run the exact pending handoff controller only after it is authoritatively dead.
 if cancelled()or(W/'QUEUE_STOP').exists()or(W/'STOP').exists():raise RuntimeError('Preserve stop/deadline')
 record=read(W/'CONTROLLER.json')
 if alive(record):return dict(skipped='Handoff controller alive')
 if (W/'run.json').exists():return dict(skipped='Whisper run exists; use strict checkpoint recovery')
 module=runpy.run_path(str(W/'coordinator.py'),run_name='queue_ready_check');module['validate_ready']()
 stamp=WORK/f'queue_repair_{int(time.time())}';stamp.mkdir()
 if(W/'QUEUE_ERROR.json').exists():(W/'QUEUE_ERROR.json').rename(stamp/'previous_error.json')
 with(stamp/'queue.log').open('ab')as log:
  c=read(W/'cluster.json');p=subprocess.Popen([c['python'],'-u',str(W/'coordinator.py')],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1'))
 return dict(queue_restarted=True,pid=p.pid,at=time.time())


def launcher(out,cp,logs):
 with(out/'queue.lock').open('a')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  runtime,c,run=validate(out);cluster=read(out/'cluster.json')
  atomic_json(out/'CONTROLLER.json',dict(pid=os.getpid(),start_ticks=Path('/proc',str(os.getpid()),'stat').read_text().rsplit(')',1)[1].split()[19],at=time.time(),owner='9h_guardian_recovery'))
  atomic_json(out/'QUEUE_STATE.json',dict(stage='guardian_strict_resume',at=time.time(),checkpoint=str(cp),controller_pid=os.getpid()))
  launch(cluster,runtime.commands(cluster,out/'config.json',resume=cp),logs)
  atomic_json(out/'QUEUE_STATE.json',dict(stage='training_finished',at=time.time(),status=read(out/'status.json')))

if __name__=='__main__':
 try:
  if sys.argv[1]=='launch':launcher(Path(sys.argv[2]),Path(sys.argv[3]),Path(sys.argv[4]))
  elif sys.argv[1]=='queue':atomic_json(WORK/'QUEUE_REPAIR_RESULT.json',relaunch_queue())
  else:atomic_json(WORK/'RECOVERY_RESULT.json',recover(Path(sys.argv[2]),sys.argv[3]))
 except Exception:
  atomic_json(WORK/'RECOVERY_ERROR.json',dict(at=time.time(),traceback=traceback.format_exc()));raise
