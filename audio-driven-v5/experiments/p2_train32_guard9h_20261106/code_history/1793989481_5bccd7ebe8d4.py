"""Nine-hour external guardian: fixed all recomputation, exact checkpoint resume."""
from pathlib import Path
import os,sys,time,json,subprocess,concurrent.futures,fcntl,runpy,hashlib,math,statistics,csv,signal
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from p2_listen.contracts import read,atomic_json,sha
from listen_cluster import transport,launch
OUT=ROOT/'experiments/p2_ltx_clip_train32_after_gpu67_20261106'
WORK=ROOT/'experiments/p2_train32_guard9h_20261106'
TRIAL=ROOT/'experiments/p2_ltx_clip_recompute_trial32_20261106'
NATIVE=ROOT/'scripts/p2_ltx_events.py'
PROBE='''import pathlib,json,os,signal,subprocess
config=CONFIG_VALUE;action=ACTION_VALUE
r=subprocess.run(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_gpu_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=15)
if r.returncode:print(json.dumps(dict(error='GPU process query failed: '+r.stderr)));raise SystemExit(0)
workers=[]
candidates={line.split(',')[0].strip() for line in r.stdout.splitlines() if line.split(',')[0].strip().isdigit()}
for pid in candidates:
 p=pathlib.Path('/proc')/pid
 try:
  a=p.joinpath('cmdline').read_bytes().split(b'\\0')
  if config.encode() not in a or b'/private/liqi_workspace/project_code/audio-driven-v5/scripts/p2_ltx_events.py' not in a:continue
  e=p.joinpath('environ').read_bytes().split(b'\\0')
  rank=next((x.split(b'=',1)[1].decode() for x in e if x.startswith(b'RANK=')),None)
  local=next((x.split(b'=',1)[1].decode() for x in e if x.startswith(b'LOCAL_RANK=')),None)
  if rank is None or local is None:continue
  workers.append(dict(pid=int(p.name),rank=int(rank),local=int(local),start_ticks=p.joinpath('stat').read_text().split()[21]))
 except OSError:continue
for w in workers:
 if action=='terminate':
  try:os.kill(w['pid'],signal.SIGTERM)
  except ProcessLookupError:pass
r=subprocess.run(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_gpu_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=15)
if r.returncode:print(json.dumps(dict(error='GPU process query failed: '+r.stderr)));raise SystemExit(0)
m=subprocess.run(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=15)
if m.returncode:print(json.dumps(dict(error='GPU free-memory query failed: '+m.stderr)));raise SystemExit(0)
print(json.dumps(dict(workers=workers,gpu_processes=r.stdout.splitlines(),gpu_error=r.stderr,free_mib=[float(x.split(',')[1]) for x in m.stdout.splitlines() if ',' in x],action=action)))
'''

def alive(pid,ticks=None):
 try:
  s=Path('/proc',str(pid),'stat').read_text().split();return s[2]!='Z' and (ticks is None or s[21]==ticks)
 except OSError:return False

def log(kind,**kw):
 r=dict(kind=kind,at=time.time());r.update(kw)
 with (WORK/'events.jsonl').open('a') as f:f.write(json.dumps(r)+'\n')
 print(json.dumps(r),flush=True);return r

def probes(cluster,action='probe'):
 code=PROBE.replace('CONFIG_VALUE',repr(str(OUT/'config.json'))).replace('ACTION_VALUE',repr(action))
 def one(n):
  for attempt in range(4):
   try:
    r=subprocess.run(transport(cluster,n,[cluster['python'],'-c',code]),capture_output=True,text=True,timeout=25)
    if r.returncode==0:return dict(node=n['name'],**json.loads(r.stdout))
    error=r.stderr[-1000:]
   except Exception as e:error=repr(e)
   time.sleep(1)
  return dict(node=n['name'],error=error)
 with concurrent.futures.ThreadPoolExecutor(4) as pool:return list(pool.map(one,cluster['nodes']))

def foreign_processes(observations):
 result=[]
 for n in observations:
  own={str(p['pid']) for p in n.get('workers',[])}
  result.extend(dict(node=n['node'],process=line) for line in n.get('gpu_processes',[]) if line.split(',')[0].strip() not in own)
 return result

def ready_after_own_cleanup(observations):
 return all('error' not in n and not n['workers'] and len(n.get('free_mib',[]))==8 and min(n['free_mib'])>=108*1024 for n in observations)

def full_report(state,cluster):
 try:
  state=dict(state,processes=probes(cluster),completed_at=time.time())
  path=WORK/'reports'/f"{int(state['at'])}.json";atomic_json(path,state)
  log('half_hour_report',step=state['step'],report=str(path),speed_ratio=state['duration_adjusted_speed_ratio'])
 except Exception as e:log('report_error',error=repr(e))

def slowdown_report(state,cluster):
 try:
  atomic_json(WORK/f"SLOWDOWN_{int(state['at'])}.json",dict(state=state,processes=probes(cluster),completed_at=time.time()))
  log('slowdown_detected',step=state['step'],ratio=state['duration_adjusted_speed_ratio'])
 except Exception as e:log('slowdown_report_error',error=repr(e))

def metrics():
 p=OUT/'metrics.jsonl'
 if not p.exists():return []
 with p.open('rb') as f:
  size=p.stat().st_size;f.seek(max(0,size-1500000));lines=f.read().splitlines()
 if size>1500000:lines=lines[1:]
 out=[]
 for line in lines:
  try:out.append(json.loads(line))
  except ValueError:pass
 return out

def latest():
 files=sorted(OUT.glob('step_*.complete.json'))
 for receipt in reversed(files):
  path=receipt.with_suffix('').with_suffix('.pt')
  if path.is_file() and path.stat().st_size>0:return path
 raise RuntimeError('No complete checkpoint to resume')

def verify_contract():
 c=read(OUT/'config.json');r=read(OUT/'run.json');protocol=read(WORK/'protocol.json')
 if sha(OUT/'config.json')!=protocol['config_sha256'] or c!=r['experiment']:raise ValueError('Production configuration changed')
 if c['training']['gradient_checkpointing']!='all' or c['training']['save_every']!=300 or r['world']!=32:raise ValueError('Stable all/32GPU/save300 contract changed')
 for f,s in r['source'].items():
  if sha(f)!=s:raise ValueError('Training source changed: '+f)
 return c

def launcher(checkpoint,logs):
 from p2_whisper_av.bootstrap import activate_ltx
 activate_ltx()
 verify_contract();cluster=read(OUT/'cluster.json')
 cli=runpy.run_path(str(NATIVE),run_name='guardian_commands')
 atomic_json(OUT/'QUEUE_STATE.json',dict(state='training',phase='guardian_restart',controller_pid=os.getpid(),resume=str(checkpoint),at=time.time()))
 launch(cluster,cli['commands'](cluster,OUT/'config.json',resume=checkpoint),logs)
 atomic_json(OUT/'QUEUE_STATE.json',dict(state='finished',controller_pid=os.getpid(),at=time.time()))

def start_monitor(cluster):
 r=read(OUT/'MONITOR.json') if (OUT/'MONITOR.json').exists() else {}
 if r.get('pid') and alive(r['pid']):return
 if (OUT/'MONITOR_STOP').exists():return
 with (OUT/'monitor.log').open('ab',buffering=0) as f:
  p=subprocess.Popen([cluster['python'],'-u',str(OUT/'monitor.py')],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'))
 atomic_json(OUT/'MONITOR.json',dict(pid=p.pid,at=time.time()))

def resume(cluster,reason):
 if (OUT/'STOP').exists() or (WORK/'STOP').exists():raise RuntimeError('Explicit STOP preserved; no restart')
 verify_contract();incident=WORK/f'incident_{int(time.time())}';incident.mkdir()
 before=probes(cluster);atomic_json(incident/'before.json',before)
 # Stop only this exact run's rank workers; launcher guards own their children.
 if any(n.get('workers') for n in before):atomic_json(incident/'termination.json',probes(cluster,'terminate'))
 deadline=time.time()+120
 while True:
  p=probes(cluster)
  if ready_after_own_cleanup(p):break
  if time.time()>deadline:raise RuntimeError('Workers or unrelated GPU jobs remain; refuse overlapping launch')
  time.sleep(5)
 # Original one-shot controller has no further retry after its all fallback fails.
 old=read(TRIAL/'CONTROLLER.json')
 if alive(old['pid'],old.get('start_ticks')):
  deadline=time.time()+30
  while alive(old['pid'],old.get('start_ticks')) and time.time()<deadline:time.sleep(1)
  if alive(old['pid'],old.get('start_ticks')):
   args=Path('/proc',str(old['pid']),'cmdline').read_bytes().split(b'\0')
   if str(ROOT/'scripts/run_p2_ltx_recompute_trial32_20261106.py').encode() not in args:raise RuntimeError('Controller identity changed')
   os.kill(old['pid'],signal.SIGTERM)
 checkpoint=latest();receipt=read(checkpoint.with_suffix('.complete.json'))
 # Native worker startup verifies full SHA, tensor finiteness and identity on every rank.
 with (incident/'launcher.log').open('ab',buffering=0) as f:
  p=subprocess.Popen([cluster['python'],'-u',str(Path(__file__).resolve()),'launch',str(checkpoint),str(incident/'launch')],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1'))
 record=dict(pid=p.pid,start_ticks=Path('/proc',str(p.pid),'stat').read_text().split()[21],checkpoint=str(checkpoint),step=receipt['step'],reason=reason,at=time.time())
 atomic_json(WORK/'CURRENT_LAUNCHER.json',record);atomic_json(incident/'RESTART.json',record)
 start_monitor(cluster);log('restarted',**record);return record

def main():
 WORK.mkdir(exist_ok=True)
 with (WORK/'guard.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  protocol=read(WORK/'protocol.json');c=verify_contract();cluster=read(OUT/'cluster.json')
  # Match speed by the actual scheduled maximum duration, avoiding short/long batch false alarms.
  with (Path(c['current_source'])/'sample_list.csv').open() as f:durations=[float(r['duration_s']) for r in csv.DictReader(f)]
  pairs=read(c['duration_sampling']['manifest'])['pairs']
  def duration(step):return max(durations[x[0]] for x in pairs[(step-1)*32:step*32])
  history=[r for r in metrics() if 85<=r['step']<=299]
  prior=read(WORK/'SPEED_BASELINE.json') if (WORK/'SPEED_BASELINE.json').exists() else None
  bins={}
  for row in history:
   if 0<row['seconds']<30:bins.setdefault(int(duration(row['step'])//2),[]).append(row['seconds'])
  expected={k:statistics.median(v) for k,v in bins.items() if len(v)>=4}
  if prior and prior.get('expected_seconds'):expected={int(k):v for k,v in prior['expected_seconds'].items()}
  atomic_json(WORK/'SPEED_BASELINE.json',dict(duration_bin_seconds=2,expected_seconds=expected,samples={k:len(v) for k,v in bins.items()}))
  last_report=0;last_step=None;last_change=time.time();startup_until=time.time()+480;slow_streak=0;restart_times=[];last_slow_alert=0
  report_pool=concurrent.futures.ThreadPoolExecutor(max_workers=1,thread_name_prefix='readonly_reports');report_future=None;slow_future=None
  log('guardian_started',deadline=protocol['deadline_utc'],policy='all',poll_seconds=15,full_report_seconds=1800)
  while time.time()<protocol['deadline_timestamp']:
   if (WORK/'STOP').exists():log('explicit_guard_stop');break
   try:
    if (OUT/'STOP').exists():log('explicit_training_stop_preserved');break
    rows=metrics();status=read(OUT/'status.json');step=status['step'];now=time.time();latest_row=rows[-1] if rows else None
    if step!=last_step:
     if last_step is not None and step>last_step:startup_until=0
     last_step=step;last_change=now
    if status.get('complete'):log('training_completed',step=step);break
    perf=read(OUT/'PERFORMANCE_STATUS.json') if (OUT/'PERFORMANCE_STATUS.json').exists() else {}
    eligible=[r for r in rows if r['step']>305][-32:]
    ratios=[r['seconds']/expected[int(duration(r['step'])//2)] for r in eligible if int(duration(r['step'])//2) in expected]
    ratio=statistics.median(ratios) if len(ratios)>=16 else None
    age=now-(OUT/'metrics.jsonl').stat().st_mtime
    # Checkpoint save and cold restart legitimately take minutes; allow explicit bounded grace.
    limit=600 if step%300==0 else 120
    fault=(now>startup_until and age>limit)
    bad_loss=latest_row is not None and not all(math.isfinite(latest_row[k]) for k in ['av','event','feature_video','feature_audio','weighted_total'])
    state=dict(at=now,deadline=protocol['deadline_utc'],remaining_seconds=max(0,protocol['deadline_timestamp']-now),step=step,seconds_since_metrics=age,stall_limit_seconds=limit,startup_grace_remaining=max(0,startup_until-now),duration_adjusted_speed_ratio=ratio,mean_recent_seconds=statistics.mean(r['seconds'] for r in eligible) if eligible else None,policy='all',save_every=300,performance=perf,automatic_restart_enabled=True,slowdown_action='Capture diagnostics; restart only on severe sustained compute degradation or lost progress')
    atomic_json(WORK/'status.json',state)
    if now-last_report>=1800 and (report_future is None or report_future.done()):
     report_future=report_pool.submit(full_report,dict(state),cluster);last_report=now
    if ratio is not None and ratio>1.35 and len(eligible)>=20:
     if now-last_slow_alert>1800:
      slow_future=report_pool.submit(slowdown_report,dict(state),cluster);last_slow_alert=now
     slow_streak=slow_streak+1 if ratio>1.8 else 0
    else:slow_streak=0
    # Avoid destructive restart loops; retain an explicit diagnostic for agent intervention.
    restart_times=[t for t in restart_times if now-t<1800]
    if fault or bad_loss or slow_streak>=8:
     if slow_streak>=8 and not fault and not bad_loss:
      sharing=probes(cluster);foreign=foreign_processes(sharing)
      if foreign:
       atomic_json(WORK/'FOREIGN_CONTENTION.json',dict(at=now,processes=foreign,action='Do not restart healthy training for another project sharing GPU; foreign processes untouched'))
       log('foreign_contention_no_destructive_restart',processes=foreign);slow_streak=0;time.sleep(15);continue
     if len(restart_times)>=3:
      atomic_json(WORK/'NEEDS_AGENT.json',dict(at=now,state=state,reason='Three restarts within30min; inspect deterministic failure before another restart'))
     else:
      reason='nonfinite_loss' if bad_loss else 'no_step_progress' if fault else 'sustained_over_1.8x_duration_adjusted_step_time'
      record=resume(cluster,reason);restart_times.append(now);startup_until=time.time()+600;last_change=time.time();last_step=record['step'];slow_streak=0
    start_monitor(cluster)
   except Exception as e:
    log('guardian_error',error=repr(e));atomic_json(WORK/'LAST_ERROR.json',dict(at=time.time(),error=repr(e)))
   time.sleep(15)
  log('guardian_window_ended',training_left_running=True)
  atomic_json(WORK/'COMPLETE.json',dict(at=time.time(),deadline=protocol['deadline_utc'],training_left_running=True))

if __name__=='__main__':
 if len(sys.argv)>1 and sys.argv[1]=='launch':launcher(Path(sys.argv[2]),Path(sys.argv[3]))
 else:main()
