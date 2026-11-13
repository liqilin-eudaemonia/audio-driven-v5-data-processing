"""Independent 9h observations; checkpoints, jobs and models remain authoritative."""
from pathlib import Path
import sys,os,json,time,math,statistics,fcntl,subprocess,concurrent.futures,csv
ROOT=Path('/private/liqi_workspace/project_code/audio-driven-v5');WORK=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from p2_listen.contracts import read,sha,atomic_json
from listen_cluster import transport
F=ROOT/'experiments/p2_current500h_av_feature31_20261107'
W=ROOT/'experiments/p2_current500h_av_whisper31_20261107'
PROBE=r'''from pathlib import Path
import json,subprocess,os
x=json.loads(PAYLOAD);gpus=x['gpus'];mask=','.join(map(str,gpus));errors=[]
def query(args):
 r=subprocess.run(['nvidia-smi','-i',mask,*args,'--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=20)
 if r.returncode:raise RuntimeError(r.stderr)
 return r.stdout.splitlines()
cards=[]
for line in query(['--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu']):
 i,u,m,f,v=[a.strip()for a in line.split(',')];cards.append(dict(gpu=int(i),uuid=u,used_mib=float(m),free_mib=float(f),util=float(v)))
apps=[]
for line in query(['--query-compute-apps=pid,gpu_uuid,used_gpu_memory']):
 a=[v.strip()for v in line.split(',')]
 if len(a)==3 and a[0].isdigit():apps.append(dict(pid=int(a[0]),uuid=a[1],used_mib=a[2]))
record=Path(x['work'])/'rank_logs'/(x['node']+'_workers.json');pids={a['pid']for a in apps}
if record.exists():pids.update(json.loads(record.read_text())['pids'])
workers=[];foreign=[]
for pid in pids:
 p=Path('/proc',str(pid))
 try:
  argv=[v.decode()for v in (p/'cmdline').read_bytes().split(b'\0')if v]
  if x['script']not in argv or x['config']not in argv:
   if pid in {a['pid']for a in apps}:foreign.append(dict(pid=pid,argv=argv[:8]))
   continue
  env=dict(v.split('=',1)for v in (p/'environ').read_bytes().decode().split('\0')if '='in v)
  if env.get('WORLD_SIZE')!='31' or env.get('CUDA_VISIBLE_DEVICES')!=mask:errors.append('rank env mismatch '+str(pid));continue
  local=int(env['LOCAL_RANK']);rank=int(env['RANK']);st=(p/'stat').read_text().rsplit(')',1)[1].split()
  if st[0]=='Z':continue
  expected=gpus[local];uu=next(c['uuid']for c in cards if c['gpu']==expected)
  workers.append(dict(pid=pid,rank=rank,local=local,gpu=expected,start_ticks=st[19],gpu_context_present=any(a['pid']==pid and a['uuid']==uu for a in apps)))
 except (OSError,KeyError,ValueError,IndexError):continue
print(json.dumps(dict(node=x['node'],cards=cards,workers=workers,foreign=foreign,errors=errors)))
'''

def alive(record):
 try:
  s=Path('/proc',str(record['pid']),'stat').read_text().rsplit(')',1)[1].split()
  return s[0]!='Z' and s[19]==record['start_ticks']
 except (OSError,KeyError):return False

def tail(out):
 p=out/'metrics.jsonl'
 if not p.exists():return []
 with p.open('rb')as f:f.seek(max(0,p.stat().st_size-500000));lines=f.read().splitlines()
 rows=[]
 for line in lines:
  try:rows.append(json.loads(line))
  except ValueError:continue
 return rows

def probe(out):
 cluster=read(out/'cluster.json');script=cluster['training_script']
 def one(n):
  payload=dict(gpus=n['gpus'],node=n['name'],work=str(out),script=script,config=str(out/'config.json'))
  code=PROBE.replace('PAYLOAD',repr(json.dumps(payload)))
  for attempt in range(3):
   try:
    r=subprocess.run(transport(cluster,n,[cluster['python'],'-c',code]),text=True,capture_output=True,timeout=55)
    if r.returncode:raise RuntimeError(r.stderr[-1000:])
    return json.loads(r.stdout)
   except Exception as e:error=repr(e)
  return dict(node=n['name'],error=error)
 with concurrent.futures.ThreadPoolExecutor(4)as pool:nodes=list(pool.map(one,cluster['nodes']))
 expected={n['name']:n['gpus']for n in cluster['nodes']};ranks=[r['rank']for n in nodes for r in n.get('workers',[])]
 healthy=(sorted(ranks)==list(range(31)) and all(not n.get('error') and not n.get('errors') and sorted(c['gpu']for c in n['cards'])==expected[n['node']] and all(w['gpu_context_present']for w in n['workers'])for n in nodes))
 return dict(at=time.time(),work=str(out),healthy31=healthy,nodes=nodes,ranks=sorted(ranks),reserved_gpu='node11:7')

def contract(out):
 c=read(out/'config.json');run=read(out/'run.json')if (out/'run.json').exists()else None
 if run:
  if c!=run['experiment']or run['world']!=31:raise ValueError('run/config/world changed')
  changed=[f for f,h in run['source'].items()if sha(f)!=h]
  if changed:raise ValueError('Source changed '+str(changed))
 return dict(success=True,configuration_sha256=sha(out/'config.json'),run_present=run is not None,files=len(run['source'])if run else 0)

def health(out,now,phase_started):
 rows=tail(out);step=rows[-1]['step']if rows else 0;flags=[]
 age=now-(out/'metrics.jsonl').stat().st_mtime if rows else now-phase_started
 limit=600 if step and step%500==0 and not (out/f'step_{step:06d}.complete.json').exists()else 180
 if not rows:limit=900
 if age>limit:flags.append('stalled_or_startup_timeout')
 keys=['av','av_video','av_audio','weighted_total']+(['feature_video','feature_audio']if out==F else [])
 if rows:
  for r in rows[-30:]:
   if not all(math.isfinite(r[k])for k in keys):flags.append('nonfinite_loss');break
   expected=r['av']+.025*(r['feature_video']+r['feature_audio'])if out==F else r['av']
   if abs(r['weighted_total']-expected)>1e-8:flags.append('wrong_loss');break
   if r['step']>=3 and any(not math.isfinite(v)or v<=0 for v in r['grad_norm'].values()):flags.append('bad_gradient');break
 return dict(work=str(out),step=step,metric_age_seconds=age,stall_limit_seconds=limit,mean_last50_seconds=statistics.mean(r['seconds']for r in rows[-50:])if rows else None,flags=sorted(set(flags)))

DURATIONS={};BASELINES={}
def speed(out,rows):
 if len(rows)<40:return None
 key=str(out)
 if key not in DURATIONS:
  c=read(out/'config.json')
  with Path(c['sample_list']).open()as f:dur=[float(r['duration_s'])for r in csv.DictReader(f)]
  pairs=read(c['duration_sampling']['manifest'])['pairs'];DURATIONS[key]=(dur,pairs)
 dur,pairs=DURATIONS[key]
 def bucket(step):return int(max(dur[i]for i,o in pairs[(step-1)*31:step*31])//2)
 if key not in BASELINES:
  stored=WORK/(out.name+'_SPEED_BASELINE.json')
  if stored.exists():BASELINES[key]={int(k):v for k,v in read(stored)['bins'].items()}
 # A duration bucket may first occur after the initial baseline was saved.
 # Add its earliest steady samples without relearning established baselines:
 # otherwise common long batches can silently lose speed coverage forever.
 baseline=BASELINES.setdefault(key,{})
 missing={}
 for r in rows:
  if r['step']>=40 and 0<r['seconds']<30:
   i=bucket(r['step'])
   if i not in baseline:missing.setdefault(i,[]).append(r['seconds'])
 additions={i:statistics.median(v[:32])for i,v in missing.items()if len(v)>=8}
 if additions:
  baseline.update(additions)
  atomic_json(WORK/(out.name+'_SPEED_BASELINE.json'),dict(at=time.time(),bins=baseline,duration_bin_seconds=2))
 ratios=[r['seconds']/BASELINES[key][bucket(r['step'])]for r in rows[-30:]if bucket(r['step'])in BASELINES[key]]
 return statistics.median(ratios)if len(ratios)>=16 else None

def event(kind,**kw):
 with (WORK/'events.jsonl').open('a')as f:f.write(json.dumps(dict(at=time.time(),kind=kind,**kw))+'\n')

def repair_action(state,now,grace_until=0):
 if now<grace_until:return None
 flags=state['flags']
 if 'handoff_controller_dead'in flags:return ['queue']
 if 'training_controller_dead'in flags:return ['recover',state['work'],'confirmed_dead']
 if 'stalled_or_startup_timeout'in flags:return ['recover',state['work'],'stalled']
 if (state.get('duration_adjusted_speed_ratio')or 0)>1.8 and state.get('slowdown_persist_seconds',0)>=180 and state.get('slowdown_new_steps',0)>=20:
  return ['recover',state['work'],'sustained_slowdown']
 return None

def transition_flags(out,step,flags,controller_alive):
 # Reaching6000 is not proof of a successful save. A failed6000 save still
 # needs fault recovery from the preceding complete checkpoint.
 expected_exit=out==F and step>=6000 and (F/'step_006000.complete.json').exists()
 result=[f for f in flags if not(expected_exit and f=='stalled_or_startup_timeout')]
 if not controller_alive and not expected_exit:result.append('training_controller_dead')
 return sorted(set(result))

def verify_handoff():
 marker=F/'step_006000.complete.json';p=F/'step_006000.pt'
 receipt=read(marker)
 if receipt['step']!=6000 or sha(p)!=receipt['sha256']:raise ValueError('Feature6000 checkpoint invalid')
 if read(F/'status.json')['step']!=6000 or alive(read(F/'CONTROLLER.json')):
  raise ValueError('Feature did not exit at6000')
 old=probe(F)
 if any(n.get('error')or n.get('workers')for n in old['nodes']):raise ValueError('Old workers remain or cannot verify exit')
 c=read(W/'config.json');r=read(W/'run.json');state=health(W,time.time(),time.time())
 from p2_current_av_whisper.config import load
 load(W/'config.json')
 if c!=r['experiment']or r['world']!=31 or state['step']<20 or state['flags']:
  raise ValueError('Whisper startup not yet accepted')
 current=probe(W)
 if not current['healthy31']:raise ValueError('Whisper31 live rank check failed')
 value=dict(at=time.time(),success=True,feature_checkpoint=str(p),feature_sha256=receipt['sha256'],feature_stopped_at=6000,whisper_step=state['step'],whisper_cluster=current,whisper_contract=contract(W),loss='AV only',speaker_whisper=True,event_loss=False,feature_loss=False)
 atomic_json(WORK/'HANDOFF_ACCEPTED.json',value);event('handoff_verified',step=state['step'])
 return value

def start_repair(action):
 for n in ('RECOVERY_RESULT.json','RECOVERY_ERROR.json','QUEUE_REPAIR_RESULT.json'):
  p=WORK/n
  if p.exists():p.rename(WORK/(str(int(time.time()*1000))+'_'+n))
 python=read(F/'cluster.json')['python']
 with(WORK/'repair.log').open('ab')as f:
  p=subprocess.Popen([python,'-u',str(WORK/'recovery.py'),*action],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1'))
 record=dict(pid=p.pid,start_ticks=Path('/proc',str(p.pid),'stat').read_text().rsplit(')',1)[1].split()[19],at=time.time(),action=action)
 atomic_json(WORK/'CURRENT_REPAIR.json',record);event('repair_started',record=record)
 return record

def main():
 with (WORK/'guard.lock').open('a')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  ticks=Path('/proc',str(os.getpid()),'stat').read_text().rsplit(')',1)[1].split()[19]
  atomic_json(WORK/'GUARDIAN.json',dict(pid=os.getpid(),start_ticks=ticks,at=time.time()))
  protocol=read(WORK/'protocol.json');start=protocol['start'];deadline=protocol['deadline'];last_out=None;phase_started=time.time();last_diagnostic=0
  last_slot=max((read(p)['slot']for p in (WORK/'reports').glob('hour_*.json')),default=-1)
  repair=read(WORK/'CURRENT_REPAIR.json')if(WORK/'CURRENT_REPAIR.json').exists()else None
  attempts=read(WORK/'REPAIR_ATTEMPTS.json')['times']if(WORK/'REPAIR_ATTEMPTS.json').exists()else []
  last_attempt=repair['at']if repair else 0;grace_until=0;slow_since=None;slow_step=0;last_recovery_verified=None;last_handoff_probe=0
  event('started',deadline=deadline)
  pool=concurrent.futures.ThreadPoolExecutor(1);future=None
  while True:
   try:
    now=time.time()
    if (WORK/'STOP').exists():event('explicit_guard_stop');break
    out=W if (W/'run.json').exists()else F
    if out!=last_out:phase_started=now;last_out=out
    state=health(out,now,phase_started);state.update(at=now,deadline=deadline,remaining_seconds=max(0,deadline-now),active=out.name)
    ratio=speed(out,tail(out));state['duration_adjusted_speed_ratio']=ratio
    if ratio and ratio>1.35:state['flags'].append('duration_adjusted_slowdown')
    if ratio and ratio>1.8:
     if slow_since is None:slow_since=now;slow_step=state['step']
    else:slow_since=None;slow_step=state['step']
    state.update(slowdown_persist_seconds=now-slow_since if slow_since else 0,slowdown_new_steps=state['step']-slow_step)
    queue=read(W/'QUEUE_STATE.json');ctrl=read(W/'CONTROLLER.json');state['handoff_queue']=queue;state['handoff_controller_alive']=alive(ctrl)
    if not (W/'run.json').exists()and not state['handoff_controller_alive']:state['flags'].append('handoff_controller_dead')
    training_controller=read(out/'CONTROLLER.json');state['training_controller_alive']=alive(training_controller)
    state['flags']=transition_flags(out,state['step'],state['flags'],state['training_controller_alive'])
    if out==F and state['step']>=6000:
     marker=F/'step_006000.complete.json'
     if marker.exists()and now-marker.stat().st_mtime>900:state['flags'].append('handoff_delayed_over15min')
    if repair and not alive(repair):
     result=None
     for n in ('RECOVERY_RESULT.json','QUEUE_REPAIR_RESULT.json','RECOVERY_ERROR.json'):
      p=WORK/n
      if p.exists()and p.stat().st_mtime>=repair['at']:result=read(p);break
     event('repair_exited',process=repair,result=result)
     if result and (result.get('restarted')or result.get('queue_restarted')):
      attempts.append(now);grace_until=now+(900 if result.get('restarted')else 60)
     elif result and result.get('skipped'):pass
     else:
      attempts.append(now);atomic_json(WORK/'NEEDS_AGENT.json',dict(at=now,reason='Recovery failed',process=repair,result=result))
     repair=None
     atomic_json(WORK/'REPAIR_ATTEMPTS.json',dict(times=attempts))
     if(WORK/'CURRENT_REPAIR.json').exists():(WORK/'CURRENT_REPAIR.json').rename(WORK/f'processed_repair_{int(now*1000)}.json')
    state.update(repair_process=repair,recovery_grace_seconds=max(0,grace_until-now),automatic_recovery_enabled=True)
    atomic_json(WORK/'status.json',state)
    if future is not None and future.done()and future.exception()is not None:
     atomic_json(WORK/'NEEDS_AGENT.json',dict(at=now,reason='Full report failed',error=repr(future.exception())))
     event('report_failed',error=repr(future.exception()));future=None;last_slot-=1
    slot=min(9,int(max(0,now-start)//3600))
    if slot>last_slot and (future is None or future.done()):
     def report(state=state.copy(),slot=slot,out=out):
      snapshot=probe(out);checks=contract(out)
      atomic_json(WORK/'reports'/f'hour_{slot:02d}.json',dict(at=time.time(),slot=slot,state=state,cluster=snapshot,contract=checks))
      if not snapshot['healthy31']:atomic_json(WORK/'NEEDS_AGENT.json',dict(at=time.time(),reason='hourly live31 check failed',snapshot=snapshot))
      event('hour_report',slot=slot,step=state['step'],healthy31=snapshot['healthy31'])
     future=pool.submit(report);last_slot=slot
    if state['flags']and now-last_diagnostic>60:
     atomic_json(WORK/'NEEDS_AGENT.json',dict(at=now,state=state,reason=';'.join(state['flags'])));event('anomaly',state=state);last_diagnostic=now
    action=repair_action(state,now,grace_until)
    attempts=[t for t in attempts if now-t<1800]
    if action and repair is None and now-last_attempt>=60 and now<deadline:
     if len(attempts)>=3:atomic_json(WORK/'NEEDS_AGENT.json',dict(at=now,reason='3 recovery failures/restarts in30min; agent diagnosis required',state=state))
     else:repair=start_repair(action);last_attempt=now
    if (WORK/'CURRENT_RECOVERY_LAUNCHER.json').exists():
     rr=read(WORK/'CURRENT_RECOVERY_LAUNCHER.json')
     if rr['work']==str(out)and state['step']>=rr['step']+3 and state['metric_age_seconds']<30 and not state['flags'] and last_recovery_verified!=rr['at']:
      snap=probe(out)
      if snap['healthy31']:
       atomic_json(WORK/'RECOVERY_ACCEPTED.json',dict(at=now,recovery=rr,step=state['step'],cluster=snap,contract=contract(out)))
       event('recovery_verified',step=state['step']);last_recovery_verified=rr['at']
    if out==W and state['step']>=20 and not(WORK/'HANDOFF_ACCEPTED.json').exists()and now-last_handoff_probe>=60:
     last_handoff_probe=now
     try:verify_handoff()
     except Exception as e:atomic_json(WORK/'HANDOFF_CHECK_PENDING.json',dict(at=now,error=repr(e)))
    if now>=deadline:
     if future:future.result()
     atomic_json(WORK/'WINDOW_ENDED.json',dict(at=time.time(),training_left_running=True));break
   except Exception as e:
    atomic_json(WORK/'LAST_OBSERVATION_ERROR.json',dict(at=time.time(),error=repr(e)))
    event('observation_error',error=repr(e))
   time.sleep(15)
  pool.shutdown(wait=True)
if __name__=='__main__':
 try:main()
 except Exception as e:atomic_json(WORK/'GUARD_ERROR.json',dict(at=time.time(),error=repr(e)));raise
