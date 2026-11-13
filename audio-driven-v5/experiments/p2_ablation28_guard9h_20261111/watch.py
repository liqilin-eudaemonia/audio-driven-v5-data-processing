"""Nine-hour hourly cluster audit and inexpensive live-process fault detection."""
from pathlib import Path
import os,sys,json,time,statistics,math,subprocess,concurrent.futures,fcntl,signal,traceback
G=Path(__file__).resolve().parent;W=G.parent/'p2_all_lora_ablations28_20261111'
sys.path.insert(0,str(W))
import runner as Q
R=Q.R
from p2_listen.contracts import read,atomic_json,sha
from listen_cluster import transport
PROBE=r'''
import subprocess,json,pathlib,os
r=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,utilization.gpu,memory.used,temperature.gpu,power.draw,clocks.sm','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
gpus=[]
for line in r.stdout.splitlines():
 v=[s.strip()for s in line.split(',')];gpus.append(dict(index=int(v[0]),uuid=v[1],util=int(v[2]),memory_mib=int(v[3]),temp=int(v[4]),power=v[5],sm_clock=v[6]))
r=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
processes=[]
for line in r.stdout.splitlines():
 v=[s.strip()for s in line.split(',')];p=pathlib.Path('/proc')/v[1]
 try:
  raw=(p/'cmdline').read_bytes();cmd=raw.replace(b'\0',b' ').decode(errors='replace');stat=(p/'stat').read_text().rsplit(')',1)[1].split();env=(p/'environ').read_bytes().split(b'\0');visible=next((v.split(b'=',1)[1].decode() for v in env if v.startswith(b'CUDA_VISIBLE_DEVICES=')),None)
  processes.append(dict(uuid=v[0],pid=int(v[1]),memory=v[2],cmd=cmd,argv=raw.decode(errors='replace').split('\0'),start=stat[19],visible=visible))
 except FileNotFoundError:pass
controllers=[]
for p in pathlib.Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  raw=(p/'cmdline').read_bytes();a=raw.decode(errors='replace').split(chr(0))
  if SCRIPT in a and 'node'in a:
   st=(p/'stat').read_text().rsplit(')',1)[1].split();env=(p/'environ').read_bytes().split(bytes([0]));visible=next((v.split(b'=',1)[1].decode()for v in env if v.startswith(b'CUDA_VISIBLE_DEVICES=')),None)
   controllers.append(dict(pid=int(p.name),argv=a,start=st[19],visible=visible))
 except (FileNotFoundError,ProcessLookupError,PermissionError):pass
print(json.dumps(dict(gpus=gpus,processes=processes,controllers=controllers)))
'''
def record(pid):
 p=Path('/proc')/str(pid);return dict(pid=pid,start=(p/'stat').read_text().rsplit(')',1)[1].split()[19],cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode())
def alive(v):
 try:r=record(v['pid']);return r['start']==v['start'] and r['cmd']==v['cmd'] and Path('/proc',str(v['pid']),'stat').read_text().rsplit(')',1)[1].split()[0]!='Z'
 except FileNotFoundError:return False

def owned(p,arm):
 a=p['argv'];return str(W/'runner.py')in a and 'train'in a and '--arm'in a and a[a.index('--arm')+1]==arm

def probe(arm):
 def one(n):
  error=None
  for attempt in range(3):
   try:
    p=subprocess.run(transport(Q.CL,n,[Q.PY,'-c',PROBE.replace('SCRIPT',repr(str(W/'runner.py')))]),capture_output=True,text=True,timeout=45)
    if p.returncode:raise RuntimeError(p.stderr[-500:])
    v=json.loads(p.stdout);cards=[g for g in v['gpus']if g['index']in n['gpus']];uuids={g['uuid']for g in cards};procs=[p for p in v['processes']if p['uuid']in uuids]
    return dict(node=n['name'],cards=cards,workers=[p for p in procs if owned(p,arm)],controllers=[p for p in v['controllers']if '--arm'in p['argv']and p['argv'][p['argv'].index('--arm')+1]==arm],foreign=[p for p in procs if not owned(p,arm)])
   except Exception as e:error=repr(e);time.sleep(2)
  return dict(node=n['name'],error=error)
 with concurrent.futures.ThreadPoolExecutor(4)as e:rows=list(e.map(one,Q.CL['nodes']))
 observed=all('error'not in n and len(n['cards'])==len(Q.CL['nodes'][i]['gpus'])for i,n in enumerate(rows))
 return dict(at=time.time(),nodes=rows,observed=observed,healthy28=observed and sum(len(n['workers'])for n in rows)==28)

def health():
 state=read(W/'STATUS.json');arm=state['arm'];out=R.output(arm);f=out/'metrics.jsonl';rows=[]
 if f.exists():
  for l in f.read_text().splitlines():
   try:rows.append(json.loads(l))
   except json.JSONDecodeError:pass
 rows=rows[-500:];status=read(out/'status.json')if(out/'status.json').exists()else dict(step=0)
 age=time.time()-f.stat().st_mtime if f.exists()else time.time()-state['at'];saving=status['step']%500==0 or status['step']==state['target'];limit=1200 if saving else 600
 flags=[]
 if age>limit and time.time()-state['at']>limit:flags.append('stalled')
 for r in rows[-10:]:
  if any(not math.isfinite(r[k])for k in ['av','event','feature_video','feature_audio','weighted_total']):flags.append('nonfinite')
 recent=rows[-50:];speed=statistics.mean(r['seconds']for r in recent)if recent else None
 return dict(at=time.time(),arm=arm,step=status['step'],target=state['target'],age=age,limit=limit,saving=saving,mean50_seconds=speed,flags=sorted(set(flags)),rows=rows)

def event(kind,**kw):
 v=dict(at=time.time(),kind=kind,**kw)
 with(G/'events.jsonl').open('a')as f:f.write(json.dumps(v)+'\n')
 print(json.dumps(v),flush=True)

def contract(arm):
 out=R.output(arm);c=read(out/'config.json');identity=read(out/'run.json')
 assert c==identity['experiment'] and identity['world']==28 and c['training']['global_batch']==28
 assert c['training']['save_every']==500 and c['training']['gradient_checkpointing']=='all'
 current=R.noevent_runtime().source()if arm=='no_event' else R.source();assert current==identity['source']
 assert all(sha(p)==h for p,h in current.items())
 return dict(success=True,source_files=len(current),config_sha256=sha(out/'config.json'))

SCHEDULE={}
def speed(h):
 arm=h['arm']
 if arm not in SCHEDULE:
  c=read(R.output(arm)/'config.json')
  schedule=R.noevent_runtime().Schedule(c)if arm=='no_event'else R.Schedule(c)
  import csv
  with open(c['sample_list'])as f:dur=[float(r['duration_s'])for r in csv.DictReader(f)]
  SCHEDULE[arm]=(schedule,dur)
 schedule,dur=SCHEDULE[arm];values=[]
 for r in h['rows']:
  if arm=='no_event':a,b=schedule.bounds(r['step']);pairs=schedule.pairs[a:b]
  else:pairs=schedule.pairs[(r['step']-1)*28:r['step']*28]
  mean=statistics.mean(dur[i]for i,_ in pairs)
  values.append(r['seconds']/mean)
 if len(values)<100:return dict(normalized_ratio=1.,enough=False)
 baseline=statistics.median(values[:-50]);recent=statistics.median(values[-50:]);return dict(normalized_ratio=recent/baseline,baseline=baseline,recent=recent,enough=True)

def recover(owner,arm,snap):
 assert snap['observed'],'Incomplete observation cannot authorize restart'
 if(W/'STOP').exists()or(R.output(arm)/'STOP').exists():raise RuntimeError('User STOP must remain effective')
 if alive(owner):
  assert str(W/'runner.py')in owner['cmd']or str(G/'recover.py')in owner['cmd']
  os.kill(owner['pid'],signal.SIGTERM)
  for _ in range(30):
   if not alive(owner):break
   time.sleep(1)
 # Guarded SSH disconnects terminate this launcher\'s process groups. Recheck actual GPU workers.
 after=probe(arm)
 if not after['observed']:raise RuntimeError('Cannot confirm remote exits')
 if any(n['workers']or n['controllers']for n in after['nodes']):
  # Terminate only revalidated exact train PIDs; reservations and unrelated jobs excluded.
  for node,n in zip(Q.CL['nodes'],after['nodes']):
   for p in n['workers']+n['controllers']:
    assert p['visible']==','.join(map(str,node['gpus']))
    code="import pathlib,os,signal; p=pathlib.Path('/proc')/str("+str(p['pid'])+"); a=(p/'cmdline').read_bytes().decode(errors='replace').split(chr(0)); s=(p/'stat').read_text().rsplit(')',1)[1].split()[19]; assert s=="+repr(p['start'])+" and "+repr(str(W/'runner.py'))+" in a and ('train'in a or 'node'in a); os.kill("+str(p['pid'])+",signal.SIGTERM)"
    subprocess.run(transport(Q.CL,node,[Q.PY,'-c',code]),capture_output=True,text=True,timeout=30,check=True)
 time.sleep(3);after=probe(arm)
 if not after['observed']or any(n['workers']or n['controllers']or n['foreign']for n in after['nodes']):raise RuntimeError('Workers or other compute still present; no duplicate launch')
 with(G/'recovery.log').open('a')as f:
  p=subprocess.Popen([Q.PY,'-u',str(G/'recover.py')],stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1'))
 owner=record(p.pid);atomic_json(G/'CURRENT_OWNER.json',owner);event('restarted_latest_complete',owner=owner,arm=arm);return owner

def run():
 protocol=read(G/'protocol.json');next_hour=protocol['start'];index=0;pending=None
 owner=read(G/'CURRENT_OWNER.json');last_repair=0
 with(G/'guard.lock').open('a')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  while True:
   now=time.time();full=now>=next_hour or now>=protocol['deadline']
   try:
    h=health();live=alive(owner);v=speed(h);urgent=bool(h['flags'])or not live or (v['enough']and v['normalized_ratio']>1.3)
    atomic_json(G/'HEARTBEAT.json',dict(at=time.time(),step=h['step'],arm=h['arm'],controller_live=live,deadline=protocol['deadline']))
    if full or urgent:
     snap=probe(h['arm']);v=speed(h);co=contract(h['arm']);compact={k:v for k,v in h.items()if k!='rows'}
     report=dict(health=compact,cluster=snap,contract=co,speed=v,controller_live=live,owner=owner)
     atomic_json(G/'LATEST.json',report)
     if full:
      atomic_json(G/f'hour{index:02d}.json',report);index+=1;next_hour=protocol['start']+index*3600
     event('audit',health=compact,healthy28=snap['healthy28'],observed=snap['observed'],speed=v,controller_live=live)
     # A failed observation is never evidence that training has stopped.
     if snap['observed']:
      failing=bool(h['flags'])or(not live and not (W/'COMPLETE.json').exists())
      if failing and now-last_repair>900:
       if pending and pending['arm']==h['arm'] and now-pending['at']>=55:
        owner=recover(owner,h['arm'],snap);last_repair=time.time();pending=None
       else:pending=dict(arm=h['arm'],at=now)
      elif v['enough']and v['normalized_ratio']>1.3:
       event('slowdown_diagnostic',ratio=v['normalized_ratio'],foreign=[dict(node=n['node'],processes=n['foreign'])for n in snap['nodes']])
       atomic_json(G/'SLOWDOWN_ALERT.json',report)
      else:pending=None
    if now>=protocol['deadline']:
     atomic_json(G/'WINDOW_FINISHED.json',dict(at=time.time(),elapsed=time.time()-protocol['start'],reports=index,training_continues=True,report='LATEST.json'));return
   except Exception as e:
    event('inspection_error',error=repr(e),traceback=traceback.format_exc());atomic_json(G/'ALERT.json',dict(at=time.time(),error=repr(e)))
   time.sleep(min(60,max(1,protocol['deadline']-time.time())))
if __name__=='__main__':run()
