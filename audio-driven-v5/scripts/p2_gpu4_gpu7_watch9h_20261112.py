"""Nine-hour hourly audit with minute liveness recovery; only owned GPU4/7 jobs."""
import fcntl,json,os,signal,statistics,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from p2_testsets_eval.protocol import read,save,sha,PYTHON
import p2_seamless_gpu7_assist_20261112 as assist
WORK=ROOT/'experiments/p2_gpu4_gpu7_watch9h_20261112'
SEAM=ROOT/'experiments/p2_lora6000_seamless_full_gpu4_20261112'
RN=ROOT/'experiments/p2_responsenet_full586_all_lora6000_6500_20261111'
SCOPES={
 'gpu4':dict(work=SEAM,script=ROOT/'scripts/p2_lora6000_seamless_full_gpu4_20261112.py',gpu=4,runs=[SEAM]),
 'gpu7':dict(work=RN,script=ROOT/'scripts/p2_responsenet_full586_all_lora_20261111.py',gpu=7,runs=[RN/'conditioning',RN/'step_006000',RN/'step_006500'])}
UUIDS={4:'GPU-c0ca8be5-8935-4e6c-d3d7-a2c5b58c37c0',7:'GPU-4d5ee808-e829-c824-c506-cad80ff133f5'}


def identity(pid,needles=()):
 try:
  root=Path('/proc')/str(pid);s=root.joinpath('stat').read_text().rsplit(')',1)[1].split();cmd=root.joinpath('cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
  return dict(pid=pid,live=s[0]!='Z' and all(x in cmd for x in needles),state=s[0],start_ticks=s[19],cpu_ticks=int(s[11])+int(s[12]),cmd=cmd)
 except (FileNotFoundError,ProcessLookupError):return dict(pid=pid,live=False)


def controller(name):
 m=SCOPES[name];path=m['work']/'CONTROLLER.json'
 return identity(read(path)['pid'],[str(m['script'])]) if path.exists() else dict(live=False,pid=None)


def children(name):
 result=[]
 for work in SCOPES[name]['runs']:
  for path in (work/'processes').glob('*.json'):
   r=read(path);cmd=r.get('command',[]);needles=list(map(str,cmd[1:])) if cmd else [str(work)]
   i=identity(r['pid'],needles)
   if i['live']:result.append(dict(i,record=str(path),stage=r.get('stage'),slot=r.get('slot'),log=r.get('log')))
 return result


def descendant_pids(roots):
 pending=list(roots);seen=set(roots)
 while pending:
  pid=pending.pop()
  try:children=list(map(int,Path(f'/proc/{pid}/task/{pid}/children').read_text().split()))
  except (FileNotFoundError,ProcessLookupError):continue
  for child in children:
   if child not in seen:seen.add(child);pending.append(child)
 return seen


def gpu_report():
 out=subprocess.check_output(['nvidia-smi','-i','4,7','--query-gpu=index,uuid,utilization.gpu,memory.used,memory.free,power.draw','--format=csv,noheader,nounits'],text=True,timeout=20)
 result={}
 for line in out.splitlines():
  v=[x.strip() for x in line.split(',')];g=int(v[0]);assert v[1]==UUIDS[g]
  result[g]=dict(uuid=v[1],util=float(v[2]),used_mib=float(v[3]),free_mib=float(v[4]),power_w=float(v[5]),processes=[])
 out=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True,timeout=20)
 for line in out.splitlines():
  v=[x.strip() for x in line.split(',')]
  for g,r in result.items():
   if v[0]==r['uuid']:r['processes'].append(dict(pid=int(v[1]),memory_mib=v[2]))
 return result


def timing():
 report={}
 sets=[('seamless6000',SEAM,'ltx_all_lora_event_feature_whisper_6000_sft2752','seamless'),('responsenet6000',RN/'step_006000','all_lora6000_sft2752','responsenet'),('responsenet6500',RN/'step_006500','all_lora6500_sft2752','responsenet')]
 for name,w,arm,ds in sets:
  paths=list((w/'latents'/arm/ds).glob('*/metadata.json'));latest=sorted(paths,key=lambda p:p.stat().st_mtime)[-20:];records=[]
  for path in latest:
   m=read(path)
   records.append(dict(id=path.parent.name,seconds=m['seconds'],frames=m['geometry']['native_frames'],fps=m['geometry']['fps'],at=path.stat().st_mtime,seconds_per_frame=m['seconds']/m['geometry']['native_frames']))
  report[name]=dict(latents=len(paths),videos=len(list((w/'generated'/arm/ds).glob('*/complete.json'))),recent=records,seconds_per_frame_median=statistics.median(r['seconds_per_frame'] for r in records) if records else None)
 return report


def stop_owned(record):
 current=identity(record['pid'],[record['cmd']])
 if not current['live'] or current['start_ticks']!=record['start_ticks']:return False
 if os.getpgid(record['pid'])==record['pid']:os.killpg(record['pid'],signal.SIGTERM)
 else:os.kill(record['pid'],signal.SIGTERM)
 return True


def recovery(name,c,state):
 m=SCOPES[name];w=m['work']
 if time.time()>=c['end_at'] or (WORK/'STOP').exists() or (w/'STOP').exists() or (w/'COMPLETE.json').exists():return
 root=controller(name);live=children(name)
 if root['live'] or live:return
 history=state.setdefault('repairs',{}).setdefault(name,[]);recent=[r for r in history if time.time()-r['at']<3600]
 if len(recent)>=3 or (recent and time.time()-recent[-1]['at']<180):return
 for run in m['runs']:
  path=run/'protocol.json'
  if not path.exists():continue
  p=read(path)
  for path,digest in (p.get('adapter_code',{}) if name=='gpu4' else p.get('execution_code',{})).items():assert sha(path)==digest,('Frozen source changed',path)
 with (w/'controller.lock').open('a') as lock:
  try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:return
 if controller(name)['live'] or children(name):return
 env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(m['gpu']),PYTHONPATH='',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONUNBUFFERED='1')
 command=[str(PYTHON),'-u',str(m['script']),'controller'];log=WORK/'repairs'/f'{name}_{time.time_ns()}.log';log.parent.mkdir(exist_ok=True)
 with log.open('w') as stream:p=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
 action=dict(pid=p.pid,at=time.time(),mode=name,physical_gpu=m['gpu'],command=command,log=str(log),reason='confirmed dead controller, no owned live children, free controller lock')
 save(w/'LAUNCH.json',action);save(WORK/'repairs'/f'{time.time_ns()}.json',action);history.append(action)


def slowdowns(t,state):
 alerts=[];baseline=state.setdefault('baseline_seconds_per_frame',{});shapes=state.setdefault('baseline_shapes',{})
 for name,r in t.items():
  rows=r['recent']
  if len(rows)<3:continue
  if name not in baseline:
   if len(rows)<20:continue
   baseline[name]=statistics.median(x['seconds_per_frame'] for x in rows)
   shapes[name]=statistics.median(x['frames'] for x in rows)
  same=lambda group:abs(statistics.median(x['frames'] for x in group)-shapes[name])<=16
  severe=same(rows[-3:]) and all(x['seconds_per_frame']>2*baseline[name] for x in rows[-3:])
  moderate=len(rows)>=20 and same(rows) and all(statistics.median(x['seconds_per_frame'] for x in group)>1.5*baseline[name] for group in [rows[:10],rows[-10:]])
  if severe or moderate:alerts.append(dict(task=name,type='severe_2x_three_commits' if severe else 'sustained_1.5x_slowdown',baseline=baseline[name],recent=statistics.median(x['seconds_per_frame'] for x in rows[-3:] if severe) if severe else statistics.median(x['seconds_per_frame'] for x in rows),requires_operator_review=True))
 return alerts


def detect_stall(name,g,state,c):
 m=SCOPES[name];path=m['work']/'status.json';s=read(path) if path.exists() else {};now=time.time()
 if (m['work']/'COMPLETE.json').exists():return
 progress=(s.get('step'),s.get('generated',s.get('videos')),s.get('new_latents',s.get('latents')),s.get('scored_shards'),s.get('stage'))
 previous=state.setdefault('progress',{}).setdefault(name,dict(value=list(progress),changed_at=now,idle_ticks=0))
 if list(progress)!=previous['value']:previous.update(value=list(progress),changed_at=now)
 previous['idle_ticks']=previous.get('idle_ticks',0)+1 if g['util']<5 else 0
 live=children(name);owners=descendant_pids({r['pid'] for r in live});foreign=[p for p in g['processes'] if p['pid'] not in owners]
 if now-previous['changed_at']<900 or previous['idle_ticks']<5 or foreign:return
 # CPU-only merge/bind and deliberate admission waits are valid phases.
 if s.get('stage') not in ('generation_and_export','cache_generation_decode_scoring'):return
 candidates=[r for r in live if r['stage']=='render']
 for r in candidates:
  last=state.setdefault('cpu',{}).get(str(r['pid']));state['cpu'][str(r['pid'])]=dict(at=now,ticks=r['cpu_ticks'])
  if last is None or (r['cpu_ticks']-last['ticks'])/max(1,now-last['at'])>2:continue
  if not controller(name)['live']:continue
  if stop_owned(r):
   save(WORK/'repairs'/f'stall_{time.time_ns()}.json',dict(at=now,reason='15min unchanged outputs plus five idle GPU checks and inactive owned renderer CPU',process=r,scope=name));previous['changed_at']=now;break


def shared_gpu4_ready():
 p=SEAM/'processes/render_0.json';device=SEAM/'devices/render_0.json'
 if not p.exists() or not device.exists():return False
 r=read(p);d=read(device)
 return identity(r['pid'],[str(SCOPES['gpu4']['script']),'worker','render'])['live'] and d.get('pid')==r['pid'] and d.get('claim_protocol')=='per-sample-flock-v1'


def enable_gpu7(c,gpus,state):
 if time.time()>=c['end_at'] or not assist.eligible() or (RN/'STOP').exists() or (SEAM/'STOP').exists() or (SEAM/'COMPLETE.json').exists():return
 launch=SEAM/'ASSIST_GPU7_LAUNCH.json'
 if (SEAM/'GPU7_GENERATION_FINISHED.json').exists():return
 if launch.exists():
  owned=identity(read(launch)['pid'],[str(ROOT/'scripts/p2_seamless_gpu7_assist_20261112.py')])
  if owned['live']:
   progress=SEAM/'progress/render/07.json';last=read(progress).get('at',read(launch)['at']) if progress.exists() else read(launch)['at']
   idle=state.get('assist_idle_ticks',0)+1 if gpus[7]['util']<5 else 0;state['assist_idle_ticks']=idle
   if time.time()-last>900 and idle>=5:
    stop_owned(owned);save(WORK/'repairs'/f'assist_stall_{time.time_ns()}.json',dict(at=time.time(),reason='owned GPU7 peer15min no commit and five idle checks',process=owned))
   return
 # Wait for the final ResponseNet metric processes and any foreign job to exit.
 if children('gpu7') or gpus[7]['processes'] or gpus[7]['free_mib']<90*1024:return
 root=controller('gpu4')
 if not root['live']:return
 if not shared_gpu4_ready():
  # The old resident was loaded before shared claims were installed. Rotate at
  # its NEXT complete commit, never by killing an arbitrary healthy sample.
  phase=state.setdefault('handoff',{})
  progress=SEAM/'progress/render/00.json';r=read(progress) if progress.exists() else {}
  marker=(r.get('pid'),r.get('at'))
  if 'initial_marker' not in phase:phase['initial_marker']=list(marker);phase['wait_started']=time.time();return
  if list(marker)==phase['initial_marker']:return
  if time.time()-r.get('at',0)>5:
   phase['initial_marker']=list(marker);return
  job=read(SEAM/'processes/render_0.json');owned=identity(job['pid'],[str(SCOPES['gpu4']['script']),'worker','render'])
  if not owned['live']:return
  # New records arrive only after atomic latent commit. Pause the controller
  # briefly so it cannot launch a duplicate while its renderer is replaced.
  os.kill(root['pid'],signal.SIGSTOP)
  try:
   if not identity(root['pid'],[str(SCOPES['gpu4']['script'])])['live']:return
   stop_owned(owned);phase.update(rotation_at=time.time(),old_pid=owned['pid'],reason='new committed latent after release, install shared claims before peer start')
  finally:
   if identity(root['pid'],[str(SCOPES['gpu4']['script'])])['live']:os.kill(root['pid'],signal.SIGCONT)
  save(WORK/'HANDOFF_GPU4_ROTATION.json',phase);return
 if not shared_gpu4_ready():return
 auth=dict(authorized=True,safe_shared_claims_verified=True,at=time.time(),gpu7_priority_task='ResponseNet6000 and6500 completed',source='explicit user9h goal',shared_claim_protocol='per-sample-flock-v1')
 save(SEAM/'ASSIST_GPU7.json',auth)
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='7',PYTHONPATH='',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONUNBUFFERED='1')
 command=[str(PYTHON),'-u',str(ROOT/'scripts/p2_seamless_gpu7_assist_20261112.py')];log=SEAM/f'gpu7_assist_{time.time_ns()}.log'
 with log.open('w') as f:p=subprocess.Popen(command,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 action=dict(pid=p.pid,at=time.time(),physical_gpu=7,command=command,log=str(log),safe_shared_claims=True)
 save(launch,action);save(WORK/'GPU7_ASSIST_LAUNCH.json',action);state['gpu7_assist_started']=action


def check(c,state,detail=False):
 g=gpu_report()
 for name in SCOPES:recovery(name,c,state);detect_stall(name,g[SCOPES[name]['gpu']],state,c)
 enable_gpu7(c,g,state)
 report=dict(at=time.time(),gpus=g,controllers={n:controller(n) for n in SCOPES},children={n:children(n) for n in SCOPES},status={n:read(m['work']/'status.json') for n,m in SCOPES.items()},gpu7_released_for_assist=assist.eligible())
 if detail or time.time()-state.get('last_speed_probe',0)>=300:
  report['throughput']=timing();report['alerts']=slowdowns(report['throughput'],state);state['last_speed_probe']=time.time()
  if report['alerts']:save(WORK/'NEEDS_OPERATOR_REVIEW.json',dict(at=time.time(),alerts=report['alerts']))
 return report


def main():
 WORK.mkdir(exist_ok=True)
 with (WORK/'watcher.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  c=read(WORK/'CONFIG.json');assert c['end_at']-c['start_at']==9*3600
  save(WORK/'CONTROLLER.json',dict(pid=os.getpid(),at=time.time(),end_at=c['end_at']))
  state=read(WORK/'STATE.json') if (WORK/'STATE.json').exists() else {}
  while time.time()<c['end_at'] and not (WORK/'STOP').exists():
   try:
    hour=int((time.time()-c['start_at'])//3600);full=hour not in state.get('audited_hours',[])
    report=check(c,state,detail=full);save(WORK/'LIVE.json',report)
    if full:save(WORK/'hourly'/f'{hour:02d}.json',report);state.setdefault('audited_hours',[]).append(hour)
    state.update(at=time.time(),end_at=c['end_at']);save(WORK/'STATE.json',state)
   except Exception as exc:save(WORK/'errors'/f'{time.time_ns()}.json',dict(at=time.time(),error=repr(exc),traceback=traceback.format_exc()))
   fast_handoff=assist.eligible() and state.get('handoff') and not shared_gpu4_ready()
   time.sleep(min(2 if fast_handoff else 60,max(0,c['end_at']-time.time())))
  final=check(c,state,detail=True) if not (WORK/'STOP').exists() else dict(stopped=True)
  save(WORK/'WINDOW_FINISHED.json',dict(at=time.time(),deadline=c['end_at'],stopped=(WORK/'STOP').exists(),evaluations_left_running=True,final=final))


if __name__=='__main__':main()
