"""Redundant cross-host supervisor for E0+Feature -> E1+Feature, with owned recovery."""
import argparse,fcntl,re,sys,uuid
from queue_common import *


def verify_pins(stage):
 release=read(JOBS[stage]['management']/'release.json')
 if release['arm']!=stage or release['steps']!=BUDGET:raise ValueError('Unexpected pinned experiment')
 for path,h in release['files'].items():
  if sha(path)!=h:raise ValueError('Pinned job input changed: '+path)
 if (OUT/'queue-release.json').exists():
  for path,h in read(OUT/'queue-release.json')['files'].items():
   if sha(path)!=h:raise ValueError('Queue source changed: '+path)


def checkpoint(run,step):
 path=Path(run)/f'step_{step:06d}.pt';marker=read(path.with_suffix('.complete.json'))
 if marker['step']!=step or not path.is_file() or sha(path)!=marker['sha256']:raise ValueError('Invalid checkpoint: '+str(path))
 return dict(path=str(path),step=step,sha256=marker['sha256'])


def latest_checkpoint(run):
 candidates=[]
 for p in Path(run).glob('step_*.complete.json'):
  match=re.fullmatch(r'step_(\d+)\.complete\.json',p.name)
  if match:candidates.append(int(match[1]))
 failures=[]
 for step in sorted(candidates,reverse=True):
  try:return checkpoint(run,step),failures
  except (OSError,ValueError,KeyError) as error:failures.append(str(error))
 return None,failures


def finished(stage,nodes):
 if len(nodes)!=3 or {n['rank'] for n in nodes}!={0,1,2}:return None
 job=JOBS[stage];status=optional(job['run']/'status.json')
 if status.get('step')!=BUDGET or status.get('budget')!=BUDGET or status.get('stopped') is not False:return None
 if any(n['owned'] or not n['locks_free'] for n in nodes):return None
 if any(n['node_state'].get('state')!='finished' or n['node_state'].get('returncode')!=0 for n in nodes):return None
 return checkpoint(job['run'],BUDGET)


def preparation(stage,attempt):
 """Called only after two clean process/lock scans; preserve all discarded attempt artifacts."""
 job=JOBS[stage];run=job['run'];archive=QUEUE/'attempts'/attempt;archive.mkdir(parents=True,exist_ok=True)
 ident=optional(run/'run.json')
 if ident and ident.get('arm')!=stage:raise ValueError('Refuse cross-arm recovery')
 saved,errors=latest_checkpoint(run)
 if saved:
  if saved['step']>BUDGET:raise ValueError('Checkpoint exceeds paired budget')
  status=optional(run/'status.json')
  if status.get('step',0)>saved['step']:(run/'status.json').replace(archive/'rolled_back_run_status.json')
  for p in run.glob('step_*'):
   match=re.match(r'step_(\d+)',p.name)
   if match and int(match[1])>saved['step']:p.replace(archive/p.name)
 elif run.exists():
  # No committed checkpoint: retain the entire failed attempt before deterministic restart.
  run.replace(archive/('uncommitted_run_'+uuid.uuid4().hex))
 for name in ['status.json','launch_results.json','pair_validation.json',*[f'node{r}.json' for r in range(3)]]:
  path=job['management']/name
  if path.exists():path.replace(archive/name)
 result=dict(stage=stage,attempt=attempt,resume=saved['path'] if saved else None,
             checkpoint=saved,rejected_checkpoints=errors,prepared_at=now())
 write(archive/'recovery_plan.json',result)
 return result


class DurableQueue:
 def __init__(self,rank):
  self.rank=rank;self.state=optional(QUEUE/'state.json')
  if not self.state:self.state=dict(stage='E0',phase='waiting_predecessor',attempts={'E0':0,'E1':0},created_at=now(),stage_observed_epoch=time.time())
 def save(self,**values):
  self.state.update(values);self.state.update(updated_at=now(),updated_epoch=time.time(),leader=dict(rank=self.rank,host=socket.gethostname(),pid=os.getpid()))
  write(QUEUE/'state.json',self.state)
 def recover(self,reason):
  stage=self.state['stage'];attempt=f'{stage}_{int(time.time())}_{uuid.uuid4().hex[:8]}'
  counts=self.state.get('attempts',{'E0':0,'E1':0});counts[stage]+=1
  self.save(phase='recovering',attempts=counts,recovery=dict(attempt=attempt,phase='stopping',reason=reason),last_error=None)
 def recovery_tick(self):
  stage=self.state['stage'];rec=self.state['recovery'];verify_pins(stage)
  if rec['phase']=='stopping':
   cluster('stop',stage)
   nodes=cluster('status',stage)
   if any(n['owned'] or not n['locks_free'] for n in nodes):raise RuntimeError('Late owned process/lock prevents recovery preparation')
   rec.update(phase='preparing');self.save(recovery=rec)
  if rec['phase']=='preparing':
   nodes=cluster('status',stage)
   if any(n['owned'] or not n['locks_free'] for n in nodes):
    rec['phase']='stopping';self.save(recovery=rec);return
   if not all(n['idle'] for n in nodes):self.save(phase='waiting_resources',resources=nodes);return
   plan_path=QUEUE/'attempts'/rec['attempt']/'recovery_plan.json'
   plan=read(plan_path) if plan_path.exists() else preparation(stage,rec['attempt'])
   rec.update(phase='launching',resume=plan['resume']);self.save(recovery=rec,attempt_started_epoch=time.time())
  if rec['phase']=='launching':
   nodes=cluster('status',stage)
   if any(n['owned'] for n in nodes):
    # Adopt a launch whose acknowledgement was lost during SSH/leader failure.
    self.save(phase='startup',recovery=None,last_error=None);return
   if not all(n['idle'] for n in nodes):self.save(phase='waiting_resources',resources=nodes);return
   if stopped(stage):self.save(phase='paused');return
   result=remote(0,'launch',stage,resume=rec.get('resume'),attempt=rec['attempt'])
   self.save(phase='startup',recovery=None,launch=result,last_error=None,attempt_started_epoch=time.time())
 def tick(self):
  stage=self.state['stage']
  if stopped(stage):self.save(phase='paused',reason='Explicit STOP present');return
  if self.state.get('phase')=='complete':return
  if self.state.get('recovery'):self.recovery_tick();return
  nodes=cluster('status',stage);progress=metric_progress(JOBS[stage]['run'])
  try:done=finished(stage,nodes)
  except (OSError,ValueError,KeyError) as error:
   self.recover('Final checkpoint verification failed: '+str(error));return
  if done:
   if stage=='E0':
    self.save(stage='E1',phase='queued',predecessor=done,predecessor_verified_at=now(),stage_observed_epoch=time.time(),attempt_started_epoch=0.,progress={'step':0,'time':0.})
    self.recover('Predecessor completed and final checkpoint verified')
   else:self.save(phase='complete',completed_at=now(),checkpoint=done,progress=progress)
   return
  mgmt=optional(JOBS[stage]['management']/'status.json')
  owned=[item for n in nodes for item in n['owned']]
  per_node=[any(p['kind'] in ('torchrun','worker') for p in n['owned']) for n in nodes]
  terminal=optional(JOBS[stage]['run']/'status.json')
  if terminal.get('step')==BUDGET and terminal.get('stopped') is False and owned:
   if time.time()-(JOBS[stage]['run']/'status.json').stat().st_mtime<300:
    self.save(phase='finishing',progress=progress);return
  failed=any(n['node_state'].get('state')=='failed' for n in nodes) or mgmt.get('state')=='failed'
  attempt_start=self.state.get('attempt_started_epoch') or self.state['stage_observed_epoch']
  recent=max(progress['time'],attempt_start)
  if failed:self.recover('Recorded controller/node failure');return
  if not owned:
   self.recover('No live owned processes and no verified completion');return
  if all(per_node):
   timeout=900 if progress['time']>=attempt_start and progress['step'] else 1800
   if time.time()-recent>timeout:self.recover('No training progress within bounded timeout');return
   self.save(phase='waiting_predecessor' if stage=='E0' else 'training',progress=progress,last_error=None,owned_counts=[len(n['owned']) for n in nodes]);return
  if mgmt.get('state')=='waiting_resources':self.save(phase='waiting_resources',progress=progress);return
  # Preflight may not yet allocate CUDA; torchrun itself is still counted above.
  if time.time()-attempt_start>120:self.recover('Partial multinode launch exceeded grace period');return
  self.save(phase='startup',progress=progress)


def heartbeat(rank,role,**extra):
 write(QUEUE/f'watchdog{rank}.json',dict(rank=rank,host=socket.gethostname(),pid=os.getpid(),role=role,time=time.time(),updated_at=now(),**extra))


def run(rank):
 if socket.gethostname()!=HOSTNAMES[rank]:raise ValueError('Watchdog hostname mismatch')
 QUEUE.mkdir(parents=True,exist_ok=True)
 with (QUEUE/f'watchdog{rank}.lock').open('a') as own:
  fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
  while True:
   with (QUEUE/'leader.lock').open('a') as lease:
    try:fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
     heartbeat(rank,'standby');time.sleep(10);continue
    queue=DurableQueue(rank);heartbeat(rank,'leader')
    while True:
     try:
      heartbeat(rank,'leader');queue.tick()
      if queue.state.get('phase')=='complete':heartbeat(rank,'complete');return
      delay=10 if queue.state.get('recovery') else 15
     except Exception as error:
      message=f'{type(error).__name__}: {error}'
      queue.save(last_error=message,retry_at_epoch=time.time()+30)
      print(now(),message,flush=True);heartbeat(rank,'leader',last_error=message);delay=30
     time.sleep(delay)


def main():
 p=argparse.ArgumentParser();p.add_argument('--node-rank',type=int,choices=range(3),required=True);p.add_argument('--detach',action='store_true');a=p.parse_args()
 if socket.gethostname()!=HOSTNAMES[a.node_rank]:raise ValueError('Watchdog hostname mismatch')
 QUEUE.mkdir(parents=True,exist_ok=True)
 if a.detach:
  with (QUEUE/f'watchdog{a.node_rank}.log').open('a') as log:
   child=subprocess.Popen([str(PYTHON),str(Path(__file__).resolve()),'--node-rank',str(a.node_rank)],cwd=ROOT,
      stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
  for _ in range(100):
   state=optional(QUEUE/f'watchdog{a.node_rank}.json')
   if state.get('pid')==child.pid:print(json.dumps(state));return
   if child.poll() is not None:raise RuntimeError('Watchdog failed to start')
   time.sleep(.2)
  raise TimeoutError('Watchdog acknowledgement timeout')
 run(a.node_rank)
if __name__=='__main__':main()
