"""Launch and supervise the authorized 24-GPU E1 only after verification."""
import argparse,fcntl,shlex,subprocess,sys,time,uuid,os
from concurrent.futures import ThreadPoolExecutor
from control import *

def node(rank,*args):
 command=[str(PYTHON),str(OUT/'run_node.py'),'--node-rank',str(rank),*map(str,args)]
 if rank:command=['ssh','-i',KEY,'-o','BatchMode=yes','-o','ConnectTimeout=15','root@'+HOSTS[rank],shlex.join(command)]
 result=subprocess.run(command,capture_output=True,text=True,timeout=90)
 if result.returncode:raise RuntimeError(f'Node {rank}: {result.stderr} {result.stdout}')
 return json.loads(result.stdout) if result.stdout.strip() else None

def parallel(*args):
 with ThreadPoolExecutor(max_workers=3) as pool:return list(pool.map(lambda rank:node(rank,*args),range(3)))

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--detach',action='store_true');parser.add_argument('--resume',type=Path)
 parser.add_argument('--check',action='store_true',help='Read-only source/config and command validation; never starts a process')
 args=parser.parse_args()
 validate_resume(args.resume)
 if args.check:
  from run_node import command
  verify_release()
  print(json.dumps(dict(status='prepared',arm='E1',launched=False,config=str(OUT/'config.json'),output=str(E1),commands=[command(r,args.resume) for r in range(3)]),indent=2));return
 if args.detach:
  command=[str(PYTHON),str(Path(__file__).resolve())]
  if args.resume:command+=['--resume',str(args.resume.resolve())]
  with (OUT/'controller.log').open('a') as log:
   child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  print(json.dumps(dict(controller_pid=child.pid)));return
 with (OUT/'controller.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  state=dict(state='verifying',controller_pid=os.getpid(),started_at=now());write(OUT/'status.json',state)
  launched=False
  try:
   verify_release()
   if (OUT/'STOP').exists():raise RuntimeError('Controller STOP present')
   evidence=read(OUT/'verification.json')
   if evidence['status']!='passed':raise ValueError('Verification incomplete')
   if not args.resume and (E1/'run.json').exists():raise ValueError('E1 exists; explicit --resume required')
   while True:
    if (OUT/'STOP').exists():
     state.update(state='paused',reason='STOP before GPU admission',updated_at=now());write(OUT/'status.json',state);return
    probes=parallel('--probe')
    if all(p['idle'] for p in probes):break
    state.update(state='waiting_resources',gpus=probes,updated_at=now());write(OUT/'status.json',state)
    time.sleep(15)
   write(OUT/'gpu_admission.json',probes)
   session=uuid.uuid4().hex;state.update(state='launching',session=session);write(OUT/'status.json',state)
   # Release inputs have been checked before all three detached supervisors start.
   launched=True
   options=['--detach','--session',session]
   if args.resume:options+=['--resume',str(args.resume.resolve())]
   nodes=parallel(*options);write(OUT/'launch_results.json',nodes)
   paired=False;stop_requested=False
   while True:
    nodes=[read(OUT/f'node{r}.json') for r in range(3)]
    if any(n['state']=='failed' for n in nodes):raise RuntimeError('Training node failed; inspect node logs')
    if (OUT/'STOP').exists() and not stop_requested:
     (E1).mkdir(parents=True,exist_ok=True);(E1/'STOP').touch();stop_requested=True
    if not paired and (E1/'run.json').exists():
     write(OUT/'pair_validation.json',verify_pair());paired=True
    metrics=E1/'metrics.jsonl';last=0
    if metrics.exists():
     with metrics.open('rb') as f:
      f.seek(0,2);f.seek(max(0,f.tell()-30000));lines=f.read().decode().splitlines()
     for line in reversed(lines):
      try:last=int(json.loads(line)['step']);break
      except (ValueError,KeyError):pass
    phase='stopping' if stop_requested else ('training' if paired else 'startup')
    state.update(state=phase,step=last,pair_verified=paired,updated_at=now());write(OUT/'status.json',state)
    if all(n['state']=='finished' for n in nodes):
     status=read(E1/'status.json')
     if not paired:raise RuntimeError('Training exited without pair validation')
     checkpoint=E1/f"step_{status['step']:06d}.pt";marker=read(checkpoint.with_suffix('.complete.json'))
     if marker['step']!=status['step'] or sha(checkpoint)!=marker['sha256']:raise ValueError('Final checkpoint corrupt')
     if status['stopped']:
      if not stop_requested:raise RuntimeError('Unexpected early training stop')
      state.update(state='paused',step=status['step'])
     else:
      if status['step']!=BUDGET or status['budget']!=BUDGET:raise ValueError('Incomplete budget')
      state.update(state='complete',step=BUDGET)
     state.update(finished_at=now(),checkpoint=str(checkpoint),sha256=marker['sha256']);write(OUT/'status.json',state);return
    time.sleep(10)
  except BaseException as error:
   if launched:
    try:parallel('--stop-owned')
    except Exception as stopping:state['stop_error']=str(stopping)
   state.update(state='failed',error=f'{type(error).__name__}: {error}',updated_at=now());write(OUT/'status.json',state);raise
if __name__=='__main__':main()
