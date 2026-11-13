"""Bounded CPU helper; signals only allowlisted jobs in this container's cgroup."""
import argparse,fcntl,signal,sys
from queue_common import *

def tokens(pid):
 try:
  status=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
  if status[0]=='Z':return [],None
  return [s.decode() for s in Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0') if s],status[19]
 except (FileNotFoundError,ProcessLookupError,PermissionError):return [],None

def argument(argv,key):
 try:return argv[argv.index(key)+1]
 except (ValueError,IndexError):return None

def classify(argv,stage,rank):
 job=JOBS[stage];mgmt=job['management']
 if str(mgmt/'controller.py') in argv and '--check' not in argv and rank==0:return 'controller'
 if str(mgmt/'run_node.py') in argv and argument(argv,'--node-rank')==str(rank) and not any(k in argv for k in ('--probe','--stop-owned')):return 'supervisor'
 if str(OUT/'queue_node.py') in argv and argument(argv,'--action')=='launch' and argument(argv,'--stage')==stage and rank==0:return 'launcher'
 if str(ROOT/'releases/p2_feature_v1/scripts/p2_listen.py') in argv and argument(argv,'--config')==str(job['config']) and argument(argv,'--arm')==stage:
  return 'torchrun' if 'torch.distributed.run' in argv else 'worker'
 return None

def owned(stage,rank):
 # /proc is host-wide and may have >700k entries; this cgroup contains only this container.
 pids=set(int(p) for p in Path('/sys/fs/cgroup/cgroup.procs').read_text().split())
 state=optional(JOBS[stage]['management']/f'node{rank}.json')
 pids.update(int(state[k]) for k in ('torchrun_pid','supervisor_pid') if state.get(k))
 if rank==0:
  controller=optional(JOBS[stage]['management']/'status.json').get('controller_pid')
  if controller:pids.add(int(controller))
 results=[]
 for pid in pids:
  if pid==os.getpid():continue
  argv,birth=tokens(pid);kind=classify(argv,stage,rank)
  if kind:
   try:pgid=os.getpgid(pid)
   except ProcessLookupError:continue
   results.append(dict(pid=pid,birth=birth,kind=kind,pgid=pgid))
 return results

def signal_owned(item,stage,rank,sig):
 argv,birth=tokens(item['pid'])
 if birth!=item['birth'] or classify(argv,stage,rank)!=item['kind']:return
 try:
  # All launcher descendants inherit this process group. Recheck a live allowlisted member.
  if item['kind'] in ('torchrun','worker') and item['pgid']!=os.getpgrp():
   if os.getpgid(item['pid'])==item['pgid']:os.killpg(item['pgid'],sig)
  else:os.kill(item['pid'],sig)
 except ProcessLookupError:pass

def lock_free(path):
 with Path(path).open('a') as f:
  try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:return False
 return True

def snapshot(stage,rank):
 mgmt=JOBS[stage]['management']
 memory=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True,timeout=10)
 apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True,timeout=10).strip()
 rows=[list(map(int,l.split(','))) for l in memory.strip().splitlines()]
 paths=[mgmt/f'node{rank}.lock']+([mgmt/'controller.lock',mgmt/'queue_launch.lock'] if rank==0 else [])
 return dict(rank=rank,host=socket.gethostname(),owned=owned(stage,rank),node_state=optional(mgmt/f'node{rank}.json'),
  locks_free=all(lock_free(p) for p in paths),gpus=rows,idle=not apps and len(rows)==8 and all(m<1024 for _,m in rows))

def stop(stage,rank):
 initial=owned(stage,rank)
 for item in sorted(initial,key=lambda x:x['kind'] not in ('controller','launcher')):signal_owned(item,stage,rank,signal.SIGTERM)
 deadline=time.monotonic()+15
 while owned(stage,rank) and time.monotonic()<deadline:time.sleep(.5)
 for item in owned(stage,rank):signal_owned(item,stage,rank,signal.SIGKILL)
 deadline=time.monotonic()+10
 while owned(stage,rank) and time.monotonic()<deadline:time.sleep(.5)
 result=snapshot(stage,rank)
 if result['owned'] or not result['locks_free']:raise RuntimeError('Owned processes or locks remain after stop')
 return dict(stopped=initial,**result)

def launch(stage,rank,resume,attempt):
 if rank!=0 or not attempt:raise ValueError('Launch belongs on node0 with attempt id')
 job=JOBS[stage]
 if resume and Path(resume).resolve().parent!=job['run'].resolve():raise ValueError('Cross-run resume rejected')
 with (job['management']/'queue_launch.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  state=optional(QUEUE/'state.json');recovery=state.get('recovery') or {}
  if state.get('stage')!=stage or recovery.get('attempt')!=attempt or recovery.get('phase')!='launching':
   raise RuntimeError('Stale launch attempt rejected')
  if stopped(stage):raise RuntimeError('STOP requested')
  others=owned(stage,rank)
  if others:raise RuntimeError('Refuse duplicate launch: '+str(others))
  cmd=[str(PYTHON),str(job['management']/'controller.py'),'--detach']
  if resume:cmd+=['--resume',str(resume)]
  result=subprocess.run(cmd,capture_output=True,text=True,timeout=35,close_fds=True)
  if result.returncode:raise RuntimeError(result.stderr+result.stdout)
  value=dict(attempt=attempt,stage=stage,launched_at=now(),controller=json.loads(result.stdout),resume=str(resume) if resume else None)
  write(QUEUE/'launches'/f'{attempt}.json',value)
  return value

def main():
 p=argparse.ArgumentParser();p.add_argument('--node-rank',type=int,choices=range(3),required=True)
 p.add_argument('--stage',choices=JOBS,required=True);p.add_argument('--action',choices=('status','stop','launch'),required=True)
 p.add_argument('--resume');p.add_argument('--attempt');a=p.parse_args()
 if socket.gethostname()!=HOSTNAMES[a.node_rank]:raise RuntimeError('Node/hostname mismatch')
 if a.action=='status':value=snapshot(a.stage,a.node_rank)
 elif a.action=='stop':value=stop(a.stage,a.node_rank)
 else:value=launch(a.stage,a.node_rank,a.resume,a.attempt)
 print(json.dumps(value),flush=True)
if __name__=='__main__':main()
