from pathlib import Path
import sys,time,os,fcntl,subprocess
W=Path(__file__).resolve().parent;sys.path.insert(0,str(W));import guard as G
C=G.C
with(W/'supervisor.lock').open('a')as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);G.write('SUPERVISOR.json',G.record(os.getpid()))
 while not(W/'STOP').exists():
  spec=C.read(W/'config.json')
  if C.read(Path(spec['training_output'])/'status.json').get('complete'):break
  r=C.read(W/'GUARD.json')if(W/'GUARD.json').exists()else None
  if not r or not C.alive(r):
   if any(C.sha(W/f)!=h for f,h in C.read(W/'SOURCE.json').items()):G.event('source_changed_restart_refused');break
   with(W/'guard.log').open('a')as f:p=subprocess.Popen([C.PY,'-u',str(W/'guard.py')],stdout=f,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1'))
   G.event('guard_started',pid=p.pid)
  time.sleep(5)
G.write('SUPERVISOR_EXITED.json',dict(at=time.time()))
