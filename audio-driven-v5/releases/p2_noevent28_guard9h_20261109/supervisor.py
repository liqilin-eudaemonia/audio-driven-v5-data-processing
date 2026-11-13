"""Restart only a verified-dead observer; training ownership remains with guardian."""
from pathlib import Path
import sys,os,time,json,fcntl,subprocess,hashlib
WORK=Path(__file__).resolve().parent;sys.path.insert(0,str(WORK))
import common as C
with(WORK/'supervisor.lock').open('a')as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 C.atomic_json(WORK/'SUPERVISOR.json',C.spawn_record(type('P',(),{'pid':os.getpid()})()))
 while time.time()<C.read(WORK/'protocol.json')['deadline']and not(WORK/'STOP').exists():
  record=C.read(WORK/'GUARDIAN.json')if(WORK/'GUARDIAN.json').exists()else None
  if not record or not C.alive(record):
   hashes=C.read(WORK/'OBSERVER_SOURCE.json')
   if any(hashlib.sha256((WORK/f).read_bytes()).hexdigest()!=h for f,h in hashes.items()):
    C.atomic_json(WORK/'NEEDS_AGENT.json',dict(at=time.time(),reason='Observer source changed; refuse unreviewed restart'));break
   with(WORK/'guardian.log').open('a')as f:p=subprocess.Popen([sys.executable,'-u',str(WORK/'guardian.py')],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1'))
   C.event('observer_restarted',pid=p.pid)
  time.sleep(30)
 C.atomic_json(WORK/'SUPERVISOR_EXITED.json',dict(at=time.time(),training_stopped=False))
