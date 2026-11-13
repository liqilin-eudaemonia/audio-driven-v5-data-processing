import fcntl,json,os,signal,socket,subprocess,sys,time,traceback
from pathlib import Path
from run import WORK,ROOT,read,save,verify

def main():
 verify();assert '172.29.252.37' in subprocess.check_output(['hostname','-I'],text=True).split()
 with (WORK/'controller.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  save(WORK/'LAUNCH.json',dict(at=time.time(),pid=os.getpid(),host=socket.gethostname(),gpus=[0,1,2],new_outputs=600,reused_outputs=1400))
  lines=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True).splitlines()
  memory={int(s.split(',')[0]):float(s.split(',')[1]) for s in lines};assert all(memory[g]<1024 for g in range(3)),memory
  jobs=[]
  signal.signal(signal.SIGTERM,lambda *_:sys.exit(143))
  try:
   for slot in range(3):
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(slot),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false')
    with (WORK/f'worker_{slot}.log').open('ab',buffering=0) as log:
     p=subprocess.Popen([sys.executable,'-u',str(WORK/'run.py'),'worker','--slot',str(slot)],env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
    jobs.append(p);save(WORK/f'process_{slot}.json',dict(pid=p.pid,slot=slot,at=time.time()))
   while any(p.poll() is None for p in jobs):
    if any(p.poll() not in (None,0) for p in jobs):raise RuntimeError('Worker failed; inspect worker log')
    save(WORK/'status.json',dict(at=time.time(),stage='planning_generation',returncodes=[p.poll() for p in jobs],progress=[read(WORK/f'progress_{s}.json') for s in range(3) if (WORK/f'progress_{s}.json').exists()]));time.sleep(5)
   assert all(p.returncode==0 for p in jobs)
   subprocess.run([sys.executable,str(WORK/'run.py'),'report'],check=True)
   save(WORK/'status.json',dict(at=time.time(),stage='generation_complete_review_pending',outputs=2000,new_outputs=600))
  finally:
   for p in jobs:
    if p.poll() is None:
     try:os.killpg(p.pid,signal.SIGTERM)
     except ProcessLookupError:pass
if __name__=='__main__':
 try:main()
 except BaseException:
  save(WORK/'FAILURE_controller.json',dict(at=time.time(),traceback=traceback.format_exc()));raise
