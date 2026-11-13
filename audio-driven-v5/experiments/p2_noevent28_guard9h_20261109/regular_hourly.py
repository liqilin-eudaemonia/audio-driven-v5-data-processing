"""Restore the prior read-only hourly observation after the bounded repair window."""
import time,os,fcntl,traceback
import common as C
import guardian as G
W=C.WORK

def main():
 assert time.time()>=C.read(W/'protocol.json')['deadline']
 with(W/'regular_hourly.lock').open('a')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  C.atomic_json(W/'REGULAR_HOURLY_CONTROLLER.json',C.spawn_record(type('P',(),{'pid':os.getpid()})(),read_only=True,interval_seconds=3600))
  speed=G.speed_baseline()
  while not(W/'REGULAR_STOP').exists():
   try:
    health=C.health();snap=C.probe();sp=speed();flags=list(health['flags'])
    if not snap['healthy28']:flags.append('worker_or_gpu_state')
    if sp['ratio']is not None and sp['ratio']>1.5:flags.append('shared_compute_slowdown')
    r=dict(at=time.time(),health=health,cluster=snap,speed=sp,contract=C.contract(True),flags=flags,read_only=True,repair_window_finished=True)
   except Exception:r=dict(at=time.time(),read_only=True,observation_error=traceback.format_exc())
   C.atomic_json(W/'REGULAR_HOURLY_LATEST.json',r)
   if r.get('health',{}).get('complete'):return
   time.sleep(3600)
if __name__=='__main__':main()
