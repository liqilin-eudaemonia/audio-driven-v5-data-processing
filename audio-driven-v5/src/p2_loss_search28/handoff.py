"""Save current noEvent at6000, verify allocation release, then start search."""
from pathlib import Path
import time,sys,fcntl,json
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from p2_listen.contracts import read,atomic_json
from .config import WORK,source_identity
OLD=ROOT/'experiments/p2_ltx_all_lora_noevent28_batch28_20261109'
def main():
 from .queue import run
 from .resources import idle
 if read(WORK/'PREFLIGHT_ACCEPTED.json')['source']!=source_identity():raise ValueError('Preflight source changed before arming')
 if not read(WORK/'TESTS_ACCEPTED.json')['success']:raise ValueError('Tests not accepted')
 with(WORK/'handoff.lock').open('a')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  st=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()
  atomic_json(WORK/'HANDOFF_CONTROLLER.json',dict(pid=__import__('os').getpid(),start_ticks=st[19],at=time.time()))
  while not(WORK/'STOP').exists():
   status=read(OLD/'status.json');step=status['step']
   if (OLD/'STOP').exists() and step<6000:raise RuntimeError('Predecessor stopped before6000; preserve user stop')
   if step>=6000:
    # STOP is observed before next update. Native periodic save6000 completes
    # first. A late launch may stop at a later step; never claim exact6000 then.
    (OLD/'STOP').touch(exist_ok=True)
    atomic_json(WORK/'HANDOFF_STATE.json',dict(stage='stop_requested',observed_step=step,at=time.time()))
    break
   time.sleep(.25)
  if(WORK/'STOP').exists():return
  cl=read(OLD/'cluster.json')
  while not(WORK/'STOP').exists():
   status=read(OLD/'status.json');step=status['step'];marker=OLD/f'step_{step:06d}.complete.json'
   if marker.exists():
    snap=idle(cl)
    if snap['idle']:
     # Check complete payload and full Adam/RNG via established native reader.
     from p2_whisper_av.bootstrap import activate_ltx
     activate_ltx()
     from p2_ltx_all_lora_noevent.checkpoint import checked
     v=checked(OLD/f'step_{step:06d}.pt')
     if len(v['rank_rng'])!=28 or v['identity']['experiment']!=read(OLD/'config.json'):raise ValueError('Invalid predecessor save')
     atomic_json(WORK/'HANDOFF_ACCEPTED.json',dict(success=True,step=step,source=str(OLD/f'step_{step:06d}.pt'),world=28,resources=snap,at=time.time()))
     run();return
   time.sleep(5)
if __name__=='__main__':main()
