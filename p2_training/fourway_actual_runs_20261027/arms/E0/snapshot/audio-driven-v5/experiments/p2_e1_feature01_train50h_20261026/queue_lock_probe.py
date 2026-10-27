"""Finite CPU-only probe of actual shared-filesystem flock exclusion across hosts."""
import fcntl,json,sys,time,os
from pathlib import Path
folder=Path(sys.argv[1]);rank=int(sys.argv[2]);folder.mkdir(parents=True,exist_ok=True)
(folder/f'ready{rank}').touch();deadline=time.monotonic()+30
while not (folder/'go').exists():
 if time.monotonic()>deadline:raise TimeoutError('Barrier timed out')
 time.sleep(.05)
with (folder/'lock').open('a') as lock:
 try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);leader=True
 except BlockingIOError:leader=False
 (folder/f'node{rank}.json').write_text(json.dumps(dict(rank=rank,leader=leader,pid=os.getpid())))
 if leader:time.sleep(4)
