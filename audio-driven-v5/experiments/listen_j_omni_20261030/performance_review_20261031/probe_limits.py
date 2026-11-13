import sys,json,concurrent.futures
from pathlib import Path
W=Path(__file__).resolve().parent.parent;sys.path.insert(0,str(W/'nod_cpu_share'));from probe import remote
code='''import json,os,fcntl,socket
from pathlib import Path
w=Path(WORK);c=json.loads((w/'config.json').read_text());resource=Path(c['nodes'][NODE]['resource_work'])
def busy(root,n):
 count=0;present=0
 for i in range(n):
  p=root/'locks'/f'decode-{socket.gethostname()}-{i}'
  if not p.exists():continue
  present+=1
  with p.open('r') as f:
   try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
   except BlockingIOError:count+=1
   else:fcntl.flock(f,fcntl.LOCK_UN)
 return dict(busy=count,present=present,limit=n)
print(json.dumps(dict(node=NODE,affinity=len(os.sched_getaffinity(0)),cpu_stat=Path('/sys/fs/cgroup/cpu.stat').read_text(),cpu_pressure=Path('/sys/fs/cgroup/cpu.pressure').read_text(),io_pressure=Path('/sys/fs/cgroup/io.pressure').read_text(),memory_pressure=Path('/sys/fs/cgroup/memory.pressure').read_text(),nod_slots=busy(resource/'corrected_nod_slots',16),decode_slots=busy(resource/'resources',c['decode_slots']))))
'''
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
 fs={n:pool.submit(remote,n,'WORK='+repr(str(W))+'; NODE='+repr(n)+'\n'+code) for n in ['node17','node8','node4','node12']};r={n:f.result() for n,f in fs.items()}
(W/'performance_review_20261031/limits.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
