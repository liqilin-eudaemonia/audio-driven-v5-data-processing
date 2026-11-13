"""CPU simulation: one port-bind failure must not terminate healthy peers."""
from pathlib import Path
import collections,sys,tempfile,time
W=Path(__file__).resolve().parent;sys.path.insert(0,str(W))
import eval28 as E,parallel_events as P

class Child:
 def __init__(self,code):self.returncode=code;self.terminated=False
 def poll(self):return self.returncode
 def wait(self,timeout=None):return self.returncode
 def terminate(self):self.terminated=True
 def kill(self):self.terminated=True

def check(module,kind):
 with tempfile.TemporaryDirectory(dir=W)as tmp:
  out=Path(tmp);c=E.read(E.R.RUN/'config.json');policy=E.read(E.folder(100)/'event_detector_config.json')
  E.atomic_json(out/'event_detector_config.json',policy)
  E.atomic_json(out/'contract.json',dict(policy=policy))
  for rank in range(24,28):E.atomic_json(out/'inference_shards'/f'{rank:02d}'/'contract.json',dict(policy=policy))
  counts=collections.Counter();children=[]
  def popen(cmd,env,stdout,stderr):
   rank=int(cmd[cmd.index('--rank')+1])if kind=='plan'else int(Path(cmd[cmd.index('--work')+1]).name)
   counts[rank]+=1;failed=rank==25 and counts[rank]==1
   if failed:stdout.write(b'EADDRINUSE: address already in use\n')
   child=Child(1 if failed else 0);children.append(child);return child
  old_popen,old_output,old_sleep,old_folder=E.subprocess.Popen,E.subprocess.check_output,E.time.sleep,E.folder
  try:
   E.subprocess.Popen=popen
   E.subprocess.check_output=lambda cmd,**kwargs:E.read(E.R.RUN/'cluster.json')['nodes'][3]['host']if cmd[0]=='hostname'else'GPU-test-'+str(cmd[2])
   E.time.sleep=lambda n:None;E.folder=lambda step:out
   if kind=='plan':E.node(c,100,'plans',3)
   else:P.node(out,3)
  finally:E.subprocess.Popen=old_popen;E.subprocess.check_output=old_output;E.time.sleep=old_sleep;E.folder=old_folder
  assert dict(counts)=={24:1,25:2,26:1,27:1},counts
  assert not any(p.terminated for p in children)
  return dict(counts)

if __name__=='__main__':
 result=dict(success=True,cpu_simulation_only=True,plan=check(E,'plan'),events=check(P,'event'),at=time.time())
 E.atomic_json(W/'PORT_RETRY_ACCEPTED.json',result);print(result)
