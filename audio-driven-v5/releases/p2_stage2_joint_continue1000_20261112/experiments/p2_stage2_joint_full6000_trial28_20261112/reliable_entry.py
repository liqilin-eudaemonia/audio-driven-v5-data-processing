"""Keep the native launcher/guard; fix its transient SSH retry race only."""
from pathlib import Path
import inspect,runpy,sys
W=Path(__file__).resolve().parent;ROOT=W.parents[1];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
import listen_cluster as N

def terminal_failure(processes,retries,out,nodes):
 for i,p in enumerate(processes):
  if p.poll()in(None,0):continue
  if p.poll()==255 and retries[i]<6:
   content=(Path(out)/(nodes[i]['name']+'.log')).read_text()
   if len(content)<3000 and 'kex_exchange_identification'in content:continue
  return True
 return False

if not getattr(N,'_joint_retry_installed',False):
 source=inspect.getsource(N.launch)
 old='if any(p.poll() not in (None, 0) for p in processes):'
 if source.count(old)!=1:raise ValueError('Native launch retry guard changed')
 source=source.replace(old,"if terminal_failure(processes,retries,out,cfg['nodes']):")
 namespace=dict(N.__dict__,terminal_failure=terminal_failure)
 exec(compile(source,str(Path(__file__))+'::native_retry','exec'),namespace);N.launch=namespace['launch'];N._joint_retry_installed=True

if __name__=='__main__':
 target=sys.argv[1];sys.argv=sys.argv[1:];runpy.run_path(target,run_name='__main__')
