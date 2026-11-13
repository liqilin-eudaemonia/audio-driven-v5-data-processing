"""Protect the selected training GPUs from confirmed out-of-workspace compute jobs."""
from pathlib import Path
import sys,time,os,json,subprocess,concurrent.futures,fcntl,traceback
W=Path(__file__).resolve().parent;ROOT=W.parents[1];OLD=ROOT/'experiments/p2_noevent28_guard9h_20261109'
sys.path.insert(0,str(OLD));import common as H
C=H.C

def record(pid):
 s=Path('/proc',str(pid),'stat').read_text().rsplit(')',1)[1].split();return dict(pid=pid,start_ticks=s[19],at=time.time())
def write(name,data):H.atomic_json(W/name,data)
def event(kind,**kw):
 r=dict(at=time.time(),kind=kind,**kw)
 with(W/'events.jsonl').open('a')as f:f.write(json.dumps(r)+'\n')
 print(json.dumps(r),flush=True)
def protected(node,cluster,out):
 off=sum(len(n['gpus'])for n in cluster['nodes'][:cluster['nodes'].index(node)]);result=[]
 for rank in range(off,off+len(node['gpus'])):
  p=out/'rank_logs'/f'worker_{rank:02d}.json'
  if p.exists():
   r=C.read(p);result.append(dict(pid=r['pid'],start_ticks=r['start_ticks']))
 return result

def call(node,cluster,payload,action):
 code='import sys,json;sys.path.insert(0,'+repr(str(W))+');import remote;print(json.dumps(remote.'+action+'(json.loads('+repr(json.dumps(payload))+'))))'
 p=subprocess.run(H.transport(cluster,node,[cluster['python'],'-c',code]),capture_output=True,text=True,timeout=20 if action=='scan'else 30)
 if p.returncode:raise RuntimeError(node['name']+' '+p.stderr[-600:])
 return json.loads(p.stdout)

def cycle(enforce=True):
 spec=C.read(W/'config.json');out=Path(spec['training_output']);cluster=C.read(spec['cluster'])
 assert C.sha(spec['cluster'])==spec['cluster_sha256']
 for f,h in C.read(W/'SOURCE.json').items():assert C.sha(W/f)==h
 def one(n):
  x=dict(gpus=n['gpus'],workspace_root=spec['workspace_root'],protected=protected(n,cluster,out))
  try:
   first=call(n,cluster,x,'scan');foreign=[r for r in first['records']if r['owner']=='foreign'];result=dict(node=n['name'],scan=first,termination=None)
   result['foreign_candidates']=foreign
   return result
  except Exception as e:return dict(node=n['name'],observation_error=repr(e),no_unverified_signal=True)
 with concurrent.futures.ThreadPoolExecutor(4)as pool:nodes=list(pool.map(one,cluster['nodes']))
 if enforce and any(n.get('foreign_candidates')for n in nodes):
  time.sleep(2)
  for node,item in zip(cluster['nodes'],nodes):
   if not item.get('foreign_candidates'):continue
   x=dict(gpus=node['gpus'],workspace_root=spec['workspace_root'],protected=protected(node,cluster,out),expected=item['foreign_candidates'])
   try:item['termination']=call(node,cluster,x,'act')
   except Exception as e:item['action_error']=repr(e)
 result=dict(at=time.time(),enforcing=enforce,nodes=nodes,reserved_gpu='node11 GPU4-7')
 write('LATEST.json',result)
 for n in nodes:
  if n.get('termination'):event('foreign_jobs_terminated',node=n['node'],result=n['termination'])
  if n.get('observation_error'):event('node_observation_error',node=n['node'],error=n['observation_error'])
  if n.get('action_error'):event('foreign_termination_error',node=n['node'],error=n['action_error'])
 return result

def main():
 with(W/'guard.lock').open('a')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);write('GUARD.json',record(os.getpid()))
  while not(W/'STOP').exists():
   s=C.read(W/'config.json');status=C.read(Path(s['training_output'])/'status.json')
   if status.get('complete'):event('training_completed_guard_exited');return
   try:cycle()
   except Exception:event('guard_cycle_error',error=traceback.format_exc())
   time.sleep(15)
if __name__=='__main__':main()
