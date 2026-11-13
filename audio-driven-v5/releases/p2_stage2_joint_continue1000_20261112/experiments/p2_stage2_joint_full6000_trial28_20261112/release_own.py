"""Reap only orphan GPU engines carrying this trial's exact private scope."""
from pathlib import Path
import concurrent.futures,subprocess,sys,time
W=Path(__file__).resolve().parent;sys.path.insert(0,str(W))
import runtime28 as R
from p2_listen.contracts import read,atomic_json

def reap():
 from listen_cluster import transport
 cl=read(R.RUN/'cluster.json')
 code='''from pathlib import Path
import json,os,signal,subprocess,time
allowed=set(ALLOCATION);scope=TRIAL_SCOPE;removed=[]
for line in subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).splitlines():
 try:
  pid=int(line.strip());p=Path('/proc')/str(pid);env=dict(v.split('=',1)for v in(p/'environ').read_text().split(chr(0))if '='in v)
  if scope not in env.get('PYTHONPATH','').split(os.pathsep):continue
  visible=env.get('CUDA_VISIBLE_DEVICES','').split(',')
  if len(visible)!=1 or int(visible[0])not in allowed:continue
  status=(p/'status').read_text();parent=int(next(v.split()[1]for v in status.splitlines()if v.startswith('PPid:')))
  if parent!=1 or 'VLLM::EngineCore'not in(p/'cmdline').read_text():continue
  os.kill(pid,signal.SIGTERM)
  for _ in range(20):
   if not p.exists()or(p/'stat').read_text().split()[2]=='Z':break
   time.sleep(.25)
  if p.exists()and(p/'stat').read_text().split()[2]!='Z':os.kill(pid,signal.SIGKILL)
  removed.append(dict(pid=pid,physical_gpu=int(visible[0])))
 except (FileNotFoundError,ProcessLookupError):pass
print(json.dumps(removed))'''
 def one(n):
  program=code.replace('ALLOCATION',repr(n['gpus'])).replace('TRIAL_SCOPE',repr(str(W/'isolation')))
  for attempt in range(3):
   result=subprocess.run(transport(cl,n,[cl['python'],'-c',program]),text=True,capture_output=True)
   if result.returncode==0:return dict(node=n['name'],removed=__import__('json').loads(result.stdout))
   if result.returncode!=255:raise RuntimeError('Own-orphan cleanup failed: '+result.stderr)
   time.sleep(2)
  raise RuntimeError('Cannot reach node for own-orphan cleanup')
 results=list(concurrent.futures.ThreadPoolExecutor(4).map(one,cl['nodes']))
 atomic_json(W/'resource_release'/f'{time.time_ns()}.json',dict(at=time.time(),nodes=results,scope='exact trial PYTHONPATH, selected GPU, EngineCore, PPid1 only'))
 return results
