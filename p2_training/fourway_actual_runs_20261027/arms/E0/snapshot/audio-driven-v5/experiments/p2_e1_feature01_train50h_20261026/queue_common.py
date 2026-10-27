"""Small, CPU-only contracts for the authorized two-stage training queue."""
import json,os,socket,subprocess,shlex,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from control import OUT,ROOT,REFERENCE,TRAIN,PYTHON,BASELINE,E1,HOSTS,HOSTNAMES,KEY,BUDGET,read,write,sha,now
QUEUE=OUT/'queue'
JOBS={'E0':dict(management=REFERENCE,run=BASELINE,config=TRAIN/'config.json'),
      'E1':dict(management=OUT,run=E1,config=OUT/'config.json')}

def optional(path,default=None):
 try:return read(path)
 except FileNotFoundError:return {} if default is None else default

def metric_progress(run):
 path=Path(run)/'metrics.jsonl'
 try:
  with path.open('rb') as f:
   f.seek(0,2);f.seek(max(0,f.tell()-65536));lines=f.read().decode().splitlines()
  for line in reversed(lines):
   try:
    row=json.loads(line);return dict(step=int(row['step']),time=float(row['time']))
   except (ValueError,KeyError):continue
 except FileNotFoundError:pass
 return dict(step=0,time=0.)

def remote(rank,action,stage,resume=None,attempt=None):
 cmd=[str(PYTHON),str(OUT/'queue_node.py'),'--node-rank',str(rank),'--action',action,'--stage',stage]
 if resume:cmd+=['--resume',str(resume)]
 if attempt:cmd+=['--attempt',attempt]
 if socket.gethostname()!=HOSTNAMES[rank]:
  cmd=['ssh','-i',KEY,'-o','UserKnownHostsFile='+str(OUT/'queue_known_hosts'),'-o','StrictHostKeyChecking=yes','-o','BatchMode=yes','-o','ConnectTimeout=10','-o','ServerAliveInterval=10','-o','ServerAliveCountMax=2','root@'+HOSTS[rank],shlex.join(cmd)]
 result=subprocess.run(cmd,capture_output=True,text=True,timeout=90,close_fds=True)
 if result.returncode:raise RuntimeError(f'{stage} node{rank} {action}: {result.stderr[-3000:]} {result.stdout[-1000:]}')
 return json.loads(result.stdout)

def cluster(action,stage):
 with ThreadPoolExecutor(max_workers=3) as pool:
  futures=[pool.submit(remote,r,action,stage) for r in range(3)]
  results=[];errors=[]
  for f in futures:
   try:results.append(f.result())
   except Exception as error:errors.append(str(error))
  if errors:raise RuntimeError('; '.join(errors))
  return results

def stopped(stage):
 job=JOBS[stage]
 return any(p.exists() for p in (QUEUE/'STOP',job['management']/'STOP',job['run']/'STOP'))
