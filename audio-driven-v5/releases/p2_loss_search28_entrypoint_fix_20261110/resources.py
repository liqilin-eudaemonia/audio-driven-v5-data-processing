"""Read-only allocation check; never terminate competing jobs."""
import json,subprocess,concurrent.futures,time
from listen_cluster import transport

def idle(cluster):
 def one(n):
  code="""import subprocess,json
rows=[x.split(',')for x in subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader,nounits'],text=True).splitlines()]
selected={u.strip()for i,u in rows if int(i)in MASK}
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader,nounits'],text=True)
print(json.dumps([x for x in apps.splitlines()if any(u in x for u in selected)]))
""".replace('MASK',repr(n['gpus']))
  for attempt in range(8):
   try:
    r=subprocess.run(transport(cluster,n,[cluster['python'],'-c',code]),capture_output=True,text=True,timeout=35,check=True)
    break
   except (subprocess.TimeoutExpired,subprocess.CalledProcessError)as exc:
    if isinstance(exc,subprocess.CalledProcessError)and exc.returncode!=255:raise
    if attempt==7:raise
    # Only repeat read-only NVML observations, never relaunch a worker.
    time.sleep(3)
  return dict(node=n['name'],compute_jobs=json.loads(r.stdout))
 with concurrent.futures.ThreadPoolExecutor(4)as pool:values=list(pool.map(one,cluster['nodes']))
 return dict(idle=all(not x['compute_jobs']for x in values),nodes=values)
