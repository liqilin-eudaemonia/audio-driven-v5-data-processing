"""Read-only allocation check; never terminate competing jobs."""
import json,subprocess,concurrent.futures
from listen_cluster import transport

def idle(cluster):
 def one(n):
  code="""import subprocess,json
rows=[x.split(',')for x in subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader,nounits'],text=True).splitlines()]
selected={u.strip()for i,u in rows if int(i)in MASK}
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader,nounits'],text=True)
print(json.dumps([x for x in apps.splitlines()if any(u in x for u in selected)]))
""".replace('MASK',repr(n['gpus']))
  r=subprocess.run(transport(cluster,n,[cluster['python'],'-c',code]),capture_output=True,text=True,timeout=35,check=True)
  return dict(node=n['name'],compute_jobs=json.loads(r.stdout))
 with concurrent.futures.ThreadPoolExecutor(4)as pool:values=list(pool.map(one,cluster['nodes']))
 return dict(idle=all(not x['compute_jobs']for x in values),nodes=values)
