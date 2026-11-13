import concurrent.futures,json,sys
from probe import remote,W
label=sys.argv[1];seconds=int(sys.argv[2]);dest=W/'nod_cpu_share'/label
script=str(W/'visual_quality_rollout/observe.py')
def run(n):
 code=f"import subprocess,json;p=subprocess.run({['python3',script,n,str(dest),str(seconds)]!r},capture_output=True,text=True);print(json.dumps(dict(code=p.returncode,out=p.stdout,err=p.stderr[-1000:])))"
 return remote(n,code,timeout=seconds+45)
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as p:
 fs={n:p.submit(run,n) for n in ['node17','node8','node4','node12']}
 for n,f in fs.items():
  d=f.result();print(n,json.dumps(d),flush=True)
