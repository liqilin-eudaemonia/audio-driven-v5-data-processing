import concurrent.futures,json
from probe import W,remote
py=str(W/'nod_cpu_share/runtime_env/bin/python');script=str(W/'nod_cpu_share/parity.py')
def run(n):
 code=f"import subprocess,json;p=subprocess.run({[py,script,n]!r},capture_output=True,text=True);print(json.dumps(dict(code=p.returncode,out=p.stdout[-7000:],err=p.stderr[-1000:])))"
 return remote(n,code,timeout=180)
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as p:
 fs={n:p.submit(run,n) for n in ['node8','node4','node12']}
 for n,f in fs.items():print(n,json.dumps(f.result()),flush=True)
