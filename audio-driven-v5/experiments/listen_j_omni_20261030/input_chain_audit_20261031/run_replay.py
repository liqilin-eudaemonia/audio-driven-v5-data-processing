import sys,json,concurrent.futures
from pathlib import Path
W=Path(__file__).resolve().parent.parent;sys.path.insert(0,str(W/'nod_cpu_share'));from probe import remote
# At most two CPU media readers; no inference or source-tree scan.
def run(n):
 cmd=['/private/software/conda/envs/vllm/bin/python',str(W/'input_chain_audit_20261031/replay.py'),n]
 code=f"import subprocess,json,os;p=subprocess.run({cmd!r},capture_output=True,text=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1'));print(json.dumps(dict(code=p.returncode,out=p.stdout[-9000:],err=p.stderr[-2500:])))"
 return remote(n,code,timeout=240)
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
 fs={n:pool.submit(run,n) for n in (sys.argv[1:] or ['node17','node8','node4','node12'])}
 for n,f in fs.items():print(n,json.dumps(f.result()),flush=True)
