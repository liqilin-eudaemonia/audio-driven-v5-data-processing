import concurrent.futures,json,sys,time
from pathlib import Path
from probe import remote,W
sys.path.insert(0,str(W.parents[1]/'src'))
from listen_benchmark.j_annotation import read,write,locked
nodes=sys.argv[1:]
for n in nodes:assert read(W/'nod_cpu_share'/f'parity-{n}.json')['passed']
with locked(W/'nod_cpu_share/config.lock'):
 c=read(W/'nod_cpu_share/config.json')
 for n in nodes:c['nodes'][n]['workers']=8
 write(W/'nod_cpu_share/config.json',c)
def run(n):
 code=f'''import sys,json,os
from pathlib import Path
w=Path({str(W)!r});sys.path[:0]=[str(w.parents[1]/'scripts'),str(w.parents[1]/'src')]
from listen_benchmark.j_annotation import read,write
from listen_j_annotation_supervisor import spawn,alive
c=read(w/'nod_cpu_share/config.json');local=w/'nod_cpu_share'/{n!r};local.mkdir(exist_ok=True)
old=read(local/'supervisor.json') if (local/'supervisor.json').exists() else None
if alive(old):print(json.dumps(dict(already_live=True,pid=old['pid'])))
else:
 child=spawn(local,'supervisor-launch',[c['python'],str(w.parents[1]/'scripts/listen_j_nod_cpu_share.py'),'supervise','--work',str(w),'--node',{n!r}],dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',GLOG_minloglevel='2'))
 write(local/'launch.json',child);print(json.dumps(child))'''
 return remote(n,code)
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as p:
 fs={n:p.submit(run,n) for n in nodes}
 for n,f in fs.items():print(n,json.dumps(f.result()))
