"""One isolated subprocess per GPU, all four arms with common clip/seed sharding."""
import sys,os,time,subprocess,json,fcntl
from pathlib import Path
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1];sys.path.insert(0,str(ROOT/'src'))
from p2_listen.contracts import read,atomic_json
PY=ROOT.parent/'Omni-LTX-stage2/.venv/bin/python'
def worker(rank):
 exps=read(OUT/'experiments.json');env={**os.environ,'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2','CUDA_VISIBLE_DEVICES':str(rank%8),'CUBLAS_WORKSPACE_CONFIG':':4096:8','PYTORCH_ALLOC_CONF':'expandable_segments:True'}
 for arm in ['E0','E1','E0F','E1F']:
  work=OUT/arm/'generation'
  while not (work/'generation_protocol.json').exists():time.sleep(3)
  command=[str(PY),str(OUT/'runner.py'),arm,'generate','--work',str(work),'--checkpoint',exps[arm]['checkpoint'],'--shards','24','--shard-index',str(rank),'--device','cuda:0']
  logfile=OUT/'logs'/f'generate-{rank:02d}-{arm}.log';logfile.parent.mkdir(exist_ok=True)
  with logfile.open('a') as log:result=subprocess.run(command,env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
  atomic_json(OUT/'workers'/f'{rank:02d}.json',dict(rank=rank,arm=arm,returncode=result.returncode,updated=time.time()))
 atomic_json(OUT/'workers'/f'{rank:02d}.json',dict(rank=rank,finished=True,updated=time.time()))
def node(rank):
 children=[]
 for local in range(8):
  global_rank=rank*8+local;logfile=OUT/'logs'/f'worker-{global_rank:02d}.log';logfile.parent.mkdir(exist_ok=True)
  with logfile.open('a') as log:children.append(subprocess.Popen([str(PY),str(__file__),'worker',str(global_rank)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,close_fds=True))
  time.sleep(1)
 codes=[p.wait() for p in children];atomic_json(OUT/f'node-{rank}.json',dict(returncodes=codes,finished=True,updated=time.time()))
if __name__=='__main__':
 kind,rank=sys.argv[1],int(sys.argv[2]);OUT.mkdir(exist_ok=True)
 if kind=='worker':worker(rank)
 elif kind=='node':node(rank)
 elif kind=='detach':
  logfile=OUT/f'node-{rank}.log'
  with logfile.open('a') as log:p=subprocess.Popen([str(PY),str(__file__),'node',str(rank)],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
  print(json.dumps({'node':rank,'pid':p.pid}))
