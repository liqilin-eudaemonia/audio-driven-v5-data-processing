"""Populate exact RGB/raw and ungated caches while English generation runs."""
import os,sys,time,subprocess,traceback,concurrent.futures
from pathlib import Path
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1];sys.path[:0]=[str(ROOT/'src'),str(OUT)]
from generate_cluster import remote,PY
from p2_glare_eval.common import read,sha,atomic_json,artifact_valid
from p2_glare_rgb.detector import Engine
from p2_glare_unfiltered.rules import process
WORK=OUT/'reactions';DP=ROOT/'.venv-p2-glare-detector/bin/python'
def worker(rank):
 jobs=read(OUT/'jobs.json')[rank::32];engine=Engine();done={};results=[]
 while len(done)<len(jobs):
  progress=False
  for j in jobs:
   if j['output'] in done or not (Path(j['output'])/'complete.json').exists():continue
   assert artifact_valid(j['output'],j['identity'])
   video=Path(j['output'])/'generated.mp4';h=sha(video);folder=WORK/'rgb'/h
   engine.detect(video,folder,h);quality=process(video,folder,WORK/'unfiltered'/h,h)
   done[j['output']]=True;results.append(dict(video=str(video),sha256=h,strict_valid=quality['strict_valid']))
   atomic_json(WORK/'stream_workers'/f'{rank:02d}.json',dict(finished=False,results=results));progress=True
  if not progress:
   if read(OUT/'status.json')['phase']=='failed':raise RuntimeError('Generation failed')
   time.sleep(3)
 atomic_json(WORK/'stream_workers'/f'{rank:02d}.json',dict(finished=True,results=results))
def node(rank):
 children=[];env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2','TF_ENABLE_ONEDNN_OPTS':'0','TF_CPP_MIN_LOG_LEVEL':'2'}
 for local in range(16):
  i=rank*16+local
  with (WORK/'logs'/f'stream-{i:02d}.log').open('a') as log:children.append(subprocess.Popen([str(DP),str(__file__),'worker',str(i)],stdout=log,stderr=subprocess.STDOUT,env=env))
  time.sleep(.2)
 codes=[p.wait() for p in children];atomic_json(WORK/f'stream-node-{rank}.json',dict(codes=codes,finished=True))
 if any(codes):raise RuntimeError(str(codes))
if __name__=='__main__':
 for d in ['rgb','logs','unfiltered']:(WORK/d).mkdir(parents=True,exist_ok=True)
 cmd=sys.argv[1]
 if cmd=='launch':
  with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
   for value in pool.map(lambda r:remote(r,[PY,__file__,'detach',r]),range(2)):print(value)
 elif cmd=='detach':
  rank=int(sys.argv[2])
  with (WORK/f'stream-node-{rank}.log').open('a') as log:p=subprocess.Popen([str(PY),str(__file__),'node',str(rank)],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,close_fds=True)
  print(p.pid)
 elif cmd=='node':node(int(sys.argv[2]))
 elif cmd=='worker':worker(int(sys.argv[2]))
