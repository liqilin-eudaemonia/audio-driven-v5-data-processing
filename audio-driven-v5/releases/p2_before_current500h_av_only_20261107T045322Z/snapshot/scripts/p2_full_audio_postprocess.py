#!/usr/bin/env python3
"""722706-sample audio-only postprocessing and durable successor supervision."""
import argparse,sys,json,traceback,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
def main():
 p=argparse.ArgumentParser();p.add_argument('command',choices=('index','prepare','worker','node','agent','daemon','audit'))
 p.add_argument('--run',type=Path,required=True);p.add_argument('--node-rank',type=int,choices=range(3));p.add_argument('--rank',type=int)
 p.add_argument('--action');p.add_argument('--generation');p.add_argument('--detach',action='store_true');a=p.parse_args();run=a.run.resolve();run.mkdir(parents=True,exist_ok=True)
 try:
  if a.command=='index':
   from listen_audio_full.indexing import freeze_scope
   result=freeze_scope(run)
  elif a.command=='prepare':
   from listen_audio_full.producer import produce
   result=produce(run)
  elif a.command=='worker':
   from listen_audio_full.producer import verify_code
   verify_code(run)
   from listen_audio_full.worker import run_worker
   result=run_worker(run,a.rank)
  else:
   from listen_audio_full.cluster import node_worker,agent,daemon,finalize
   if a.command=='node':result=node_worker(run,a.node_rank,a.generation)
   elif a.command=='agent':result=agent(run,a.node_rank,a.action,a.generation)
   elif a.command=='daemon':result=daemon(run,a.node_rank,a.detach)
   else:result=finalize(run,a.generation or 'manual-'+str(time.time_ns()))
  print(json.dumps(result),flush=True)
 except Exception:
  if a.command!='agent':
   from p2_listen.contracts import atomic_json
   atomic_json(run/'errors'/f'{a.command}-{a.rank if a.rank is not None else a.node_rank}.json',dict(error=traceback.format_exc(),time=time.time()))
  raise
if __name__=='__main__':main()
