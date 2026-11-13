"""Explicit unequal node counts, but equal one-sample work per rank."""
from pathlib import Path
import sys,os,argparse,time,subprocess,json,socket
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
def main():
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['node','train']);p.add_argument('--index',type=int);p.add_argument('--config',required=True);p.add_argument('--resume');p.add_argument('--stop-after',type=int,required=True);a=p.parse_args()
 c=json.loads(Path(a.config).read_text());out=Path(c['output']);cl=json.loads((out/'cluster.json').read_text())
 if a.mode=='node':
  n=cl['nodes'][a.index];off=sum(len(x['gpus'])for x in cl['nodes'][:a.index]);children=[];streams=[]
  try:
   for local,gpu in enumerate(n['gpus']):
    rank=off+local;env=dict(os.environ,RANK=str(rank),LOCAL_RANK=str(local),WORLD_SIZE='28',LOCAL_WORLD_SIZE=str(len(n['gpus'])),MASTER_ADDR=cl['nodes'][0]['host'],MASTER_PORT=str(cl['master_port']),P2_TRAIN_MEMORY_GIB='108'if n['name']=='node11'else'134')
    f=(out/'rank_logs'/f'rank_{rank:02d}.log').open('a');streams.append(f)
    cmd=[cl['python'],'-u',str(Path(__file__)),'train','--config',a.config,'--stop-after',str(a.stop_after)]
    if a.resume:cmd+=['--resume',a.resume]
    children.append(subprocess.Popen(cmd,env=env,stdout=f,stderr=subprocess.STDOUT))
   while any(x.poll()is None for x in children):
    if any(x.poll()not in(None,0)for x in children):raise RuntimeError('Search rank failed; inspect rank logs')
    time.sleep(.5)
   if any(x.returncode for x in children):raise RuntimeError('Nonzero rank exit')
  finally:
   for x in children:
    if x.poll()is None:x.terminate()
   for x in children:
    try:x.wait(timeout=15)
    except subprocess.TimeoutExpired:x.kill();x.wait()
   for f in streams:f.close()
 else:
  from p2_whisper_av.bootstrap import activate_ltx
  activate_ltx()
  from p2_loss_search28.config import validate,source_identity
  from p2_listen.contracts import read,atomic_json
  c=validate(c);ready=read(out/'preflight.json')
  if ready['source']!=source_identity():raise ValueError('Reviewed source changed')
  rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK'])
  stat=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()
  atomic_json(out/'rank_logs'/f'worker_{rank:02d}.json',dict(pid=os.getpid(),start_ticks=stat[19],host=socket.gethostname(),rank=rank,world=28,physical_gpu=int(os.environ['CUDA_VISIBLE_DEVICES'].split(',')[local])))
  from p2_loss_search28.training import train
  print(json.dumps(train(c,a.resume,a.stop_after)),flush=True)
if __name__=='__main__':main()
