"""Fill idle authorized GPUs with ready SFT exports while rendering drains."""
import argparse,concurrent.futures,copy,fcntl,json,os,subprocess,sys,threading,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT.parent/'OmniResponse')]
from p2_testsets_eval.protocol import read,save,sha,PYTHON
from p2_response200_fourway import link
from listen_cluster import transport


def run(work):
 child=work/'sft_event';aux=work/'idle_sft_decode';aux.mkdir(exist_ok=True)
 with (aux/'controller.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  cp=read(child/'protocol.json');items=[x for rank in range(cp['workers']) for x in read(child/'shards'/f'{rank:02d}.json')]
  if len(items)!=200 or len({x['key'] for x in items})!=200:raise ValueError('Need fixed200')
  cluster=read(work/'cluster.json');jobs=[]
  def query(node):
   try:
    report=subprocess.check_output(transport(cluster,node,['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits']),text=True,timeout=25)
    values={int(s[0]):[float(x) for x in s[1:]] for line in report.splitlines() if (s:=line.split(','))}
    return [(node,g) for g in node['gpus'] if values[g][0]<2048 and values[g][1]<10]
   except Exception:return []
  with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
   for result in pool.map(query,cluster['nodes']):jobs+=result
  if not jobs:raise ValueError('No idle authorized GPUs')
  manifest=aux/'items.json';save(manifest,items)
  ap=copy.deepcopy(cp);ap.update(shared_item_queue=True,shared_items_manifest=str(manifest),shared_items_sha256=sha(manifest),workers=len(jobs))
  save(aux/'protocol.json',ap)
  for name in ['generated','latents','sample_claims']:
   (child/name).mkdir(exist_ok=True);link(child/name,aux/name)
  for rank in range(len(jobs)):save(aux/'shards'/f'{rank:02d}.json',items)
  save(aux/'JOBS.json',[dict(rank=i,node=n['name'],gpu=g) for i,(n,g) in enumerate(jobs)])
  save(aux/'status.json',dict(stage='decoding_ready_sft_latents',gpus=len(jobs),at=time.time()))
  sem={n['name']:threading.Semaphore(2) for n in cluster['nodes']}
  def one(job):
   rank,(node,gpu)=job
   cmd=['env',f'CUDA_VISIBLE_DEVICES={gpu}','OMP_NUM_THREADS=4','OPENBLAS_NUM_THREADS=2','PYTHONUNBUFFERED=1',str(PYTHON),str(ROOT/'scripts/p2_testsets_eval.py'),'worker','--work',str(aux),'--stage','decode','--rank',str(rank)]
   for attempt in range(1,5):
    path=aux/'logs'/f'rank{rank:02d}_{attempt}.log';path.parent.mkdir(exist_ok=True)
    with path.open('w') as log:
     with sem[node['name']]:
      process=subprocess.Popen(transport(cluster,node,cmd),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT);time.sleep(.1)
     save(aux/'processes'/f'{rank:02d}.json',dict(pid=process.pid,node=node['name'],gpu=gpu,log=str(path)))
     code=process.wait()
    if code==0:return dict(rank=rank,node=node['name'],gpu=gpu,success=True)
    if attempt==4:raise RuntimeError('Gap decoder failed '+str(path))
    time.sleep(2)
  with concurrent.futures.ThreadPoolExecutor(max_workers=len(jobs)) as pool:results=list(pool.map(one,enumerate(jobs)))
  save(aux/'status.json',dict(stage='complete',gpus=len(jobs),results=results,at=time.time()))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True);run(p.parse_args().work)
