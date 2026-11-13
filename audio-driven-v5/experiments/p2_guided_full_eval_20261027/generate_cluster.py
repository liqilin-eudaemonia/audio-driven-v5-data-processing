"""Run the unchanged guidance generator on the full paired 24 x 2 x 4 x 2 scope."""
import sys,os,json,subprocess,time,shlex,concurrent.futures
from pathlib import Path
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1];sys.path.insert(0,str(ROOT/'src'))
from p2_listen.contracts import read,atomic_json,sha,digest
from p2_glare_eval.common import job_key,artifact_valid
from p2_guided_prompt.common import load_protocol,prompt_record
PY=ROOT.parent/'Omni-LTX-stage2/.venv/bin/python';ARMS=['E0','E1','E0F','E1F']
HOSTS=['172.29.252.11','172.29.15.91','172.29.225.51'];KEY='/private/ssh_keys/root_dedicated-developjob-lq-dev-machine-1nkpf_ed25519';KH=ROOT/'experiments/p2_e1_feature01_train50h_20261026/queue_known_hosts'

def remote(rank,cmd):
 if rank:cmd=['ssh','-i',KEY,'-o','UserKnownHostsFile='+str(KH),'-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=10','root@'+HOSTS[rank],shlex.join(list(map(str,cmd)))]
 return subprocess.run(list(map(str,cmd)),capture_output=True,text=True,timeout=60,check=True).stdout

def prepare():
 jobs=[];audit=[]
 for kind in ['train','test']:
  old=ROOT/f'experiments/p2_{kind}24_fourway_eval_20261027';base=ROOT/f'experiments/p2_guided_{kind}24_fourway_eval_20261027';work=ROOT/f'experiments/p2_guided_prompt_{kind}24_20261027';p=load_protocol(work);exps=read(old/'experiments.json')
  assert p['observations']==read(old/'observations.json')
  base.mkdir(exist_ok=True);atomic_json(base/'observations.json',p['observations'],immutable=True)
  atomic_json(base/'experiments.json',exps,immutable=True)
  for arm in ARMS:
   e=exps[arm];assert sha(e['checkpoint'])==e['sha256']
   for seed in [137,251]:
    for obs in p['observations']:
     sid=obs['sample_id'];g=prompt_record(work,sid);dest=base/arm/'generation/generated'/job_key(sid,seed)
     ident=dict(protocol_sha256=digest(p),sample_id=sid,sample_seed=seed,variant='guided',guidance_sha256=digest(g),checkpoint_sha256=e['sha256'])
     dest.parent.mkdir(parents=True,exist_ok=True)
     previous=ROOT/'experiments/p2_guided_prompt_train24_20261027/validation_summary.json'
     if kind=='train' and not dest.exists():
      matches=[r for r in read(previous)['rows'] if r['arm']==arm and r['seed']==seed and r['sample_id']==sid and r['variant']=='guided']
      if matches and artifact_valid(matches[0]['output'],ident):dest.symlink_to(matches[0]['output'],target_is_directory=True)
     jobs.append(dict(kind=kind,arm=arm,seed=seed,sample_id=sid,work=str(work),run=e['run'],checkpoint=e['checkpoint'],output=str(dest),identity=ident,restoration=str(old/'source_restoration.json')))
   audit.append(dict(dataset=kind,arm=arm,outputs=48,seeds=[137,251],generation=e['config']['generation'],checkpoint_sha256=e['sha256'],observations_equal=True))
 assert len(jobs)==384
 atomic_json(OUT/'jobs.json',jobs,immutable=True);atomic_json(OUT/'pairing_audit.json',dict(rows=audit,only_intervention='positive Gemma text, exact negative text and all other inference settings unchanged'),immutable=True)
 (OUT/'logs').mkdir(exist_ok=True);(OUT/'workers').mkdir(exist_ok=True)
 return dict(jobs=len(jobs),reusable=sum(artifact_valid(j['output'],j['identity']) for j in jobs))

def worker(rank):
 env={**os.environ,'CUDA_VISIBLE_DEVICES':str(rank%8),'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2','CUBLAS_WORKSPACE_CONFIG':':4096:8','PYTORCH_ALLOC_CONF':'expandable_segments:True'};results=[]
 for n,j in enumerate(read(OUT/'jobs.json')[rank::24]):
  if not artifact_valid(j['output'],j['identity']):
   cmd=[str(PY),str(ROOT/'scripts/p2_guided_prompt.py'),'generate']
   for k in ['work','run','checkpoint','sample_id','output','seed','restoration']:cmd+=['--'+k.replace('_','-'),str(j[k])]
   cmd+=['--device','cuda:0'];start=time.time()
   with (OUT/'logs'/f'generate-{rank:02d}-{n:02d}.log').open('a') as log:r=subprocess.run(cmd,env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
   if r.returncode:raise RuntimeError(f'Generation failed rank={rank}, job={n}, code={r.returncode}')
   assert artifact_valid(j['output'],j['identity'])
  results.append({k:j[k] for k in ['kind','arm','sample_id','seed','output']});atomic_json(OUT/'workers'/f'{rank:02d}.json',dict(finished=False,results=results,updated=time.time()))
 atomic_json(OUT/'workers'/f'{rank:02d}.json',dict(finished=True,results=results,updated=time.time()))

def node(rank):
 children=[]
 for local in range(8):
  gpu=rank*8+local
  with (OUT/'logs'/f'worker-{gpu:02d}.log').open('a') as log:children.append(subprocess.Popen([str(PY),str(__file__),'worker',str(gpu)],stdout=log,stderr=subprocess.STDOUT))
  time.sleep(.3)
 codes=[p.wait() for p in children];atomic_json(OUT/f'node-{rank}.json',dict(finished=True,codes=codes,time=time.time()));assert not any(codes)

if __name__=='__main__':
 kind=sys.argv[1]
 if kind=='prepare':print(prepare())
 elif kind=='worker':worker(int(sys.argv[2]))
 elif kind=='node':node(int(sys.argv[2]))
 elif kind=='detach':
  rank=int(sys.argv[2])
  with (OUT/f'node-{rank}.log').open('a') as log:p=subprocess.Popen([str(PY),str(__file__),'node',str(rank)],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,close_fds=True)
  print(json.dumps(dict(rank=rank,pid=p.pid)))
 elif kind=='launch':
  with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
   for value in pool.map(lambda r:remote(r,[PY,__file__,'detach',r]),range(3)):print(value)
