"""Sequential full28 stage queue; advancement requires generation review."""
from pathlib import Path
import sys,argparse,copy,csv,math,json,time,fcntl,os
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from p2_listen.contracts import read,sha,atomic_json
from .config import WORK,CANDIDATES,make,validate,source_identity,normalized

def prepare():
 from p2_ltx_all_lora.config import expected
 from p2_j_duration_execution import make_pairs
 c=expected(WORK/'unused');t=c['training']
 with Path(c['sample_list']).open()as f:rows=list(csv.DictReader(f))
 size=len(rows);steps=math.ceil(size/28)
 assert size==284218 and steps==10151
 plan=dict(size=size,batch=28,seed=t['seed'],steps=steps,window=448,sample_list_sha256=sha(c['sample_list']),pairs=make_pairs([dict(requested_duration_s=float(x['duration_s']))for x in rows],steps,28,t['seed'],0,'duration',448))
 atomic_json(WORK/'schedule.json',plan,immutable=True)
 protocol=dict(world=28,global_batch=28,nodes=dict(node12=list(range(8)),node8=list(range(8)),node4=list(range(8)),node11=list(range(4))),candidates={k:list(v)for k,v in CANDIDATES.items()},stages=[500,1500,3000,6000],generation_based_promotion=True,initialization='same_pretrained_and_seed',sample_list_sha256=plan['sample_list_sha256'])
 atomic_json(WORK/'protocol.json',protocol,immutable=True)
 base=read(ROOT/'experiments/p2_ltx_all_lora_noevent28_batch28_20261109/cluster.json')
 for i,name in enumerate(CANDIDATES):
  c=make(name);out=Path(c['output']);out.mkdir(parents=True,exist_ok=True);(out/'rank_logs').mkdir(exist_ok=True)
  atomic_json(out/'config.json',c,immutable=True)
  cl=copy.deepcopy(base);cl.update(master_port=30110+i,training_script=str(Path(__file__).parent/'worker.py'),training_config=str(out/'config.json'))
  atomic_json(out/'cluster.json',cl,immutable=True)
 atomic_json(WORK/'PREPARED.json',dict(source=source_identity(),candidates=list(CANDIDATES),at=time.time()))
 return protocol

def preflight():
 from p2_whisper_av.bootstrap import activate_ltx
 activate_ltx()
 from .training import preflight as check
 from p2_listen.contracts import digest
 first=None
 for name in CANDIDATES:
  c=validate(read(WORK/'runs'/name/'config.json'))
  if first is None:first=check(c)
  v=copy.deepcopy(first);v['configuration_sha256']=digest(c)
  atomic_json(Path(c['output'])/'preflight.json',v)
 atomic_json(WORK/'PREFLIGHT_ACCEPTED.json',dict(success=True,at=time.time(),source=source_identity(),samples=first['samples'],canary_reads=first['canary_reads'],schedule_pairs=first['schedule_pairs']))

def stages():
 yield 500,list(CANDIDATES)
 for target in (1500,3000,6000):
  path=WORK/f'promotion_{target}.json'
  if not path.exists():return
  v=read(path)
  if v.get('target')!=target or v.get('basis')!='generated_av_quality' or not v.get('evidence') or not v.get('candidates')or len(set(v['candidates']))!=len(v['candidates'])or any(n not in CANDIDATES for n in v['candidates']):raise ValueError('Promotion requires explicit generation evidence and valid candidates')
  for f in v['evidence']:
   if not Path(f).is_file():raise ValueError('Missing generation review evidence')
  yield target,v['candidates']

def run():
 from .checkpoint import checked
 from .resources import idle
 from listen_cluster import launch
 with(WORK/'queue.lock').open('a')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  st=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split();atomic_json(WORK/'CONTROLLER.json',dict(pid=os.getpid(),start_ticks=st[19],at=time.time()))
  if not read(WORK/'TESTS_ACCEPTED.json').get('success'):raise ValueError('Search tests not accepted')
  expected=read(WORK/'PREFLIGHT_ACCEPTED.json')['source']
  if expected!=source_identity():raise ValueError('Source changed after preflight')
  for target,names in stages():
   for name in names:
    if(WORK/'STOP').exists():return
    c=validate(read(WORK/'runs'/name/'config.json'));out=Path(c['output']);receipt=out/f'STAGE_{target}.json'
    if receipt.exists():
     cp=checked(out/f'step_{target:06d}.pt')
     if cp['step']!=target:raise ValueError('Invalid completed stage')
     continue
    # A deliberate user STOP is never removed by the queue.
    if(out/'STOP').exists():raise RuntimeError('Candidate has explicit STOP')
    cl=read(out/'cluster.json');snap=idle(cl)
    atomic_json(WORK/'RESOURCES_LATEST.json',snap)
    if not snap['idle']:raise RuntimeError('Selected training GPUs still have compute jobs; no overlapping launch')
    files=sorted(out.glob('step_*.complete.json'));resume=None
    if files:
     resume=files[-1].with_suffix('').with_suffix('.pt');v=checked(resume)
     if v['identity']['experiment']!=c or len(v['rank_rng'])!=28:raise ValueError('Wrong candidate checkpoint')
     if v['step']>target:raise ValueError('Checkpoint beyond stage')
    elif(out/'run.json').exists():raise RuntimeError('Run identity exists without complete checkpoint; inspect startup')
    stamp=time.time_ns();commands=[]
    for i,n in enumerate(cl['nodes']):
     cmd=['env','CUDA_VISIBLE_DEVICES='+','.join(map(str,n['gpus'])),'PYTHONUNBUFFERED=1',*[k+'='+str(v)for k,v in cl['network'].items()],cl['python'],'-u',cl['training_script'],'node','--index',str(i),'--config',str(out/'config.json'),'--stop-after',str(target)]
     if resume:cmd+=['--resume',str(resume)]
     commands.append(cmd)
    atomic_json(WORK/'STATE.json',dict(stage='training',candidate=name,target=target,at=time.time()))
    launch(cl,commands,out/f'nodes_{target}_{stamp}')
    cp=checked(out/f'step_{target:06d}.pt')
    if cp['step']!=target or cp['identity']['experiment']!=c or len(cp['rank_rng'])!=28:raise ValueError('Stage did not finish correctly')
    atomic_json(receipt,dict(success=True,step=target,checkpoint=str(out/f'step_{target:06d}.pt'),sha256=sha(out/f'step_{target:06d}.pt'),at=time.time()))
   atomic_json(WORK/'STATE.json',dict(stage='awaiting_generation_review',completed_target=target,at=time.time()))
  # No automatic loss-based ranking or speculative promotion.

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','preflight','run']);a=p.parse_args();v=globals()[a.action]()
 if v is not None:print(json.dumps(v,ensure_ascii=False))
