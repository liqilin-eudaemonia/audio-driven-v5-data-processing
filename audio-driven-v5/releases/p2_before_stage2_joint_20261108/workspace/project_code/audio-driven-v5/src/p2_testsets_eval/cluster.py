"""Wait for the authorized training queue, then use the selected inference nodes for inference."""
import concurrent.futures,fcntl,json,os,subprocess,sys,time,threading
from pathlib import Path
from .protocol import ROOT,OMNI,PYTHON,METRIC_PYTHON,read,save,rows,sha
CLI=ROOT/'scripts/p2_testsets_eval.py'

def stopped(work):return (work/'STOP').exists()
def state(work,stage,**kw):save(work/'status.json',dict(stage=stage,at=time.time(),**kw))
def environment():
 e=dict(os.environ);e.update(PYTHONPATH='/private/liqi_workspace/.cache/paired_video_metrics/python',
     OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',PYTHONUNBUFFERED='1')
 return e

def run_cluster(work,stage,limit=None,arms=None):
 sys.path.insert(0,str(ROOT/'scripts'));from listen_cluster import transport
 p=read(work/'protocol.json');cfg=read(p['cluster']);stage_key=(('smoke_'+stage) if limit else stage)+('_'+'_'.join(arms) if arms else '');state(work,stage_key)
 startup={n['name']:threading.Semaphore(2) for n in cfg['nodes']}
 def task(rank):
  node=cfg['nodes'][rank//8];gpu=node['gpus'][rank%8]
  for attempt in range(1,9):
   if stopped(work):raise RuntimeError('User STOP requested')
   command=['env',f'CUDA_VISIBLE_DEVICES={gpu}','OMP_NUM_THREADS=4','OPENBLAS_NUM_THREADS=2',
       'TOKENIZERS_PARALLELISM=false','PYTHONUNBUFFERED=1','PYTORCH_ALLOC_CONF=expandable_segments:True',
       'PYTHONPATH=',
       str(METRIC_PYTHON if stage=='metrics' else PYTHON),str(CLI if stage=='metrics' else p.get('worker_cli',CLI)),
       'metrics' if stage=='metrics' else 'worker','--work',str(work),'--rank',str(rank)]
   if stage!='metrics':command+=['--stage',stage]
   if limit:command+=['--limit',str(limit)]
   if arms:command+=['--arms',*arms]
   folder=work/'logs'/stage_key;folder.mkdir(parents=True,exist_ok=True)
   log=folder/f'rank{rank:02d}_{time.time_ns()}_attempt{attempt}.log'
   with log.open('w') as stream:
    with startup[node['name']]:
     process=subprocess.Popen(transport(cfg,node,command),stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
     time.sleep(.5)
    save(work/'processes'/stage_key/f'{rank:02d}.json',dict(pid=process.pid,node=node['name'],gpu=gpu,log=str(log),command=command,at=time.time()))
    code=process.wait()
   if code==0:return dict(rank=rank,success=True,attempts=attempt,log=str(log))
   content=log.read_text(errors='replace')
   if code!=255 or 'kex_exchange_identification' not in content:
    if attempt>=3:raise RuntimeError(f'{stage} rank {rank} failed: {log}')
   time.sleep(min(2*attempt,15)+(rank%4)*.25)
  raise RuntimeError(f'{stage} rank {rank} failed after connection retries; last log {log}')
 with concurrent.futures.ThreadPoolExecutor(max_workers=p['workers']) as pool:
  results=list(pool.map(task,range(p['workers'])))
 save(work/'stages'/(stage_key+'.json'),dict(complete=True,results=results,at=time.time()))

def bind(work,arms=None):
 p=read(work/'protocol.json');chosen=[a for a in p['arms'] if arms is None or a['name'] in arms]
 def one(pair):
  arm,dataset=pair;spec=p['datasets'][dataset];refs=rows(Path(spec['prepared'])/'references.jsonl')
  out=work/'bindings'/arm['name']/dataset
  if (out/'binding_audit.json').exists() and (out/'pairs.jsonl').exists():
   if read(out/'binding_audit.json')['status']=='ready_for_scoring':return
  if out.exists():out.rename(out.with_name(out.name+f'.failed.{time.time_ns()}'))
  mapping=work/'maps'/arm['name']/(dataset+'.jsonl');mapping.parent.mkdir(parents=True,exist_ok=True)
  output=[]
  from .runtime import ready
  for r in refs:
   folder=work/'generated'/arm['name']/dataset/r['id']
   if not ready(folder):raise ValueError('Missing generation: '+str(folder))
   output.append(dict(id=r['id'],dataset=dataset,generated=str(folder/'generated.mp4'),generated_audio=str(folder/'listener.wav'),target_role='listener',timeline='same_pair_window'))
  mapping.write_text(''.join(json.dumps(r)+'\n' for r in output))
  module='responsenet_pairs' if dataset=='responsenet' else 'seamless'
  cmd=[str(METRIC_PYTHON),'-m',f'evaluation.{module}.bind','--prepared',spec['prepared'],
       '--generated-map',str(mapping),'--output',str(out)]
  if dataset=='responsenet':cmd+=['--size','512']
  log=work/'logs'/f'bind_{arm["name"]}_{dataset}.log';log.parent.mkdir(parents=True,exist_ok=True)
  with log.open('w') as f:subprocess.run(cmd,cwd=OMNI,env=environment(),stdout=f,stderr=subprocess.STDOUT,check=True)
 with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
  list(pool.map(one,[(a,d) for a in chosen for d in p['datasets']]))
 from evaluation.listener_metrics.suite import enrich
 for rank in range(p['workers']):
  item_ids={x['id'] for x in read(work/'shards'/f'{rank:02d}.json')}
  for arm in chosen:
   for dataset in p['datasets']:
    selected=[r for r in rows(work/'bindings'/arm['name']/dataset/'pairs.jsonl') if r['id'] in item_ids]
    assert selected
    selected=enrich(selected,rows(Path(p['datasets'][dataset]['prepared'])/'references.jsonl'),rows(work/'maps'/arm['name']/(dataset+'.jsonl')),work)
    path=work/'metric_manifests'/arm['name']/dataset/f'{rank:02d}.jsonl';path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(''.join(json.dumps(r)+'\n' for r in selected))
 for arm in chosen:save(work/'stages'/('bind_'+arm['name']+'.json'),dict(complete=True,at=time.time()))

def metrics_worker(work,rank,arms=None):
 from .metrics import metrics_worker as execute
 return execute(work,rank,arms)


def merge(work):
 import numpy as np
 from evaluation.video.paired_metrics.run import summarize,METRICS,write_json
 p=read(work/'protocol.json');all_summaries={};all_audio={}
 from evaluation.listener_metrics.run import quality_summary
 from evaluation.listener_metrics.core import aggregate_activity
 for arm in p['arms']:
  for dataset,spec in p['datasets'].items():
   results=[];dist={k:[[],[]] for k in ['fid','fvd']};contracts=[];paircounts={k:0 for k in ['fid','fvd']}
   for rank in range(p['workers']):
    folder=work/'metric_shards'/arm['name']/dataset/f'{rank:02d}'
    summary=read(folder/'summary.json');contracts.append(read(folder/'contract.json'))
    results+=rows(folder/'pairs.jsonl')
    with np.load(folder/'features.npz') as features:
     for metric in dist:
      if features[metric+'_gt'].size:
       dist[metric][0].append(features[metric+'_gt']);dist[metric][1].append(features[metric+'_generated'])
      paircounts[metric]+=summary['coverage'][metric]['pairs']
   ids=[r['id'] for r in results];expected={r['id'] for r in rows(Path(spec['prepared'])/'references.jsonl')}
   assert len(ids)==len(set(ids)) and set(ids)==expected
   for c in contracts:
    assert c['code']==contracts[0]['code'] and c['assets']==contracts[0]['assets']
    for key in ('face_device','emotion_device'):
     assert c.get('execution',{}).get(key)==contracts[0].get('execution',{}).get(key)
   summary=summarize(results,dist,set(METRICS),contracts[0]['schema'],contracts[0]['datasets'],len(results),{})
   for metric in dist:summary['coverage'][metric]['pairs']=paircounts[metric]
   summary.update(arm=arm['name'],checkpoint=arm['checkpoint'],seed=p['seed'],planner=arm.get('planner',p['planner']),
       dataset=dataset,aggregation='existing metric formulas; FID/FVD recomputed from all shard features, never averaged')
   out=work/'metrics'/arm['name']/dataset;out.mkdir(parents=True,exist_ok=True)
   (out/'pairs.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in results))
   audio=[r for rank in range(p['workers']) for r in rows(work/'metric_shards'/arm['name']/dataset/f'{rank:02d}'/'audio/pairs.jsonl')]
   assert len(audio)==len(expected) and {r['id'] for r in audio}==expected
   valid=[r for r in audio if 'audio' in r]
   quality=quality_summary(audio,('utmos',))['utmos']
   summary['metrics']['utmos']=quality['generated']['mean']
   summary['audio']=dict(quality=dict(utmos=quality),valid_pairs=len(valid),requested_pairs=len(audio),
       pairs_with_errors=sum(bool(r['errors']) for r in audio),
       activity={k:aggregate_activity([r['audio']['comparison'][k] for r in valid]) for k in ('acoustic','speech')})
   if summary['audio']['pairs_with_errors']:summary['status']='partial'
   (out/'audio_pairs.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in audio))
   all_audio[arm['name']+'/'+dataset]=audio
   write_json(out/'summary.json',summary)
   save(out/'contract.json',dict(shard_contracts=contracts,protocol_sha256=sha(work/'protocol.json')))
   all_summaries[arm['name']+'/'+dataset]=summary
 common={}
 for dataset in p['datasets']:
  sets=[]
  for arm in p['arms']:
   audio=all_audio[arm['name']+'/'+dataset]
   sets.append({(r['id'],tuple(w['window'])) for r in audio for w in r.get('audio',{}).get('quality_windows',[]) if w['utmos']['status']=='success'})
  shared=set.intersection(*sets);scores={}
  for arm in p['arms']:
   subset=[]
   for r in all_audio[arm['name']+'/'+dataset]:
    if 'audio' not in r:continue
    windows=[w for w in r['audio']['quality_windows'] if (r['id'],tuple(w['window'])) in shared]
    if windows:subset.append(dict(id=r['id'],audio=dict(quality_windows=windows)))
   scores[arm['name']]=quality_summary(subset,('utmos',))['utmos']
  common[dataset]=dict(window_keys=sorted(shared),arms=scores)
 save(work/'UTMOS_COMMON_WINDOWS.json',common)
 save(work/'RESULTS.json',all_summaries)
 partial=any(s['status']!='complete' for s in all_summaries.values())
 state(work,'completed_with_metric_failures' if partial else 'complete',results=str(work/'RESULTS.json'))
 save(work/'COMPLETE.json',dict(status='partial' if partial else 'complete',all_generations=p['total_outputs'],
       scores=str(work/'RESULTS.json'),completed_at=time.time()))

def queue(work):
 sys.path.insert(0,str(ROOT/'scripts'));from listen_cluster import transport
 work=Path(work);lock=(work/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 p=read(work/'protocol.json');cfg=read(p['cluster'])
 try:
  while True:
   if stopped(work):state(work,'stopped');return
   statuses={a['kind']:read(Path(a['checkpoint']).parent/'status.json') for a in p['arms']}
   if all(r['complete'] and r['step']==8927 for r in statuses.values()):break
   state(work,'waiting_training',training=statuses);time.sleep(30)
  hashes={}
  for a in p['arms']:
   marker=read(Path(a['checkpoint']).with_suffix('.complete.json'));assert marker['step']==a['step']
   assert sha(a['checkpoint'])==marker['sha256'];hashes[a['name']]=marker['sha256']
  save(work/'checkpoint_hashes.json',hashes)
  while True:
   if stopped(work):state(work,'stopped');return
   free=[]
   for node in cfg['nodes']:
    try:
     result=subprocess.check_output(transport(cfg,node,['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits']),text=True,timeout=30)
    except (subprocess.CalledProcessError,subprocess.TimeoutExpired,OSError) as exc:
     save(work/'resource_errors'/(node['name']+'.json'),dict(at=time.time(),error=repr(exc),action='retry resource query next poll'))
     free=[];break
    free += [float(x) for x in result.splitlines()]
   if len(free)==p['workers'] and min(free)>=125*1024:break
   state(work,'waiting_training_gpu_release',free_mib=free);time.sleep(30)
  first=p['arms'][0]['name']
  if not (work/'SMOKE_COMPLETE.json').exists():
   for stage in ['cache','render','decode']:
    names=None if stage=='cache' else [first]
    key='smoke_'+stage+('_'+first if names else '')
    if not (work/'stages'/(key+'.json')).exists():run_cluster(work,stage,limit=2,arms=names)
   save(work/'SMOKE_COMPLETE.json',dict(status='generation_passed',ranks=p['workers'],samples=2*p['workers'],arm=first,metrics='separate_host',at=time.time()))
  if not (work/'stages/cache.json').exists():run_cluster(work,'cache')
  for arm in p['arms']:
   for stage in ['render','decode']:
    if not (work/'stages'/(stage+'_'+arm['name']+'.json')).exists():run_cluster(work,stage,arms=[arm['name']])
   if not (work/'stages'/('bind_'+arm['name']+'.json')).exists():
    state(work,'binding',arm=arm['name']);bind(work,[arm['name']])
  state(work,'generation_complete_metrics_independent',total_outputs=p['total_outputs'])
  save(work/'GENERATION_COMPLETE.json',dict(total_outputs=p['total_outputs'],at=time.time()))
 except Exception as e:
  state(work,'failed',error=repr(e));raise
