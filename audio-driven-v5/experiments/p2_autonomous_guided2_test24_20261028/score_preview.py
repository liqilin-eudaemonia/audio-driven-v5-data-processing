"""Score exactly the three preview IDs against the same-ID oracle controls."""
import os,sys,time,threading,fcntl,uuid
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];WORK=HERE/'preview_metrics';QUEUE=WORK/'queue';REF=ROOT/'experiments/p2_reintegrated_final_eval_20261028';PY='/private/liqi_workspace/project_code/Omni-LTX-stage2/.venv/bin/python'
sys.path[:0]=[str(ROOT/'src'),str(REF),str(ROOT/'scripts')];os.environ['P2_EVAL_CLUSTER_WORK']=str(QUEUE)
from p2_listen.contracts import read,sha,digest,atomic_json
from p2_glare_eval.common import artifact_valid
ARMS=['auto_av','auto_av_event','oracle_av','oracle_av_event']
RUNTIME=ROOT/'configs/p2_omni_runtime.lock.json'

def prepare():
 from p2_omni_eval.protocol import prepare as native_prepare
 ids=read(HERE/'preview_scope.json')['sample_ids'];jobs=read(HERE/'preview_jobs.json');expected={(s,k) for s in ids for k in [137,251]}
 originals=read(REF/'jobs.test24.json');gt={r['sample_id']:r for r in read(ROOT/'experiments/p2_test24_fourway_eval_20261027/E0/omni_pairs.json')['samples']}
 sources={};videos=set()
 for arm in ARMS:
  source_arm='guided_av_event_step1249' if arm.endswith('av_event') else 'guided_av_step1249'
  selected=[j for j in (jobs if arm.startswith('auto') else originals) if j['sample_id'] in ids and j['arm']==('autonomous_'+source_arm if arm.startswith('auto') else source_arm)]
  assert {(j['sample_id'],j['seed']) for j in selected}==expected and len(selected)==6
  rows=[]
  for j in selected:
   out=Path(j['output']);m=read(out/'complete.json');assert artifact_valid(out,m['contract']);meta=read(out/'completed.json')
   if arm.startswith('auto'):assert meta['target_annotation_access'] is False and meta['plan_source']=='qwen_autonomous'
   r=dict(sample_id=j['sample_id'],sample_seed=j['seed'],gt_video=gt[j['sample_id']]['gt_video'],gt_audio=gt[j['sample_id']]['gt_audio'],generated_video=str(out/'generated.mp4'),generated_audio=str(out/'listener.wav'));rows.append(r)
   videos.update(r[k] for k in ['gt_video','generated_video'])
  folder=WORK/arm;atomic_json(folder/'pairs.json',dict(schema='p2-omni-pairs-v1',samples=rows),immutable=True)
  if arm.startswith('auto'):native_prepare(folder/'pairs.json',folder/'omni',RUNTIME)
  else:
   path=REF/'metrics'/source_arm/'test24/omni';sources[arm]=dict(path=str(path),results_sha256=sha(path/'results.json'),protocol_sha256=sha(path/'protocol.json'))
 atomic_json(WORK/'scope.json',dict(sample_ids=ids,seeds=[137,251],conditions=ARMS,outputs_per_condition=6,oracle_sources=sources,
    role='3of24 planner-success subset; no inference population conclusion',audio='independent listener.wav; not speaker-mixed'),immutable=True)
 for i,video in enumerate(sorted(videos)):atomic_json(WORK/'videos'/f'{i:03d}.json',[video],immutable=True)
 return sorted(videos)

def row(arm,index):
 from p2_omni_eval.common import subprocess_env,job_key,complete
 from p2_omni_eval.pipeline import compute_row,check_protocol,has_error
 from p2_omni_eval.backends import Backends
 p=read(WORK/arm/'omni/protocol.json');os.environ.update(subprocess_env(p['runtime'],'cuda:0'));check_protocol(p)
 r=p['rows'][index];dest=WORK/arm/'omni/rows'/job_key(r['sample_id'],r['sample_seed']);dest.mkdir(parents=True,exist_ok=True);identity=dict(protocol=digest(p),row=r,device='cuda:0')
 with (dest/'row.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  if artifact_valid(dest,identity):return
  attempt=dest/'attempts'/uuid.uuid4().hex;attempt.mkdir(parents=True);backend=Backends(p,'cuda:0');result=compute_row(r,p,backend,attempt)
  atomic_json(attempt/'result.json',result);atomic_json(dest/'result.json',result)
  if has_error(result):raise RuntimeError(result)
  complete(dest,identity,[dest/'result.json',*attempt.glob('*.npy')]);check_protocol(p)

def reaction_scope():
 from p2_glare_eval.media import probe,aligned
 scope=read(WORK/'scope.json');ids=scope['sample_ids'];seeds=scope['seeds'];records={};valid={};coverage={}
 for arm in ARMS:
  records[arm]={};valid[arm]={};coverage[arm]=[]
  for r in read(WORK/arm/'pairs.json')['samples']:
   geo={s:probe(r[s+'_video']) for s in ['gt','generated']};aligned(geo['gt'],geo['generated']);h={s:sha(r[s+'_video']) for s in geo};passed=True
   for side,value in h.items():
    for mode in ['rgb','unfiltered']:
     folder=REF/'detections'/mode/value;m=read(folder/'complete.json');assert artifact_valid(folder,m['contract'])
    passed=passed and read(REF/'detections/rgb'/value/'complete.json')['valid']
   key=(r['sample_id'],r['sample_seed']);valid[arm][key]=passed;coverage[arm].append(dict(sample_id=key[0],seed=key[1],strict_valid=passed,quality={side:read(REF/'detections/unfiltered'/value/'quality.json') for side,value in h.items()}))
   for mode in ['strict','unfiltered']:
    if mode=='strict' and not passed:continue
    item=dict(sample_id=key[0],sample_seed=key[1],gt_video=r['gt_video'],generated_video=r['generated_video'],geometry=geo['gt'])
    for side,value in h.items():
     path=REF/'detections'/('rgb' if mode=='strict' else 'unfiltered')/value/'scores.csv';item[side+'_csv']=str(path);item[side+'_csv_sha256']=sha(path)
    records[arm][key,mode]=item
 common=[s for s in ids if all(valid[a][s,k] for a in ARMS for k in seeds)]
 atomic_json(WORK/'reaction_scope.json',dict(strict_ids=common,unfiltered_ids=ids,coverage=coverage),immutable=True)
 for mode,members in [('strict',common),('unfiltered',ids)]:
  for arm in ARMS:
   for seed in seeds:atomic_json(WORK/'reaction_inputs'/f'{arm}.{mode}.{seed}.json',[records[arm][(s,seed),mode] for s in members],immutable=True)
 return common

def score(arm,mode,seed):
 from p2_glare_eval.scoring import score_seed
 rows=read(WORK/'reaction_inputs'/f'{arm}.{mode}.{seed}.json');result=score_seed(rows,'cuda:0',16) if rows else None
 atomic_json(WORK/'reaction_scores'/f'{arm}.{mode}.{seed}.json',dict(result=result,samples=len(rows),seed=seed,mode=mode))

def controller():
 import cluster_eval as q
 from listen_cluster import launch
 WORK.mkdir(exist_ok=True);QUEUE.mkdir(exist_ok=True);lock=(QUEUE/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 videos=prepare();runtime=read(RUNTIME);mp=runtime['paths']['metrics_python'];dp=ROOT/'.venv-p2-glare-detector/bin/python'
 for arm in ARMS[:2]:
  for i in range(6):q.publish(dict(id=f'row_{arm}_{i}',kind='omni_row',priority=0,cmd=[mp,str(__file__),'row',arm,str(i)]))
 for i,video in enumerate(videos):q.publish(dict(id=f'detect_{i:03d}',kind='detection',priority=0,cmd=['env','CUDA_VISIBLE_DEVICES=','TF_ENABLE_ONEDNN_OPTS=0','TF_CPP_MIN_LOG_LEVEL=2',str(dp),str(REF/'eval_metrics.py'),'detect',str(WORK/'videos'/f'{i:03d}.json')]))
 cfg=read(ROOT/'experiments/p2_full50h_duration384_20261028/cluster.json');cancel=threading.Event();errors=[]
 commands=[['env','P2_EVAL_CLUSTER_WORK='+str(QUEUE),PY,str(REF/'cluster_eval.py'),'node',str(i)] for i in range(3)]
 def run():
  try:launch(cfg,commands,QUEUE/'launches'/str(time.time_ns()),cancel_event=cancel)
  except Exception as e:errors.append(repr(e))
 thread=threading.Thread(target=run);thread.start();scored=False
 try:
  while not (QUEUE/'STOP').exists():
   if errors:raise RuntimeError(errors)
   ts=q.all_tasks();failed=[t['id'] for t in ts if q.state(t).get('failures',0)>=3]
   if failed:raise RuntimeError(failed)
   if not scored and all(q.done(t) for t in ts if t['kind']=='detection'):
    common=reaction_scope()
    for mode in ['strict','unfiltered']:
     for arm in ARMS:
      for seed in [137,251]:q.publish(dict(id=f'reaction_{arm}_{mode}_{seed}',kind='reaction',priority=1,cmd=[mp,str(__file__),'score',arm,mode,str(seed)]))
    scored=True
   ts=q.all_tasks();counts={k:dict(total=sum(t['kind']==k for t in ts),complete=sum(t['kind']==k and q.done(t) for t in ts),running=sum(t['kind']==k and q.state(t).get('status')=='running' for t in ts)) for k in ['omni_row','detection','reaction']};atomic_json(WORK/'status.json',dict(at=time.time(),phase='computing',tasks=counts))
   if scored and all(q.done(t) for t in ts):
    atomic_json(QUEUE/'AVAILABLE_COMPLETE.json',dict(status='complete'));atomic_json(WORK/'status.json',dict(status='metrics_computed',tasks=counts));break
   time.sleep(3)
  if (QUEUE/'AVAILABLE_COMPLETE.json').exists():thread.join(timeout=45)
 finally:cancel.set();thread.join(timeout=45)
if __name__=='__main__':
 if sys.argv[1]=='row':row(sys.argv[2],int(sys.argv[3]))
 elif sys.argv[1]=='score':score(sys.argv[2],sys.argv[3],int(sys.argv[4]))
 elif sys.argv[1]=='controller':
  try:controller()
  except Exception as e:atomic_json(WORK/'status.json',dict(status='failed',error=repr(e)));raise
 else:raise ValueError(sys.argv[1])
