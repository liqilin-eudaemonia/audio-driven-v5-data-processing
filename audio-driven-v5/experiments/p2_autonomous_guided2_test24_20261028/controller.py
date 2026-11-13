import os,sys,json,time,threading,fcntl
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];RUN=HERE/'run';WORK=HERE/'cluster24';REF=ROOT/'experiments/p2_reintegrated_final_eval_20261028';PY='/private/liqi_workspace/project_code/Omni-LTX-stage2/.venv/bin/python'
os.environ['P2_EVAL_CLUSTER_WORK']=str(WORK)
sys.path[:0]=[str(REF),str(ROOT/'src'),str(ROOT/'scripts')]
import cluster_eval as q
from p2_listen.contracts import read,sha,digest,atomic_json
from listen_cluster import launch
CLI=ROOT/'scripts/p2_autonomous_eval.py'
def pub(id,kind,priority,cmd,**kw):
 t=dict(id=id,kind=kind,priority=priority,cmd=list(map(str,cmd)),**kw);q.publish(t);return t
def status(**kw):atomic_json(HERE/'status.json',dict(dict(at=time.time(),training_paused=True,scope='original test24,2 final guided frozen-Qwen arms,2 seeds,96 outputs'),**kw))
def controller():
 WORK.mkdir(exist_ok=True);lock=(WORK/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert (ROOT/'experiments/p2_full50h_duration384_20261028/STOP').exists()
 p=read(RUN/'protocol.json');jobs=p['jobs'];assert len(jobs)==96
 obs=sorted(p['observations'],key=lambda o:o['sample_id']);smoke_id=min(obs,key=lambda o:o['requested_duration_s'])['sample_id']
 smoke=next(j for j in jobs if j['sample_id']==smoke_id and j['source_arm']=='guided_av_step1249' and j['seed']==137)
 for i,o in enumerate(obs):pub(f'plan_{i:02d}','plan',-5 if o['sample_id']==smoke_id else 0,[PY,CLI,'plan','--work',RUN,'--shard-index',i,'--shards',24])
 cfg=read(ROOT/'experiments/p2_full50h_duration384_20261028/cluster.json');cancel=threading.Event();errors=[]
 commands=[['env','P2_EVAL_CLUSTER_WORK='+str(WORK),PY,str(REF/'cluster_eval.py'),'node',str(i)] for i in range(3)]
 def run():
  try:launch(cfg,commands,WORK/'launches'/str(time.time_ns()),cancel_event=cancel)
  except Exception as e:errors.append(repr(e))
 thread=threading.Thread(target=run);thread.start();smoke_passed=False
 try:
  while not (WORK/'STOP').exists():
   if errors:raise RuntimeError(errors)
   ts=q.all_tasks();failed=[t['id'] for t in ts if q.state(t).get('failures',0)>=3]
   if failed:raise RuntimeError('Repeated task failures: '+str(failed))
   for i,o in enumerate(obs):
    if q.done({'id':f'plan_{i:02d}'}):
     for part in ['text','input_plan']:pub(f'{part}_{i:02d}','cache',-4 if o['sample_id']==smoke_id else 1,[PY,CLI,'cache','--work',RUN,'--part',part,'--shard-index',i,'--shards',24])
   si=next(i for i,o in enumerate(obs) if o['sample_id']==smoke_id)
   if all(q.done({'id':f'{part}_{si:02d}'}) for part in ['text','input_plan']):
    pub('s0_generate','autonomous_generate',-3,[PY,CLI,'generate','--job',RUN/'jobs'/(digest(smoke)+'.json')],job=smoke)
   if q.done({'id':'s0_generate'}):
    meta=read(Path(smoke['output'])/'completed.json');assert meta['target_annotation_access'] is False and meta['gt_media_access'] is False and meta['plan_source']=='qwen_autonomous'
    if not smoke_passed:atomic_json(HERE/'GPU_S0.json',dict(status='passed',job=smoke,metadata=meta,at=time.time()))
    smoke_passed=True
   if smoke_passed:
    for j in jobs:
     if j==smoke:continue
     i=next(i for i,o in enumerate(obs) if o['sample_id']==j['sample_id'])
     if all(q.done({'id':f'{part}_{i:02d}'}) for part in ['text','input_plan']):
      pub('generate_'+digest(j)[:16],'autonomous_generate',2,[PY,CLI,'generate','--job',RUN/'jobs'/(digest(j)+'.json')],job=j,duration=obs[i]['requested_duration_s'])
    gens=[t for t in q.all_tasks() if t['kind']=='autonomous_generate']
    for arm in sorted({j['arm'] for j in jobs}):
     group=[t for t in gens if t['job']['arm']==arm]
     if len(group)==48 and all(q.done(t) for t in group):pub('metrics_'+arm,'metrics',3,[PY,HERE/'run_metrics.py',arm])
    ts=q.all_tasks()
    if sum(t['kind']=='metrics' and q.done(t) for t in ts)==2 and all(q.done(t) for t in ts):
     marker=dict(status='complete',outputs=96,arms=2,samples=24,seeds=[137,251],plan_source='qwen_autonomous',training_paused=True)
     atomic_json(WORK/'AVAILABLE_COMPLETE.json',marker);status(**marker);break
   ts=q.all_tasks();counts={k:dict(total=sum(t['kind']==k for t in ts),complete=sum(t['kind']==k and q.done(t) for t in ts),running=sum(t['kind']==k and q.state(t).get('status')=='running' for t in ts)) for k in ['plan','cache','autonomous_generate','metrics','metric_shard']}
   status(phase='full_evaluation' if smoke_passed else 'gpu_s0_and_parallel_cache',tasks=counts,gpu_s0_passed=smoke_passed)
   time.sleep(3)
  if (WORK/'AVAILABLE_COMPLETE.json').exists():thread.join(timeout=45)
 finally:cancel.set();thread.join(timeout=45)
if __name__=='__main__':
 try:controller()
 except Exception as e:status(phase='failed',error=repr(e));raise
