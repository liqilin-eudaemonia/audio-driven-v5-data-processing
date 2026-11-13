import os,sys,time,threading,fcntl
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];RUN=HERE/'run';WORK=HERE/'preview_queue';REF=ROOT/'experiments/p2_reintegrated_final_eval_20261028';PY='/private/liqi_workspace/project_code/Omni-LTX-stage2/.venv/bin/python'
os.environ['P2_EVAL_CLUSTER_WORK']=str(WORK);sys.path[:0]=[str(REF),str(ROOT/'src'),str(ROOT/'scripts')]
import cluster_eval as q
from p2_listen.contracts import read,digest,atomic_json
from p2_autonomous_eval.protocol import open_protocol,read_plan
from listen_cluster import launch
CLI=ROOT/'scripts/p2_autonomous_eval.py'
def main():
 WORK.mkdir(exist_ok=True);lock=(WORK/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 p=open_protocol(RUN);obs=sorted(p['observations'],key=lambda o:o['sample_id']);ids=[]
 for o in obs:
  sid=o['sample_id']
  if (RUN/'plans'/sid/'complete.json').exists():read_plan(RUN,p,sid);ids.append(sid)
 assert len(ids)==3
 jobs=[j for j in p['jobs'] if j['sample_id'] in ids];assert len(jobs)==12
 atomic_json(HERE/'preview_jobs.json',jobs);atomic_json(HERE/'preview_scope.json',dict(sample_ids=ids,outputs=12,seeds=[137,251],role='3of24 accepted planning diagnostic; not full evaluation'))
 for sid in ids:
  index=next(i for i,o in enumerate(obs) if o['sample_id']==sid)
  for part in ['text','input_plan']:
   q.publish(dict(id=f'{part}_{index}',kind='cache',priority=0,cmd=[PY,str(CLI),'cache','--work',str(RUN),'--part',part,'--shard-index',str(index),'--shards','24']))
 cfg=read(ROOT/'experiments/p2_full50h_duration384_20261028/cluster.json');cancel=threading.Event();errors=[]
 commands=[['env','P2_EVAL_CLUSTER_WORK='+str(WORK),PY,str(REF/'cluster_eval.py'),'node',str(i)] for i in range(3)]
 def execute():
  try:launch(cfg,commands,WORK/'launches'/str(time.time_ns()),cancel_event=cancel)
  except Exception as e:errors.append(repr(e))
 thread=threading.Thread(target=execute);thread.start()
 try:
  while not (WORK/'STOP').exists():
   if errors:raise RuntimeError(errors)
   ts=q.all_tasks();failed=[t['id'] for t in ts if q.state(t).get('failures',0)>=3]
   if failed:raise RuntimeError(failed)
   for j in jobs:
    index=next(i for i,o in enumerate(obs) if o['sample_id']==j['sample_id'])
    if all(q.done({'id':f'{part}_{index}'}) for part in ['text','input_plan']):
     q.publish(dict(id='preview_'+digest(j)[:16],kind='autonomous_preview',priority=1 if j['seed']==137 else 2,cmd=[PY,str(CLI),'generate','--job',str(RUN/'jobs'/(digest(j)+'.json'))],job=j))
   ts=q.all_tasks();gens=[t for t in ts if t['kind']=='autonomous_preview'];counts={k:dict(total=sum(t['kind']==k for t in ts),complete=sum(t['kind']==k and q.done(t) for t in ts),running=sum(t['kind']==k and q.state(t).get('status')=='running' for t in ts)) for k in ['cache','autonomous_preview']}
   atomic_json(HERE/'preview_status.json',dict(at=time.time(),phase='generating',tasks=counts,training_paused=True))
   if len(gens)==12 and all(q.done(t) for t in ts):
    result=dict(status='complete',outputs=12,samples=3,metrics_computed=False)
    atomic_json(WORK/'AVAILABLE_COMPLETE.json',result);atomic_json(HERE/'preview_status.json',result);break
   time.sleep(3)
  if (WORK/'AVAILABLE_COMPLETE.json').exists():thread.join(timeout=45)
 finally:cancel.set();thread.join(timeout=45)
if __name__=='__main__':
 try:main()
 except Exception as e:atomic_json(HERE/'preview_status.json',dict(status='failed',error=repr(e)));raise
