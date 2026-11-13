from pathlib import Path
import sys,os,copy,time,argparse,fcntl,json,subprocess
W=Path(__file__).resolve().parent;sys.path.insert(0,str(W))
if (W/'run/run.json').exists():
 import isolate_events
 isolate_events.activate()
import runtime28 as R
from p2_listen.contracts import read,atomic_json,digest,sha
PY='/private/liqi_workspace/project_code/Omni-LTX-stage2/.venv/bin/python'
def prepare():
 parent=W.parent/'p2_stage2_joint_balance_v2_20261112/candidate_full6000';old=read(parent/'config.json');R.ORIGINAL_VALIDATE(old)
 pre=read(parent/'preflight.json');assert pre['source']==R.ORIGINAL_SOURCE()and pre['configuration_sha256']==digest(old)
 out=R.RUN;out.mkdir();(out/'rank_logs').mkdir()
 c=copy.deepcopy(old);c['global_batch']=28;c['output']=str(out);c['generator']['output']=str(out)
 atomic_json(out/'policy.json',read(c['policy']));c['policy']=str(out/'policy.json');c['policy_sha256']=sha(c['policy']);R.validate(c)
 T=R.activate();assert T.data_identity(c)==pre['data_identity']
 schedule=R.Schedule(c);original=R.Schedule(dict(c));assert len(schedule.pairs)==8400
 for step in(1,100,300):assert [schedule.indices(schedule.size,step,28,schedule.seed,r,28)[0]for r in range(28)]==schedule.pairs[(step-1)*28:step*28]
 pre.update(configuration_sha256=digest(c),source=R.source(),schedule_sha256=digest(schedule.pairs),trial_samples=8400)
 cl=copy.deepcopy(read(W.parent/'p2_ltx_all_lora_noevent28_batch28_20261109/cluster.json'));cl.update(master_port=29987,training_script=str(W/'train_worker.py'),training_config=str(out/'config.json'))
 atomic_json(out/'config.json',c,immutable=True);atomic_json(out/'preflight.json',pre,immutable=True);atomic_json(out/'cluster.json',cl,immutable=True)
 atomic_json(W/'PREPARED.json',dict(success=True,at=time.time(),world=28,batch=28,source=R.source(),reused_real32_and16_preflight=str(parent),frozen_scope_and_losses_unchanged=True,gpu_started=False))
 print('joint28 prepared',flush=True)
def launch(resume=None):
 import p2_whisper_event_feature_train31 as native
 from listen_cluster import launch as cluster_launch
 c=R.validate(read(R.RUN/'config.json'));cl=read(R.RUN/'cluster.json');T=R.activate();T.precheck(c,Path(resume)if resume else None)
 with(W/'launch.lock').open('a')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  resources=native.selected_resources(cl);atomic_json(W/'RESOURCES.json',resources)
  reports={n['node']:n for n in resources['nodes']}
  for attempt in range(8):
   missing=[n for n in cl['nodes']if not reports[n['name']].get('gpus')]
   if not missing:break
   time.sleep(min(2+attempt,8))
   for report in native.selected_resources(dict(cl,nodes=missing))['nodes']:reports[report['node']]=report
  resources['nodes']=[reports[n['name']]for n in cl['nodes']];resources['selected_all_idle']=all(n.get('selected_idle')for n in resources['nodes']);atomic_json(W/'RESOURCES.json',resources)
  if any(not n.get('gpus')for n in resources['nodes']):raise RuntimeError('Resource probe could not reach a node after retries; availability unknown')
  if not resources['selected_all_idle']:raise RuntimeError('Approved28 GPUs not idle; preserve other jobs')
  cmds=[]
  for i,n in enumerate(cl['nodes']):
   cmd=['env','PYTHONPATH='+str(W/'isolation'),'CUDA_VISIBLE_DEVICES='+','.join(map(str,n['gpus'])),'PYTHONUNBUFFERED=1',*[k+'='+str(v)for k,v in cl['network'].items()],PY,'-u',str(W/'train_worker.py'),'node','--index',str(i)]
   if resume:cmd+=['--resume',str(resume)]
   cmds.append(cmd)
  atomic_json(W/'TRAINING_CONTROLLER.json',dict(pid=os.getpid(),at=time.time(),resume=resume))
  cluster_launch(cl,cmds,W/('train_nodes_'+str(time.time_ns())))
 print(read(R.RUN/'status.json'),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','launch']);p.add_argument('--resume');a=p.parse_args();prepare()if a.action=='prepare'else launch(a.resume)
