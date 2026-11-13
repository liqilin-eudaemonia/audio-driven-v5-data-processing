from pathlib import Path
import argparse,copy,fcntl,json,os,sys,time
W=Path(__file__).resolve().parent;sys.path.insert(0,str(W));import runtime as R
from p2_listen.contracts import read,sha,digest,atomic_json
PY='/private/liqi_workspace/project_code/Omni-LTX-stage2/.venv/bin/python'

def prepare():
 import eval28 as E
 old=read(R.PARENT/'run/config.json')
 for step in(0,100):E.E.verify_report(old,step)
 if E.E.decision(old,100)['decision']!='continue_to_300':raise ValueError('Parent100 safety failed')
 c=copy.deepcopy(old);c.update(output=str(R.RUN),steps=1000,milestones=[0,300,600,900,1000],save_every=300)
 c['generator']['output']=str(R.RUN)
 c['continuation']=dict(parent=str(R.PARENT_STATE),parent_sha256=read(R.PARENT_STATE.with_suffix('.complete.json'))['sha256'],
  user_request='Continue joint full-LoRA to1000, save every300 and at final1000; no intermediate evaluation pauses',
  optimizer_rng_preserved=True)
 R.RUN.mkdir();(R.RUN/'rank_logs').mkdir()
 R.activate();R.validate(c);pre=read(R.PARENT/'run/preflight.json');did=R.T.data_identity(c)
 if did!=pre['data_identity']:raise ValueError('Continuation data changed')
 schedule=R.Schedule(c);previous=R.OLD.Schedule(old)
 if schedule.pairs[:2800]!=previous.pairs[:2800]:raise ValueError('Original100 schedule changed')
 for step in(101,300,600,900,1000):
  if[schedule.indices(schedule.size,step,28,schedule.seed,r,28)[0]for r in range(28)]!=schedule.pairs[(step-1)*28:step*28]:raise ValueError('Rank schedule differs')
 pre.update(configuration_sha256=digest(c),source=R.source(),schedule_sha256=digest(schedule.pairs),trial_samples=28000)
 raw=read(R.PARENT/'run/CALIBRATION.json');cal=copy.deepcopy(raw)
 cal.update(configuration_sha256=digest(c),source_sha256=digest(R.source()),inherited=True,inherited_from=str(R.PARENT/'run/CALIBRATION.json'),
  original_receipt_sha256=sha(R.PARENT/'run/CALIBRATION.json'),inheritance_reason='Only output/budget/checkpoint cadence change; model, losses, LRs, data, calibration inputs and batch28 unchanged')
 cl=copy.deepcopy(read(R.PARENT/'run/cluster.json'));cl.update(master_port=29989,training_script=str(W/'worker.py'),training_config=str(R.RUN/'config.json'))
 atomic_json(R.RUN/'config.json',c,immutable=True);atomic_json(R.RUN/'cluster.json',cl,immutable=True);atomic_json(R.RUN/'preflight.json',pre,immutable=True);atomic_json(R.RUN/'CALIBRATION.json',cal,immutable=True)
 p=R.PARENT_STATE
 atomic_json(R.RUN/'CONTINUATION_ACCEPTED.json',dict(success=True,at=time.time(),configuration_sha256=digest(c),source=R.source(),
  parent_checkpoint_sha256=c['continuation']['parent_sha256'],parent_checkpoint_stat=dict(bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns),
  parent_latest_reports={str(s):sha(R.PARENT/'run/evaluation'/f'step_{s:06d}'/'REPORT.json')for s in(0,100)},
  parent_full_checkpoint_checked_receipt=sha(R.PARENT/'run/evaluation/step_000100/MODEL_EXPORT.json'),
  no_parameter_or_optimizer_reinitialization=True,no_rng_or_sample_schedule_reset=True,
  original_safety_calibration_inherited_not_remeasured=True,saves=[300,600,900,1000]))
 R.precheck(c,p)
 print('Continuation prepared:100 ->1000, saves300/600/900/1000',flush=True)

def launch(resume=None):
 import p2_whisper_event_feature_train31 as native
 from listen_cluster import launch as cluster_launch
 c=R.validate(read(R.RUN/'config.json'));cl=read(R.RUN/'cluster.json');T=R.activate();path=Path(resume)if resume else R.PARENT_STATE;T.precheck(c,path)
 with(W/'launch.lock').open('a')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  reports={}
  for attempt in range(12):
   missing=[n for n in cl['nodes']if not reports.get(n['name'],{}).get('gpus')]
   if not missing:break
   for r in native.selected_resources(dict(cl,nodes=missing))['nodes']:reports[r['node']]=r
   if any(not reports[n['name']].get('gpus')for n in cl['nodes']):time.sleep(min(2+attempt,8))
  if any(not reports.get(n['name'],{}).get('gpus')for n in cl['nodes']):raise RuntimeError('Node unreachable after resource retries')
  atomic_json(W/'RESOURCES.json',reports)
  if not all(r['selected_idle']for r in reports.values()):raise RuntimeError('Selected28 GPUs occupied; preserve other tasks')
  cmds=[]
  for i,n in enumerate(cl['nodes']):
   cmds.append(['env','PYTHONPATH='+str(R.PARENT/'isolation'),'CUDA_VISIBLE_DEVICES='+','.join(map(str,n['gpus'])),'PYTHONUNBUFFERED=1',
    *[k+'='+str(v)for k,v in cl['network'].items()],PY,'-u',str(W/'worker.py'),'node','--index',str(i),'--resume',str(path)])
  atomic_json(W/'CONTROLLER.json',dict(pid=os.getpid(),at=time.time(),resume=str(path),target=1000,saves=[300,600,900,1000]))
  cluster_launch(cl,cmds,W/('nodes_'+str(time.time_ns())))
 print(read(R.RUN/'status.json'),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','launch']);p.add_argument('--resume');a=p.parse_args();prepare()if a.action=='prepare'else launch(a.resume)
