"""Prepare, validate and explicitly run the short online joint-alignment trial."""
from pathlib import Path
import argparse,copy,fcntl,json,os,sys,time
ROOT=Path(__file__).resolve().parents[1];SELF=Path(__file__).resolve()
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from p2_stage2_joint_v2.config import build_config,validate
from p2_listen.contracts import read,sha,atomic_json


def layout(cluster):
 from p2_current500h_component_ablations import layout as original
 return original(cluster)


def launcher():
 import p2_whisper_event_feature_train31 as native
 native.SELF=SELF;native.layout=layout;return native


def commands(c,resume=None):
 out=Path(c['output'])
 if resume and Path(resume).resolve().parent!=out.resolve():raise ValueError('Resume only this joint run')
 result=launcher().commands(read(out/'cluster.json'),out/'config.json',resume,out/'rank_logs')
 # This CLI has no inherited arbitrary stop-after; training exits at0/100/300.
 return result


def prepare(a):
 from p2_stage2_joint_v2.policy import DEFAULT
 out=a.output.resolve()
 if out.exists():raise FileExistsError('Use a new independent joint output')
 validation=a.validation.resolve();receipt=read(validation/'VALIDATION_SELECTED.json')
 if not receipt['success']or receipt['count']!=64 or receipt['test_used']:raise ValueError('Require independent64dev validation receipt')
 c=build_config(a.parent_config,a.checkpoint,a.adapter,out,planning_weight=a.planning_weight)
 c['calibration']['samples']=a.calibration_samples
 out.mkdir(parents=True)
 atomic_json(out/'policy.json',DEFAULT,immutable=True)
 for key,name in [('planning_training_ids','planning_training_ids.json'),('planning_caption_hashes','planning_caption_hashes.json'),('validation_manifest','validation.json')]:
  c[key]=str(validation/name);c[key+'_sha256']=sha(validation/name)
 from p2_stage2_joint_v2.calibration import select
 selected=select(c);atomic_json(out/'calibration_manifest.json',selected,immutable=True)
 c.update(calibration_manifest=str(out/'calibration_manifest.json'),calibration_manifest_sha256=sha(out/'calibration_manifest.json'))
 c.update(policy=str(out/'policy.json'),policy_sha256=sha(out/'policy.json'),validation_receipt=str(validation/'VALIDATION_SELECTED.json'),validation_receipt_sha256=sha(validation/'VALIDATION_SELECTED.json'))
 validate(c)
 cluster=copy.deepcopy(read(ROOT/'experiments/p2_current_component_ablations31_20261108/no_whisper/cluster.json'))
 cluster.update(master_port=29865,training_script=str(SELF),training_config=str(out/'config.json'))
 atomic_json(out/'config.json',c,immutable=True);atomic_json(out/'cluster.json',cluster,immutable=True)
 atomic_json(out/'COMMANDS.json',commands(c),immutable=True)
 atomic_json(out/'PREPARED.json',dict(success=True,training_started=False,at=time.time(),world=31,reserved_gpu='node11:7',milestones=[0,100,300],parent_is_explicit=True))
 return dict(prepared=True,output=str(out),training_started=False)


def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('command',choices=['prepare','preflight','commands','train','node','launch','select-validation','evaluate','decide'])
 p.add_argument('--output',type=Path);p.add_argument('--config',type=Path);p.add_argument('--parent-config',type=Path);p.add_argument('--checkpoint',type=Path);p.add_argument('--adapter',type=Path)
 p.add_argument('--validation',type=Path);p.add_argument('--planning-seal',type=Path);p.add_argument('--selection',type=Path)
 p.add_argument('--resume',type=Path);p.add_argument('--logs',type=Path);p.add_argument('--node-index',type=int);p.add_argument('--attempt',type=Path);p.add_argument('--step',type=int,choices=[0,100,300]);p.add_argument('--device',default='cuda:0')
 p.add_argument('--planning-weight',type=float,default=0.0001);p.add_argument('--calibration-samples',type=int,choices=range(8,17),default=16)
 a=p.parse_args()
 def need(*names):
  for name in names:
   if getattr(a,name)is None:p.error('--'+name.replace('_','-')+' required')
 if a.command=='select-validation':
  need('planning_seal','selection','output')
  from p2_stage2_joint_v2.validation import select
  result=select(a.planning_seal,a.selection,a.output)
 elif a.command=='prepare':
  need('output','parent_config','checkpoint','adapter','validation');result=prepare(a)
 else:
  need('config');c=validate(read(a.config));out=Path(c['output'])
  if a.command=='node':return launcher().node(c,a.node_index,a.resume,a.attempt)
  if a.command=='commands':result=commands(c,a.resume)
  elif a.command=='preflight':
   import torch
   torch.set_num_threads(2)
   from p2_stage2_joint_v2.training import preflight
   v=preflight(c);result=dict(success=v['success'],samples=v['samples'],canary_reads=len(v['canary_reads']))
  elif a.command=='train':
   from p2_stage2_joint_v2.training import train
   result=train(c,a.resume)
  elif a.command=='evaluate':
   need('step')
   from p2_stage2_joint_v2.evaluation import evaluate
   result=evaluate(c,a.step,a.device)
  elif a.command=='decide':
   need('step')
   from p2_stage2_joint_v2.evaluation import decision
   result=decision(c,a.step)
  else:
   need('logs')
   from listen_cluster import launch
   if (out/'STOP').exists():raise RuntimeError('Preserve explicit STOP')
   if (out/'run.json').exists()and not a.resume:raise ValueError('Existing trial requires explicit resume')
   from p2_stage2_joint_v2.training import precheck
   precheck(c,a.resume)
   with (out/'launch.lock').open('a')as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    resources=launcher().selected_resources(read(out/'cluster.json'));atomic_json(out/'RESOURCE_PRECHECK.json',resources)
    if not resources['selected_all_idle']:raise RuntimeError('Selected31 GPUs are busy; preserve existing tasks')
    launch(read(out/'cluster.json'),commands(c,a.resume),a.logs)
   result=read(out/'status.json')
 print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
