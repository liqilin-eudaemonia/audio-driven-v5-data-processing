"""Bounded validation-only feasibility check; local warmup adaptation explicit."""
import os,sys,time,signal,traceback,shutil,json,gc
from pathlib import Path
import numpy as np,torch
ROOT=Path('/private/liqi_workspace/project_code/audio-driven-v5');SOURCE=ROOT/'experiments/p2_l2l_reliable_retrain_20261107';W=ROOT/'experiments/p2_l2l_bounded_predictor_check_20261107'
sys.path.insert(0,str(SOURCE/'code_snapshot'))
from l2l_reliable_core import *
from l2l_reliable_data import load,prepare
import l2l_reliable as runner
START=time.monotonic();PHASE='setup'
def stop(signum,frame):
 if PHASE=='training':runner.interrupted()
 else:raise TimeoutError('Bounded20minute diagnostic limit reached')
signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGALRM,stop);signal.alarm(1200)
try:
 backend();prepare(SOURCE)
 for name in ['CODE_FROZEN.json','DATA_COMPLETE.json','data','split.json','source_features.json','code_snapshot']:
  dst=W/name
  if not dst.exists():dst.symlink_to(SOURCE/name,target_is_directory=(SOURCE/name).is_dir())
 vqfolder=W/'seed_137/vq';vqfolder.mkdir(parents=True,exist_ok=True)
 dst=vqfolder/'best.pth'
 if not dst.exists():dst.symlink_to(SOURCE/'seed_137/vq/best.pth')
 state=torch.load(dst,map_location='cpu',weights_only=False);val=state['selection']['best'];init=state['selection']['initial']
 checks=dict(selected_after_initialization=state['step']>0,improved_from_initial=val['total']<init['total'],beats_mean_by_five_percent=val['reconstruction']<=val['train_mean_baseline']*.95,multiple_codes=val['unique_codes']>=2)
 assert all(checks.values())
 save(vqfolder/'COMPLETE.json',dict(passed=True,diagnostic_only=True,checks=checks,best_step=state['step'],checkpoint_sha256=sha(dst),validation=val))
 del state
 runner.POLICY=dict(runner.POLICY,seeds=[137],max_steps=10000,min_steps=3000,val_every=500,patience=6,warmup=1000)
 original_factory=runner.factories
 def adjusted_factory(repo,stage,seed):
  model,opt,cfg,vqcfg=original_factory(repo,stage,seed)
  if stage=='predictor':opt.n_warmup_steps=1000
  return model,opt,cfg,vqcfg
 runner.factories=adjusted_factory
 save(W/'PROTOCOL.json',dict(diagnostic_only=True,formal_baseline=False,test_opened=False,source_vq=str(dst.resolve()),vq_step=1000,vq_training_unchanged=True,
   predictor_warmup=1000,upstream_predictor_warmup=100000,optimizer='upstream Adam and ScheduledOptim; only warmup adapted',policy=runner.POLICY,
   walltime_limit_seconds=1200,maximum_updates=10000,automatic_test=False,automatic_replicates=False,code_sha256=sha(Path(__file__))))
 data,stats=load(SOURCE);repo=W/'code_snapshot/l2l_src';vq,_=runner.load_best(W,repo,137,'vq');tokens=tokenize(vq,data);del vq;gc.collect();torch.cuda.empty_cache()
 PHASE='training';receipt=runner.stage_run(W,repo,data,stats,137,'predictor',tokens);del tokens
 if runner.STOP:raise TimeoutError('Stopped at safe update boundary under20minute bound')
 PHASE='source_only_validation_rollout';runner.validation_rollout(W,repo,stats,137)
 gate=read(W/'seed_137/ROLLOUT_GATE.json')
 save(W/'RESULTS.json',dict(passed=receipt['passed'] and gate['passed'],diagnostic_only=True,formal_baseline=False,test_opened=False,
    predictor=receipt,rollout_gate=gate,elapsed_seconds=time.monotonic()-START,learning_protocol='Local adaptation, not original100000warmup'))
except SystemExit:
 save(W/'STOPPED.json',dict(phase=PHASE,elapsed_seconds=time.monotonic()-START,diagnostic_only=True,test_opened=False))
except Exception:
 save(W/'FAILED.json',dict(phase=PHASE,traceback=traceback.format_exc(),elapsed_seconds=time.monotonic()-START,diagnostic_only=True,test_opened=False));raise
finally:
 signal.alarm(0)
 save(W/'EXITED.json',dict(phase=PHASE,elapsed_seconds=time.monotonic()-START,gpu_released_on_exit=True,diagnostic_only=True))
