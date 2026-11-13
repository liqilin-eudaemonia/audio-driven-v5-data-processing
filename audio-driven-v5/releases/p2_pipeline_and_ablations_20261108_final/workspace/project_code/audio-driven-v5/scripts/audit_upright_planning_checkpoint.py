#!/usr/bin/env python3
"""Read-only CPU audit of an immutable native LoRA training checkpoint."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import socket
import time

ROOT=Path('/private/liqi_workspace/project_code/audio-driven-v5')
WORK=ROOT/'experiments/p2_planning_sft_upright_3b_gpu7_20261105'
FILES=('adapter_model.safetensors','adapter_config.json','optimizer.pt','scheduler.pt','trainer_state.json','rng_state.pth')

def read(p):return json.loads(Path(p).read_bytes())
def ensure(ok,msg):
 if not ok:raise ValueError(msg)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def fingerprint(p):
 s=p.stat();return (s.st_size,s.st_mtime_ns)

def audit(path,output):
 import numpy as np
 import torch
 from safetensors import safe_open
 torch.set_num_threads(1);torch.set_num_interop_threads(1)
 ensure(os.environ.get('CUDA_VISIBLE_DEVICES')=='','Audit must be CPU-only')
 path=path.resolve();ensure(path.parent==(WORK/'training').resolve() and path.name.startswith('checkpoint-'),'Not this run training checkpoint')
 step=int(path.name.split('-')[-1]);begin=time.time();before={}
 for name in FILES:
  before[name]=fingerprint(path/name);ensure(before[name][0]>0,'Empty '+name)
 state=read(path/'trainer_state.json');config=read(WORK/'config.json');identity=read(WORK/'training/run.json');data=read(WORK/'DATA_READY.json');seal=read(WORK/'data_seal.json');prep=read(WORK/'PREPARED.json')
 ensure(state['global_step']==step and state['max_steps']==data['total_optimizer_steps']==2752,'Step or full epoch budget differs')
 ensure(identity['smoke'] is False and identity['world_size']==1 and identity['train_count']==704368 and identity['config']==config,'Wrong training identity')
 ensure(identity['train_ids']==seal['splits']['train']['ids_sha256'],'Training membership changed')
 ensure(sha(WORK/'config.json')==prep['config_sha256'] and sha(WORK/'data_seal.json')==data['seal_sha256'],'Config/seal SHA differs')
 for filename,h in identity['code'].items():ensure(sha(WORK/'vendor/p2_planning_sft'/filename)==h,'Training source changed: '+filename)
 for filename,info in seal['files'].items():ensure(fingerprint(Path(seal['directory'])/filename)==(info['bytes'],info['mtime_ns']),'Sealed training data changed: '+filename)
 for filename,h in seal['reader_code'].items():
  p=Path(filename) if filename.startswith('/') else ROOT/'src'/filename
  ensure(sha(p)==h,'Sealed media reader changed: '+filename)
 logs=[r for r in state.get('log_history',[]) if 'loss' in r]
 ensure(logs and logs[-1]['step']==step and all(math.isfinite(float(r[k])) for r in logs for k in ('loss','grad_norm') if k in r),'Nonfinite/missing checkpoint loss history')
 ac=read(path/'adapter_config.json');ensure(ac['r']==config['lora']['rank']==32 and ac['lora_alpha']==config['lora']['alpha']==64 and ac['lora_dropout']==0.,'Wrong adapter configuration')
 trainable=read(WORK/'training/trainable.json');expected_names={n.replace('.default.','.') for n in trainable['names']};elements=0;dtypes=set()
 with safe_open(str(path/'adapter_model.safetensors'),framework='pt',device='cpu') as tensors:
  names=set(tensors.keys());ensure(names==expected_names and len(names)==504,'Adapter tensor membership differs')
  for name in names:
   t=tensors.get_tensor(name);ensure('.model.layers.' in name and ('lora_A' in name or 'lora_B' in name),'Unexpected perception/base weight')
   ensure(t.is_floating_point() and bool(torch.isfinite(t).all()),'Nonfinite adapter tensor: '+name)
   ensure(t.ndim==2 and (t.shape[0]==32 if 'lora_A' in name else t.shape[1]==32),'Wrong adapter rank: '+name)
   elements+=t.numel();dtypes.add(str(t.dtype))
 ensure(elements==trainable['parameters']==59867136,'Adapter parameter count differs')
 opt=torch.load(path/'optimizer.pt',map_location='cpu',weights_only=True);ids=[i for g in opt['param_groups'] for i in g['params']]
 ensure(len(ids)==len(set(ids))==len(opt['state'])==504 and set(ids)==set(opt['state']),'Incomplete optimizer parameters')
 moment_elements=0;optimizer_steps=set()
 for key,value in opt['state'].items():
  optimizer_steps.add(float(value['step']));a=value['exp_avg'];b=value['exp_avg_sq']
  ensure(a.shape==b.shape and bool(torch.isfinite(a).all()) and bool(torch.isfinite(b).all()) and bool((b>=0).all()),'Invalid Adam moments')
  moment_elements+=a.numel()
 ensure(optimizer_steps=={float(step)} and moment_elements==elements,'Optimizer state step/shape differs')
 scheduler=torch.load(path/'scheduler.pt',map_location='cpu',weights_only=True);ensure(scheduler['last_epoch']==step,'Scheduler step differs')
 ensure(len(scheduler['_last_lr'])==len(opt['param_groups']),'LR group mismatch')
 for lr,g in zip(scheduler['_last_lr'],opt['param_groups']):ensure(math.isfinite(lr) and lr>=0 and abs(lr-g['lr'])<1e-12,'Optimizer/scheduler LR differs')
 # This is our own native RNG payload, never an external/untrusted checkpoint.
 rng=torch.load(path/'rng_state.pth',map_location='cpu',weights_only=False)
 ensure({'python','numpy','cpu','cuda'}<=set(rng),'Missing random state')
 ensure(rng['cpu'].dtype==torch.uint8 and rng['cpu'].ndim==1 and rng['cpu'].numel()>0,'Invalid CPU RNG payload')
 cuda=rng['cuda'];cuda=[cuda] if isinstance(cuda,torch.Tensor) else list(cuda)
 ensure(cuda and all(t.dtype==torch.uint8 and t.ndim==1 and t.numel()>0 for t in cuda),'Invalid CUDA RNG payload')
 py0=random.getstate();np0=np.random.get_state();cpu0=torch.random.get_rng_state()
 try:random.setstate(rng['python']);np.random.set_state(rng['numpy']);torch.random.set_rng_state(rng['cpu'])
 finally:random.setstate(py0);np.random.set_state(np0);torch.random.set_rng_state(cpu0)
 ensure(not torch.cuda.is_initialized(),'Audit initialized a GPU')
 files={}
 for name in FILES:
  digest=sha(path/name);ensure(fingerprint(path/name)==before[name],'Checkpoint changed during audit: '+name)
  files[name]=dict(bytes=before[name][0],mtime_ns=before[name][1],sha256=digest)
 result=dict(passed=True,checkpoint=str(path),step=step,host=socket.gethostname(),adapter_tensors=504,adapter_parameters=elements,adapter_dtypes=sorted(dtypes),optimizer_states=len(opt['state']),optimizer_steps=sorted(optimizer_steps),scheduler_last_epoch=scheduler['last_epoch'],learning_rates=scheduler['_last_lr'],all_adapter_and_Adam_tensors_finite=True,python_numpy_cpu_rng_restore_checked=True,cuda_rng_payload_readable=True,no_GPU_context_created=True,training_membership_config_code_and_source_seal_checked=True,loss_history_entries=len(logs),last_logged_loss=logs[-1]['loss'],files=files,scope='CPU readback; no training, model update, CUDA state restore, or live checkpoint reload performed',seconds=time.time()-begin,at=time.time())
 output.parent.mkdir(parents=True,exist_ok=True);ensure(not output.exists(),'Preserve previous audit receipt');temp=output.with_name(output.name+'.tmp');temp.write_text(json.dumps(result,indent=2)+'\n');temp.replace(output);print(json.dumps(result),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args();audit(a.checkpoint,a.output)
