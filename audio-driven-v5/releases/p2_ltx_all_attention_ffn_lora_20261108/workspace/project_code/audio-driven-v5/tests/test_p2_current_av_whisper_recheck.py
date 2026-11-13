"""Independent behavioral and two-process reduction checks of the actual new arm."""
import copy,socket,json
from dataclasses import replace
from pathlib import Path
import pytest,torch
import torch.distributed as dist
import torch.multiprocessing as mp
from test_p2_current_av_whisper import build,CONFIG
from p2_current_av_whisper.model import av_objective
from p2_whisper_av.execution import accumulator

def varied(b,rank):
 hidden=b['whisper'].hidden.clone();hidden[...,0]+=rank*.3
 return dict(b,whisper=replace(b['whisper'],hidden=hidden))

def distributed_worker(rank,address,folder):
 torch.set_num_threads(2);torch.manual_seed(314159)
 c,cfg,m,b,s,sa=build();m.renderer.set_gradient_checkpointing(True)
 dist.init_process_group('gloo',init_method=address,rank=rank,world_size=2)
 try:
  o=torch.optim.AdamW(m.groups(),weight_decay=0.)
  for step in range(2):
   o.zero_grad(set_to_none=True);torch.manual_seed(1000+step*2+rank)
   av_objective(m(varied(b,rank),s,sa),c,{},1,2)[0].backward()
   a=accumulator(m.parameters(),1,c['execution']);a.add();a.finish(2)
   torch.nn.utils.clip_grad_norm_([p for p in m.parameters()if p.requires_grad],1.,error_if_nonfinite=True)
   o.step()
  torch.save({'model':m.state_dict(),'optimizer':o.state_dict(),'condition':b['condition'].hidden,'speaker':b['whisper'].hidden,'latents':b['av']['av']['latents']['latents']},Path(folder)/f'rank{rank}.pt')
 finally:dist.destroy_process_group()

def test_two_real_processes_match_single_process_average_and_adam(tmp_path):
 torch.manual_seed(314159)
 c,cfg,m,b,s,sa=build();m.renderer.set_gradient_checkpointing(True)
 params=[p for p in m.parameters()if p.requires_grad];o=torch.optim.AdamW(m.groups(),weight_decay=0.)
 for step in range(2):
  grads=[]
  for rank in range(2):
   o.zero_grad(set_to_none=True);torch.manual_seed(1000+step*2+rank)
   av_objective(m(varied(b,rank),s,sa),c,{},1,2)[0].backward()
   grads.append([None if p.grad is None else p.grad.detach().double().clone()for p in params])
  for i,p in enumerate(params):
   used=[g[i]for g in grads if g[i] is not None];p.grad=sum(used).to(p.dtype)if used else None
  torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);o.step()
 with socket.socket()as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
 mp.spawn(distributed_worker,args=(f'tcp://127.0.0.1:{port}',str(tmp_path)),nprocs=2,join=True)
 differences=[]
 for rank in range(2):
  actual=torch.load(tmp_path/f'rank{rank}.pt',weights_only=True)
  assert torch.equal(actual['condition'],b['condition'].hidden) and torch.equal(actual['speaker'],b['whisper'].hidden) and torch.equal(actual['latents'],b['av']['av']['latents']['latents'])
  differences.append({'rank':rank,'max_parameter_abs_error':max(float((actual['model'][k]-v).abs().max())for k,v in m.state_dict().items()),'max_adam_abs_error':max(float((actual['optimizer']['state'][k][n]-v).abs().max())for k,state in o.state_dict()['state'].items()for n,v in state.items())})
  for k,v in m.state_dict().items():torch.testing.assert_close(actual['model'][k],v,rtol=2e-5,atol=2e-7,msg=k)
  for k,state in o.state_dict()['state'].items():
   for name,v in state.items():torch.testing.assert_close(actual['optimizer']['state'][k][name],v,rtol=2e-5,atol=2e-7)

 (tmp_path/'comparison.json').write_text(json.dumps({'success':True,'identical_inputs':True,'ranks':differences,'world':2,'backend':'CPU Gloo','rtol':2e-5,'atol':2e-7}))

def test_new_arm_speaker_content_affects_both_av_outputs_after_learning():
 from omni_av.ltx_runtime import with_text
 c,cfg,m,b,s,sa=build();o=torch.optim.AdamW(m.groups(),weight_decay=0.)
 for _ in range(3):
  o.zero_grad(set_to_none=True);m(b,s,sa)['av'].backward();o.step()
 m.eval();u=m.connector(b['condition']);torch.manual_seed(44)
 prepared=s.prepare_training_inputs(with_text(b['av']['av'],b['text']['positive']),sa)
 def output(speaker):
  p=m.prepare(prepared,u,speaker);return m.renderer(video=p.video,audio=p.audio,perturbations=None)
 original=output(b['whisper']);changed=output(varied(b,2)['whisper'])
 assert all(not torch.equal(x,y)for x,y in zip(original,changed))
 hidden=b['whisper'].hidden.clone();hidden[:,~b['whisper'].valid[0]]=10000
 masked=output(replace(b['whisper'],hidden=hidden))
 assert all(torch.equal(x,y)for x,y in zip(original,masked))

def test_cold_cli_training_reaches_gpu_gate_without_import_or_identity_failure():
 import subprocess,os,sys
 root=CONFIG.parents[2]
 r=subprocess.run([sys.executable,str(root/'scripts/p2_current500h_av_whisper.py'),'train','--config',str(CONFIG)],
  env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),cwd=root,text=True,capture_output=True,timeout=60)
 assert r.returncode!=0 and 'Real LTX training requires GPU'in r.stderr,r.stderr
