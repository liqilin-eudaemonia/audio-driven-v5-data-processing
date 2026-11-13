#!/usr/bin/env python3
"""Disposable real-model execution-policy benchmark; no production updates."""
from pathlib import Path
import argparse,sys,json,time,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
def main():
 import torch
 from p2_listen.config import extractor_config
 from p2_listen.contracts import atomic_json,EVENTS
 from p2_matrix16.config import load
 from p2_matrix16.model import MatrixModel
 from p2_matrix16.training import objective
 from p2_legacy_qwen_lora import backend
 from p2_listen.numerics import configure_training_numerics
 from p2_listen.performance import BucketGradientAccumulator,configure_renderer
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--policy',required=True);p.add_argument('--output',required=True);p.add_argument('--memory',type=float,default=128);p.add_argument('--max-duration',type=float,default=100);p.add_argument('--qwen-checkpoint-off',action='store_true');a=p.parse_args()
 c=load(a.config);d=backend.data_config(c);cfg=extractor_config(d);configure_training_numerics()
 from omni_av_p2.ltx import load_renderer,append_conditions
 from omni_av_p2.data import load_omni
 from omni_av.ltx_runtime import with_text
 from p2_listen.runtime import av_loss
 torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(a.memory*2**30/torch.cuda.get_device_properties(0).total_memory)
 rows=[]
 try:
  dataset=backend.Dataset(c);ordered=sorted((i for i in range(len(dataset.rows)) if dataset.rows[i]['requested_duration_s']<=a.max_duration),key=lambda i:dataset.rows[i]['requested_duration_s']);longest=ordered[-1];typical=ordered[len(ordered)//2]
  renderer,strategy,sampler=load_renderer(cfg,'cuda:0');__import__("p2_legacy_execution").configure(renderer,a.policy)
  runtime,processor=load_omni(cfg,'cuda:0');torch.manual_seed(c['training']['seed']);model=MatrixModel(c,cfg,renderer,runtime,processor).to('cuda:0');backend.initialize_stage2(Path(c['legacy_parent'])/'step_001268.pt',model,c,backend.data_identity(c));model.train()
  if a.qwen_checkpoint_off:runtime.thinker.model.gradient_checkpointing_disable()
  optimizer=torch.optim.AdamW(model.groups(),weight_decay=0.)
  def av(u,pack,text):return av_loss(renderer,append_conditions(strategy.prepare_training_inputs(with_text(pack['av'],text),sampler),u),pack['support'],return_components=True)
  for iteration,index in enumerate([longest]*3+[typical]*3):
   bundle=dataset.sample(index,'cuda:0');torch.manual_seed(91000+iteration);gen=torch.Generator(device='cuda:0').manual_seed(92000+iteration)
   optimizer.zero_grad(set_to_none=True);torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter();acc=BucketGradientAccumulator(model.parameters(),32)
   with torch.autocast('cuda',dtype=torch.bfloat16):
    values=model(bundle,{k:dataset.parent['statistics'][k]['pos_weight'] for k in EVENTS},gen,av);loss,_,_=objective(values,c,values['event_counts'],1,1)
   if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
   loss.backward();acc.add();acc.finish(1)
   main=[p for g in optimizer.param_groups if g['name']!='events' for p in g['params']]
   norm=float(torch.nn.utils.clip_grad_norm_(main,1.,error_if_nonfinite=True));torch.nn.utils.clip_grad_norm_(model.event_heads.parameters(),1.,error_if_nonfinite=True)
   optimizer.step();torch.cuda.synchronize();seconds=time.perf_counter()-start
   h=hashlib.sha256()
   for name,param in model.named_parameters():h.update(name.encode());h.update(param.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
   item=dict(iteration=iteration,sample_id=dataset.rows[index]['sample_id'],duration=dataset.rows[index]['requested_duration_s'],loss=float(loss.detach()),gradient_norm=norm,seconds=seconds,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,parameter_sha256=h.hexdigest());rows.append(item);print(json.dumps(item),flush=True)
   del bundle,values,loss,acc
  result=dict(success=True,policy=a.policy,memory_cap_gib=a.memory,rows=rows,qwen_checkpointing=not a.qwen_checkpoint_off,formal_training=False)
 except torch.cuda.OutOfMemoryError as e:result=dict(success=False,policy=a.policy,rows=rows,error='CUDA OOM',detail=str(e)[-1000:],formal_training=False)
 atomic_json(a.output,result);print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)
if __name__=='__main__':main()
