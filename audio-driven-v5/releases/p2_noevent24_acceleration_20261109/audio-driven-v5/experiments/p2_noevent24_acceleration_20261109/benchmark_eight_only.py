"""Eight-NIC-only repeated matched batches; reuse/diagnostic/fused changes disabled."""
import os,sys,time,json,csv,argparse,copy,gc
from pathlib import Path
W=Path(__file__).resolve().parent;ROOT=W.parents[1];SPLIT=ROOT/'experiments/p2_ltx_split24_8_batch31_20261109'
sys.path[:0]=[str(SPLIT),str(ROOT/'src'),str(ROOT/'scripts')]
import torch
import torch.distributed as dist

def main():
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--label',required=True);p.add_argument('--candidate',choices=['fused','reuse','diagnostics','combined'],default='fused');a=p.parse_args()
 rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);local=int(os.environ['LOCAL_RANK']);assert world==24
 torch.cuda.set_device(local);torch.set_num_threads(2);device=f'cuda:{local}'
 torch.cuda.set_per_process_memory_fraction(134*2**30/torch.cuda.get_device_properties(local).total_memory,local)
 from p2_whisper_av.bootstrap import activate_ltx
 activate_ltx()
 import control as C,runtime
 from p2_ltx_all_lora_noevent.config import load
 from p2_ltx_all_lora_noevent.data import Dataset,data_config,to_device
 from p2_ltx_all_lora_noevent.model import Model
 from p2_listen.config import extractor_config
 from p2_listen.numerics import configure_training_numerics
 from p2_listen.performance import configure_renderer
 from p2_whisper_av.execution import accumulator
 from p2_ltx_events.heads import objective
 from p2_whisper_event_feature.execution import CPUInputStream
 out=W/a.label;out.mkdir(exist_ok=True);configure_training_numerics();dist.init_process_group('nccl',device_id=torch.device(device),timeout=__import__('datetime').timedelta(seconds=180))
 def progress(**kw):
  if rank==0:C.write(out/'progress.json',dict(at=time.time(),**kw))
 def sync():torch.cuda.synchronize(device)
 def max_values(vals):
  x=torch.tensor(vals,dtype=torch.float64,device=device);dist.all_reduce(x,op=dist.ReduceOp.MAX);return x.tolist()
 try:
  c=load(C.path('noevent',True)/'config.json');dataset=Dataset(c);schedule=C.native('noevent','schedule').Schedule(c);cfg=extractor_config(data_config(c))
  progress(stage='load_checkpoint');saved=C.native('noevent','checkpoint').checked(Path(a.checkpoint));assert saved['identity']==C.read(Path(c['output'])/'run.json')
  start=saved['step'];progress(stage='load_model',resume_step=start)
  from omni_av_p2.ltx import load_renderer
  renderer,strategy,sampler=load_renderer(cfg,device);configure_renderer(renderer,'all');torch.manual_seed(c['training']['seed']);model=Model(c,cfg,renderer).to(device);model.train()
  with Path(c['sample_list']).open()as f:long_index,long_row=max(enumerate(csv.DictReader(f)),key=lambda z:float(z[1]['duration_s']))
  parameters=[p for p in model.parameters()if p.requires_grad];main_params=[p for g in model.groups()if g['name']!='events'for p in g['params']]
  results=[];reference=None
  for round_i,kind in enumerate(['warmup','baseline','baseline']):
   optimizer=None;gc.collect();torch.cuda.empty_cache();model.load_state_dict(saved['model'],strict=True)
   optimizer=torch.optim.AdamW(model.groups(),weight_decay=0.,fused=True if kind == 'fused' else None);state=copy.deepcopy(saved['optimizer']);optimizer.load_state_dict(state);del state
   if kind == 'fused':
    for g in optimizer.param_groups:g.update(fused=True,foreach=False)
    for parameter,state in optimizer.state.items():state['step']=state['step'].to(parameter.device)
   reuse=None;times=[];losses=[];torch.cuda.reset_peak_memory_stats(device)
   # Two real longest updates, then restore before matched production batches.
   pairs=schedule.indices(len(dataset.rows),start+1,31,c['training']['seed'],0,1)
   def requests():
    for offset in range(-2,6):
     full=[(long_index,10000000+offset*31+i)for i in range(31)]if offset<0 else schedule.indices(len(dataset.rows),start+1+offset,31,c['training']['seed'],0,1)
     yield offset,runtime.partition(full,rank,24)
   with CPUInputStream(dataset,requests(),2)as stream:
    for offset in range(-2,6):
     if offset==0:
      optimizer.zero_grad(set_to_none=True);optimizer.state.clear();model.load_state_dict(saved['model'],strict=True);optimizer.load_state_dict(copy.deepcopy(saved['optimizer']))
      if kind == 'fused':
       for g in optimizer.param_groups:g.update(fused=True,foreach=False)
       for parameter,state in optimizer.state.items():state['step']=state['step'].to(parameter.device)
     dist.barrier();sync();begin=time.monotonic();inputs=next(stream);wait=time.monotonic()-begin
     optimizer.zero_grad(set_to_none=True)
     if reuse is None:acc=accumulator(parameters,len(inputs['chosen']),c['execution'])
     else:acc=reuse
     if kind in ('reuse','combined'):reuse=acc
     compute=add_time=0.;av=0.
     for sample,(_,ordinal) in zip(inputs['samples'],inputs['chosen']):
      value=to_device(sample,device);torch.manual_seed(c['training']['seed']+ordinal);gen=torch.Generator(device=device).manual_seed(c['training']['seed']+1000000+ordinal)
      sync();t=time.monotonic()
      with torch.autocast('cuda',dtype=torch.bfloat16):
       values=model(value,{},gen,strategy,sampler);loss,_,_=objective(values,c,{},31,1)
      if not torch.isfinite(loss):raise ValueError('nonfinite loss')
      loss.backward();sync();compute+=time.monotonic()-t;av+=float(values['av'].detach())/31
      t=time.monotonic();acc.add();sync();add_time+=time.monotonic()-t
      del value,values,loss
     t=time.monotonic();acc.finish(world);sync();reduce_time=time.monotonic()-t
     t=time.monotonic()
     if kind in ('diagnostics','combined'):
      norms={g['name']:float(torch.stack([torch.linalg.vector_norm(p.grad,dtype=torch.float64).square()for p in g['params']if p.grad is not None]).sum().sqrt())for g in optimizer.param_groups}
     else:norms={g['name']:sum(float(p.grad.double().square().sum())for p in g['params']if p.grad is not None)**.5 for g in optimizer.param_groups}
     torch.nn.utils.clip_grad_norm_(main_params,c['training']['clip_norm'],error_if_nonfinite=True);sync();diagnostic=time.monotonic()-t
     t=time.monotonic();optimizer.step();sync();adam=time.monotonic()-t;elapsed=time.monotonic()-begin
     timing=max_values([elapsed,wait,compute,add_time,reduce_time,diagnostic,adam])
     x=torch.tensor(av,device=device,dtype=torch.float64);dist.all_reduce(x)
     if offset>=0:times.append(timing);losses.append(float(x))
     progress(stage='benchmark',round=round_i,kind=kind,offset=offset,timing=timing)
     del inputs,acc
   peak=max_values([torch.cuda.max_memory_reserved(device)/2**30])[0]
   if rank==0:
    torch.set_num_threads(8)
    actual={n:p.detach().cpu()for n,p in model.named_parameters()if p.requires_grad}
    # Optimizer keys must use parameter order rather than object addresses.
    opt={str(k)+'/'+n:v.detach().to(device='cpu',copy=True)for k,s in optimizer.state_dict()['state'].items()for n,v in s.items()if torch.is_tensor(v)}
    if reference is None:reference=(actual,opt,losses)
    diffs={};ok=True
    for tag,now,ref in [('parameters',actual,reference[0]),('adam',opt,reference[1])]:
     maximum=max(float((now[k]-ref[k]).abs().max())for k in ref);equal=all(torch.equal(now[k],ref[k])for k in ref)
     close=all(torch.allclose(now[k],ref[k],rtol=1e-5,atol=1e-8 if tag=='parameters'else 1e-12)for k in ref)
     diffs[tag]=dict(max_abs=maximum,bitwise_equal=equal,close=close);ok &=close
    row=dict(kind=kind,round=round_i,timings=times,timing_columns=['total','input','forward_backward','accumulate','reduce','diagnostics_clip','adam'],losses=losses,peak_gib=peak,comparison=diffs,accepted_numerics=ok,mean_seconds=sum(t[0]for t in times)/len(times))
    results.append(row);C.write(out/'RESULTS.json',dict(results=results,checkpoint=a.checkpoint,global_batch=31,formal_updates=0))
   torch.set_num_threads(2)
   optimizer.zero_grad(set_to_none=True);optimizer.state.clear();reuse=None
  progress(stage='complete')
 finally:dist.destroy_process_group()
if __name__=='__main__':main()
