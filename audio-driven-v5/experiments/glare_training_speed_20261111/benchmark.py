import pathlib,sys,json,time,copy
R=pathlib.Path('/private/liqi_workspace/project_code/audio-driven-v5');sys.path[:0]=[str(R/'src'),str(R/'scripts')]
import torch
from baseline_training.glare_cached import Registry,CachedFLOAT
from baseline_training.glare_execution_fast import FastCachedFLOAT
from baseline_training.training import build,seed,set_train
out=pathlib.Path(__file__).parent
torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
seed(137)
d=Registry(R/'experiments/glare_cache_28gpu_20261110/production_gpu_fer/TRAINING_CACHES_LATEST.json')
g=torch.Generator().manual_seed(137);ids=torch.randperm(len(d),generator=g)[:256].tolist();data=[d[i] for i in ids]
model,_=build('glare','cuda:0');model.fmt.initialize_weights();set_train(model,'glare')
params=[p for p in model.parameters() if p.requires_grad]
initial={n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad}

def run(fast,steps):
 model.load_state_dict(initial,strict=False);seed(781)
 wrap=FastCachedFLOAT(model) if fast else CachedFLOAT(model)
 opt=torch.optim.AdamW(params,lr=1e-5,weight_decay=1e-5)
 times=[];losses=[]
 for step in range(steps):
  torch.cuda.synchronize();t=time.perf_counter();opt.zero_grad(set_to_none=True);pending=[];total=0.
  for j in range(8):
   loss=wrap(data[(step*8+j)%len(data)])['loss']
   if not fast:assert torch.isfinite(loss)
   (loss/8).backward()
   if fast:pending.append(loss.detach())
   else:total+=float(loss.detach())/8
  if fast:
   vals=torch.stack(pending)
   assert bool(torch.isfinite(vals).all())
   # Keep FP64 CPU summation exactly as the original reporting arithmetic.
   total=sum(vals.cpu().tolist())/8
  torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);opt.step()
  torch.cuda.synchronize();times.append(time.perf_counter()-t);losses.append(total)
 final={n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad}
 return dict(fast=fast,times=times,losses=losses),final
results=[];reference=None
for fast in [False,True,True,False]:
 r,s=run(fast,40)
 if reference is None:reference=s;ref_losses=r['losses']
 else:
  r['all_trainable_tensors_exact']=all(torch.equal(reference[k],s[k]) for k in s)
  r['max_parameter_abs_diff']=max(float((reference[k]-s[k]).abs().max()) for k in s)
  r['parameters_allclose']=all(torch.allclose(reference[k],s[k],atol=1e-6,rtol=1e-5) for k in s)
  r['losses_exact']=r['losses']==ref_losses
 results.append(r);print(json.dumps(r),flush=True)
(out/'RESULT.json').write_text(json.dumps(dict(results=results,ids=ids,production_unchanged=True),indent=2)+'\n')
