import pathlib,sys,json,time,statistics
R=pathlib.Path('/private/liqi_workspace/project_code/audio-driven-v5');sys.path[:0]=[str(R/'src'),str(R/'scripts')]
import torch
from baseline_training.glare_cached import Registry
from baseline_training.glare_execution_fast import FastCachedFLOAT
from baseline_training.glare_batched import BatchedCachedFLOAT,length_batches
from baseline_training.training import build,seed,set_train
out=pathlib.Path(__file__).parent;torch.set_num_threads(2);seed(137)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
d=Registry(R/'experiments/glare_cache_28gpu_20261110/production_gpu_fer/TRAINING_CACHES_LATEST.json')
ids=torch.randperm(len(d),generator=torch.Generator().manual_seed(137))[:768].tolist();data=[d[i] for i in ids]
m,_=build('glare','cuda:0');m.opt.reaction_loss_weight=.05;set_train(m,'glare')
params=[p for p in m.parameters() if p.requires_grad]
results=[]
for limit in [1,8,16,1]:
 wrap=BatchedCachedFLOAT(m);opt=torch.optim.AdamW(params,lr=5e-4,weight_decay=1e-5);scaler=torch.cuda.amp.GradScaler()
 times=[];groupcounts=[]
 for update in range(3):
  vals=data[update*256:(update+1)*256];groups=list(length_batches(vals,limit));groupcounts.append(len(groups))
  torch.cuda.synchronize();t=time.perf_counter();opt.zero_grad(set_to_none=True);losses=[]
  with torch.autocast('cuda',dtype=torch.float16):
   for group in groups:
    loss=wrap(group)['loss']
    with torch.autocast('cuda',enabled=False):scaler.scale(loss*len(group)/256).backward()
    losses.append(loss.detach())
  assert torch.isfinite(torch.stack(losses)).all()
  scaler.unscale_(opt);gn=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
  scaler.step(opt);scaler.update();torch.cuda.synchronize();times.append(time.perf_counter()-t)
 r=dict(limit=limit,seconds=times,forward_calls=groupcounts,peak_gib=torch.cuda.max_memory_allocated()/1024**3)
 results.append(r);print(json.dumps(r),flush=True)
# All-zero masks and unequal observed counts must retain equal pair loss weighting.
masks=torch.tensor([[[True],[False]],[[True],[True]]]);pred=torch.tensor([[[1.],[100.]],[[2.],[2.]]],device='cuda');target=torch.zeros_like(pred)
m.reaction_loss.mask=masks.to('cuda');actual=m.reaction_loss(pred,target);assert torch.allclose(actual,torch.tensor(1.,device='cuda'))
(out/'RESULT.json').write_text(json.dumps(dict(results=results,ids=ids,weight=.05,no_padding=True,pair_weighting_check=True),indent=2)+'\n')
