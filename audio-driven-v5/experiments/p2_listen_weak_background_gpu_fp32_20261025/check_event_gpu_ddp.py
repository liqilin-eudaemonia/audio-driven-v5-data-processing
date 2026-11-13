"""Fixed-state GPU event-gradient equivalence with real weak labels/time layouts.
Random U isolates the event objective from iterative mixed-precision AV updates.
"""
import copy,json,os,sys
from pathlib import Path
from types import SimpleNamespace
from contextlib import nullcontext
import torch
import torch.distributed as dist
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from p2_listen.config import load,activate
from p2_listen.contracts import EVENTS,read,atomic_json,digest
from p2_listen.dataset import load_observations
from p2_listen.features import load_input,folder,identity
from p2_listen.cache import load_part,tree_to
from p2_listen.annotations import mask_classes
from p2_listen.events import EventHeads,event_sums,normalized_event_loss
from p2_listen.training import indices

torch.set_num_threads(2)
rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK']);world=int(os.environ['WORLD_SIZE'])
assert world==2
torch.cuda.set_device(local);device=f'cuda:{local}';dist.init_process_group('nccl')
c=load(OUT/'config.json');activate(c);obs=load_observations(c)
run=read(OUT/'runs/s0_20261027/E1/run.json');enabled=run['K_train'];pos={k:run['statistics'][k]['pos_weight'] for k in EVENTS}
chosen=indices(len(obs),1,8,20261027)
samples=[]
for i,ordinal in chosen:
 o=obs[i];label=read(OUT/'labels/canonical'/(o['sample_id']+'.json'))
 _,layout=load_input(c,o,'cpu')
 targets=mask_classes(load_part(folder(c,'events',o['sample_id']),identity(c,'events',o,label=label)),enabled)
 samples.append((ordinal,o['sample_id'],tree_to(layout,device),tree_to(targets,device)))

def condition(ordinal,layout):
 g=torch.Generator(device=device).manual_seed(1000+ordinal)
 return SimpleNamespace(video=torch.randn(1,layout['video_valid'].shape[1],32,device=device,generator=g,requires_grad=True),
    audio=torch.randn(1,layout['audio_valid'].shape[1],16,device=device,generator=g,requires_grad=True),
    video_valid=layout['video_valid'],audio_valid=layout['audio_valid'])

results={}
for mode in ('probe_only','joint_event'):
 torch.manual_seed(137)
 raw=EventHeads(32,16,128).to(device);initial=copy.deepcopy(raw.state_dict())
 model=torch.nn.parallel.DistributedDataParallel(raw,device_ids=[local],broadcast_buffers=False)
 local_samples=samples[rank::world]
 counts={k:torch.tensor(sum(int(x[3][k]['known'].sum()) for x in local_samples),device=device,dtype=torch.float32) for k in EVENTS}
 for v in counts.values():dist.all_reduce(v)
 total=torch.zeros((),device=device);u_norms=[]
 for j,(ordinal,sid,layout,target) in enumerate(local_samples):
  u=condition(ordinal,layout)
  with model.no_sync() if j<len(local_samples)-1 else nullcontext():
   with torch.autocast('cuda',dtype=torch.bfloat16):
    logits=model(u,layout,mode);assert all(v.dtype==torch.float32 for v in logits.values())
    sums,_=event_sums(logits,target,pos);loss=normalized_event_loss(sums,counts,world)
   loss.backward()
  total+=loss.detach()/world
  u_norms.append([0. if getattr(u,m).grad is None else float(getattr(u,m).grad.norm()) for m in ('video','audio')])
 dist.all_reduce(total)
 ddp_grads={k:p.grad.detach().clone() for k,p in raw.named_parameters()}
 if mode=='probe_only':assert all(n==[0.,0.] for n in u_norms)
 if rank==0:
  ref=EventHeads(32,16,128).to(device);ref.load_state_dict(initial)
  ref_total=0.
  for ordinal,sid,layout,target in samples:
   u=condition(ordinal,layout)
   with torch.autocast('cuda',dtype=torch.bfloat16):
    sums,_=event_sums(ref(u,layout,mode),target,pos);loss=normalized_event_loss(sums,counts)
   loss.backward();ref_total+=float(loss.detach())
  errors={k:float((ddp_grads[k]-p.grad).abs().max()) for k,p in ref.named_parameters()}
  passed=all(torch.allclose(ddp_grads[k],p.grad,atol=1e-6,rtol=1e-5) for k,p in ref.named_parameters()) and abs(float(total)-ref_total)<1e-6
  results[mode]=dict(passed=passed,max_parameter_gradient_error=max(errors.values()),loss_error=abs(float(total)-ref_total),
      counts={k:int(v) for k,v in counts.items()},gradient_atol=1e-6,gradient_rtol=1e-5,loss_atol=1e-6,local_u_gradient_norms=u_norms)
  del ref
 del model,raw
 dist.barrier()
if rank==0:
 report=dict(passed=all(v['passed'] for v in results.values()),world=world,global_batch=8,precision='FP32 head inside BF16 autocast',
   real_sample_ids=[x[1] for x in samples],label_signatures=run['labels'],synthetic_U=True,AV_renderer_included=False,checks=results)
 atomic_json(OUT/'event_gpu_ddp_equivalence.json',report)
 print(json.dumps(report,indent=2))
 assert report['passed'],report
 dist.barrier()
else:dist.barrier()
dist.destroy_process_group()
