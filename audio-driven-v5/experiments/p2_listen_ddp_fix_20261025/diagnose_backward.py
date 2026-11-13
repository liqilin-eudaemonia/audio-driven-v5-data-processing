"""Same-state real AV backward: repeatability, audit side effects and loss scaling."""
import os,sys,json,copy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
import torch
from p2_listen.config import load,activate,extractor_config
from p2_listen.contracts import EVENTS,read,atomic_json
from p2_listen.dataset import load_observations,load_targets
from p2_listen.features import load_training_sample,load_text
from p2_listen.annotations import mask_classes
from p2_listen.model import ListeningExtractor
from p2_listen.training import indices
from p2_listen.events import normalized_event_loss
from p2_listen.runtime import av_loss
c=load(ROOT/'experiments/p2_listen_weak_background_gpu_fp32_20261025/config.json');cfg=extractor_config(c)
from omni_av_p2.ltx import load_renderer,append_conditions
from omni_av.ltx_runtime import with_text
seed=20261027;device='cuda:0';torch.set_num_threads(4);torch.cuda.set_device(0)
rows=load_observations(c);targets=load_targets(c,rows)
run=read(Path(c['output'])/'runs/s0_20261027/E1/run.json');weights={k:run['statistics'][k]['pos_weight'] for k in EVENTS}
chosen=indices(len(rows),1,8,seed)
samples=[load_training_sample(c,rows[i],targets[rows[i]['sample_id']],device) for i,_ in chosen]
samples=[(*x[:4],mask_classes(x[4],run['K_train'])) for x in samples]
counts={k:torch.tensor(sum(int(s[4][k]['known'].sum()) for s in samples),device=device,dtype=torch.float32) for k in EVENTS}
torch.manual_seed(seed);model=ListeningExtractor(cfg,c['events']['head_width']).to(device);model.train()
renderer,strategy,sampler=load_renderer(cfg,device);renderer.train();text=load_text(c,device)['positive']
condition,layout,teacher,av,labels=samples[0];del samples
ordinal=chosen[0][1]

def run_backward(scale,audit,deterministic=False):
 torch.use_deterministic_algorithms(deterministic)
 model.zero_grad(set_to_none=True)
 torch.manual_seed(seed+ordinal);generator=torch.Generator(device=device).manual_seed(seed+1000000+ordinal)
 with torch.autocast('cuda',dtype=torch.bfloat16):
  value=model(condition,layout,teacher,labels,'probe_only',weights,generator)
  prepared=strategy.prepare_training_inputs(with_text(av['av'],text),sampler)
  lav=av_loss(renderer,append_conditions(prepared,value['u']),av['support'])
  feat=sum(value['features'].values());event=normalized_event_loss(value['event_sums'],counts)
  loss=(lav/8+feat/8+.05*event)*scale
 if audit:
  inputs=[value['u'].video,value['u'].audio]
  for objective in (lav,feat,event,*value['event_sums'].values()):
   torch.autograd.grad(objective,inputs,retain_graph=True,allow_unused=True)
 loss.backward();torch.cuda.synchronize()
 grads={n:p.grad.detach().float().cpu()/scale for n,p in model.named_parameters() if p.grad is not None}
 return grads,dict(loss=float(loss.detach())/scale,av=float(lav.detach()),event=float(event.detach()),audit=audit,scale=scale,deterministic=deterministic,
                   norm=float(torch.linalg.vector_norm(torch.stack([g.norm() for g in grads.values()]))))

def compare(a,b):
 errors=sorted([(k,float((a[k]-b[k]).abs().max())) for k in a],key=lambda x:x[1],reverse=True)
 diff2=sum(float((a[k]-b[k]).double().square().sum()) for k in a);ref2=sum(float(a[k].double().square().sum()) for k in a)
 return dict(max_abs=errors[0][1],relative_l2=(diff2/max(ref2,1e-30))**.5,largest=errors[:8])
report={}
for deterministic in (False,True):
 tag='deterministic' if deterministic else 'default'
 try:
  ref,info=run_backward(1,False,deterministic);report[tag]={'reference':info}
  for name,scale,audit in [('repeat',1,False),('audit',1,True),('scale2',2,False)]:
   g,i=run_backward(scale,audit,deterministic);report[tag][name]=dict(info=i,comparison=compare(ref,g));del g
   atomic_json(OUT/'backward_diagnosis.json',report);print(tag,name,report[tag][name],flush=True)
  del ref
 except Exception as e:
  report[tag]={'error':repr(e)};atomic_json(OUT/'backward_diagnosis.json',report);raise
atomic_json(OUT/'backward_diagnosis.json',report)
