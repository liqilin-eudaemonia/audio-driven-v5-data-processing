"""Benchmark candidates: same Euler/CFG equations, batch dimensions only."""
from dataclasses import fields,replace
from types import SimpleNamespace
import torch
import torch.nn.functional as F


def cat_modality(values):
 result={}
 for field in fields(values[0]):
  name=field.name;xs=[getattr(x,name) for x in values]
  if all(x is None for x in xs):result[name]=None
  elif isinstance(xs[0],bool):
   assert all(x==xs[0] for x in xs);result[name]=xs[0]
  else:
   assert all(isinstance(x,torch.Tensor) and x.shape[0]==1 or isinstance(x,torch.Tensor) and x.shape[0]==xs[0].shape[0] for x in xs)
   if name in ['context','context_mask']:
    length=max(x.shape[1] for x in xs)
    xs=[F.pad(x,(0,0,0,length-x.shape[1])) if x.ndim==3 else F.pad(x,(0,length-x.shape[1]),value=False) for x in xs]
   result[name]=torch.cat(xs,dim=0)
 return type(values[0])(**result)


def pack(prepared,negative):
 batch=SimpleNamespace(video=cat_modality([p.video for p in prepared]),audio=cat_modality([p.audio for p in prepared]),
     video_loss_mask=torch.cat([p.video_loss_mask for p in prepared]),single_inputs=prepared)
 neg={}
 for mod in ['video','audio']:
  length=max(x[mod].shape[1] for x in negative)
  neg[mod]=torch.cat([F.pad(x[mod],(0,0,0,length-x[mod].shape[1])) for x in negative])
  neg[mod+'_mask']=torch.cat([F.pad(x[mod+'_mask'],(0,length-x[mod+'_mask'].shape[1]),value=False) for x in negative])
 return batch,neg


@torch.inference_mode()
def sample(renderer,prepared,negative,steps,video_cfg,audio_cfg,seed=137,combine_cfg=False):
 from ltx_core.components.schedulers import LTX2Scheduler
 sigmas=LTX2Scheduler().execute(steps=steps).to(prepared.video.latent.device)
 keep=~prepared.video_loss_mask;reference=prepared.video.latent.float()
 videos=[];audios=[]
 # Match the existing per-sample RNG resets, including random values at pinned tokens.
 for original in getattr(prepared,'single_inputs',[prepared]):
  torch.manual_seed(seed)
  original_reference=original.video.latent.float();original_keep=~original.video_loss_mask
  videos.append(torch.where(original_keep[...,None],original_reference,torch.randn_like(original_reference)))
  audios.append(torch.randn_like(original.audio.latent).float())
 video=torch.cat(videos);audio=torch.cat(audios)
 renderer.eval()
 for i in range(steps):
  sigma=torch.full_like(prepared.video.sigma,float(sigmas[i]));dt=float(sigmas[i+1]-sigmas[i])
  vi=replace(prepared.video,latent=video.to(prepared.video.latent.dtype),sigma=sigma,
       timesteps=torch.where(keep,0,sigma[:,None]).to(prepared.video.timesteps.dtype))
  ai=replace(prepared.audio,latent=audio.to(prepared.audio.latent.dtype),sigma=sigma,
       timesteps=sigma[:,None].expand_as(prepared.audio.timesteps))
  nv=replace(vi,context=negative['video'],context_mask=negative['video_mask'])
  na=replace(ai,context=negative['audio'],context_mask=negative['audio_mask'])
  with torch.autocast('cuda',dtype=torch.bfloat16):
   if combine_cfg:
    allv,alla=renderer(video=cat_modality([vi,nv]),audio=cat_modality([ai,na]),perturbations=None)
    pv,uv=allv.chunk(2);pa,ua=alla.chunk(2)
   else:
    pv,pa=renderer(video=vi,audio=ai,perturbations=None)
    uv,ua=renderer(video=nv,audio=na,perturbations=None)
   pv=uv.float()+video_cfg*(pv.float()-uv.float());pa=ua.float()+audio_cfg*(pa.float()-ua.float())
  video=torch.where(keep[...,None],reference,video+dt*pv.float());audio=audio+dt*pa.float()
  if not torch.isfinite(video).all() or not torch.isfinite(audio).all():raise FloatingPointError('Nonfinite generated AV')
 return dict(video=video,audio=audio)
