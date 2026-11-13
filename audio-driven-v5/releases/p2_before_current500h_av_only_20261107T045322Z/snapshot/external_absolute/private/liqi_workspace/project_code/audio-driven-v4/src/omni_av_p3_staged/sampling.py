from dataclasses import replace
import torch

@torch.no_grad()
def guided_sample(renderer,prepared,negative,steps=30,video_cfg=3.,audio_cfg=7.,sigmas=None,progress=None):
 """Native LTX schedule, Euler, text+planning CFG; no STG/rescale/upscaler."""
 renderer.eval()
 if sigmas is None:
  from ltx_core.components.schedulers import LTX2Scheduler
  sigmas=LTX2Scheduler().execute(steps=steps)
 sigmas=sigmas.to(prepared.video.latent.device)
 if len(sigmas)!=steps+1 or abs(float(sigmas[0])-1.)>1e-6 or abs(float(sigmas[-1]))>1e-7 or not (sigmas[:-1]>sigmas[1:]).all():raise ValueError('Invalid sigma schedule')
 keep=~prepared.video_loss_mask;reference=prepared.video.latent.float()
 video=torch.where(keep[...,None],reference,torch.randn_like(prepared.video.latent).float());audio=torch.randn_like(prepared.audio.latent).float()
 for i in range(steps):
  sigma=torch.full_like(prepared.video.sigma,float(sigmas[i]));dt=float(sigmas[i+1]-sigmas[i])
  vi=replace(prepared.video,latent=video.to(prepared.video.latent.dtype),sigma=sigma,timesteps=torch.where(keep,0,sigma[:,None]).to(prepared.video.timesteps.dtype))
  ai=replace(prepared.audio,latent=audio.to(prepared.audio.latent.dtype),sigma=sigma,timesteps=sigma[:,None].expand_as(prepared.audio.timesteps))
  with torch.autocast('cuda',dtype=torch.bfloat16,enabled=video.is_cuda):
   pv,pa=renderer(video=vi,audio=ai,perturbations=None)
   if video_cfg!=1 or audio_cfg!=1:
    nv=replace(vi,context=negative['video'],context_mask=negative['mask']);na=replace(ai,context=negative['audio'],context_mask=negative['mask'])
    uv,ua=renderer(video=nv,audio=na,perturbations=None)
    pv=uv.float()+video_cfg*(pv.float()-uv.float());pa=ua.float()+audio_cfg*(pa.float()-ua.float())
  video=torch.where(keep[...,None],reference,video+dt*pv.float());audio=audio+dt*pa.float()
  if not torch.isfinite(video).all() or not torch.isfinite(audio).all():raise FloatingPointError('Nonfinite AV sample')
  if progress is not None and ((i+1)%5==0 or i+1==steps):progress(i+1,steps)
 if not torch.equal(video[keep],reference[keep]):raise ValueError('Reference first frame changed')
 return dict(video=video,audio=audio)
