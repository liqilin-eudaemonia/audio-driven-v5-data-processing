"""Inference hooks: retain Qwen U and Whisper on both CFG paths."""
from dataclasses import replace

import torch


@torch.no_grad()
def sample(model, prepared, u, speaker, negative, **sampling_kwargs):
    model.eval()
    positive = model.prepare(prepared,u,speaker)
    negative_input = replace(prepared,
        video=replace(prepared.video,context=negative['video'],context_mask=negative['mask']),
        audio=replace(prepared.audio,context=negative['audio'],context_mask=negative['mask']))
    unconditional = model.prepare(negative_input,u,speaker)
    # Audio/video U have different token counts; masks must stay separate.
    negative_context = dict(video=unconditional.video.context,audio=unconditional.audio.context,
        video_mask=unconditional.video.context_mask,audio_mask=unconditional.audio.context_mask)
    return guided_sample(model.renderer,positive,negative_context,**sampling_kwargs)


@torch.no_grad()
def guided_sample(renderer, prepared, negative, steps=30, video_cfg=3., audio_cfg=7., sigmas=None, progress=None):
    """Baseline Euler/CFG with per-stream masks and unchanged Whisper evidence."""
    renderer.eval()
    if sigmas is None:
        from ltx_core.components.schedulers import LTX2Scheduler
        sigmas = LTX2Scheduler().execute(steps=steps)
    sigmas = sigmas.to(prepared.video.latent.device)
    if (len(sigmas) != steps+1 or abs(float(sigmas[0])-1.) > 1e-6 or abs(float(sigmas[-1])) > 1e-7
            or not (sigmas[:-1] > sigmas[1:]).all()):
        raise ValueError('Invalid sigma schedule')
    keep = ~prepared.video_loss_mask;reference = prepared.video.latent.float()
    video = torch.where(keep[...,None],reference,torch.randn_like(prepared.video.latent).float())
    audio = torch.randn_like(prepared.audio.latent).float()
    for i in range(steps):
        sigma = torch.full_like(prepared.video.sigma,float(sigmas[i]));dt = float(sigmas[i+1]-sigmas[i])
        vi = replace(prepared.video,latent=video.to(prepared.video.latent.dtype),sigma=sigma,
            timesteps=torch.where(keep,0,sigma[:,None]).to(prepared.video.timesteps.dtype))
        ai = replace(prepared.audio,latent=audio.to(prepared.audio.latent.dtype),sigma=sigma,
            timesteps=sigma[:,None].expand_as(prepared.audio.timesteps))
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=video.is_cuda):
            pv,pa = renderer(video=vi,audio=ai,perturbations=None)
            if video_cfg != 1 or audio_cfg != 1:
                nv = replace(vi,context=negative['video'],context_mask=negative['video_mask'])
                na = replace(ai,context=negative['audio'],context_mask=negative['audio_mask'])
                uv,ua = renderer(video=nv,audio=na,perturbations=None)
                pv = uv.float()+video_cfg*(pv.float()-uv.float())
                pa = ua.float()+audio_cfg*(pa.float()-ua.float())
        video = torch.where(keep[...,None],reference,video+dt*pv.float());audio = audio+dt*pa.float()
        if not torch.isfinite(video).all() or not torch.isfinite(audio).all():
            raise FloatingPointError('Nonfinite AV sample')
        if progress is not None and ((i+1)%5 == 0 or i+1 == steps):
            progress(i+1,steps)
    if not torch.equal(video[keep],reference[keep]):
        raise ValueError('Reference first frame changed')
    return dict(video=video,audio=audio)


def load_model(path, device='cuda:0'):
    from .bootstrap import activate_ltx
    activate_ltx()
    from p2_listen.config import extractor_config
    from p2_clip_events.backend import data_config
    from p2_listen.features import weights_identity
    from .config import validate, source_identity
    from .model import QwenWhisperAV
    from .checkpoint import checked
    saved = checked(path);identity = saved['identity'];c = validate(identity['experiment'])
    if identity['source'] != source_identity():
        raise ValueError('Inference implementation differs from checkpoint')
    d = data_config(c)
    from omni_av_p2.ltx import load_renderer
    if identity['assets'] != {k:weights_identity(d,k) for k in ('omni','ltx','gemma')}:
        raise ValueError('Inference base model assets differ')
    cfg = extractor_config(d);renderer,strategy,sampler = load_renderer(cfg,device)
    torch.manual_seed(c['training']['seed'])
    model = QwenWhisperAV(c,cfg,renderer).to(device)
    model.load_state_dict(saved['model'],strict=True);model.eval()
    return model,strategy,sampler
