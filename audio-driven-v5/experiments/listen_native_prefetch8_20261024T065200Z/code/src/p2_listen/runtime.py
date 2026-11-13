"""Native LTX adapters, explicit masks and symmetric U conditioning in CFG."""
from dataclasses import replace
import math
import torch
from .config import extractor_config
from .layout import geometry
from .features import unload


def masked_mse(pred, target, mask):
    if pred.shape != target.shape or mask.shape != pred.shape[:2]:
        raise ValueError("Renderer target/mask shape mismatch")
    error = (pred.float() - target.float()).square().mean(-1)
    return (error.masked_fill(~mask, 0).sum(-1) / mask.sum(-1).clamp_min(1)).mean()


def av_loss(renderer, prepared, support):
    # The old native strategy ignores audio_loss_mask in its reduction.
    pv, pa = renderer(video=prepared.video, audio=prepared.audio, perturbations=None)
    vm = prepared.video_loss_mask & support["video"].to(pv.device)
    am = prepared.audio_loss_mask & support["audio"].to(pa.device)
    return masked_mse(pv,prepared.video_targets,vm) + masked_mse(pa,prepared.audio_targets,am)


def support_masks(g, video_shape, audio_shape, audio_margin_s=0.05):
    _, _, f, h, w = video_shape
    # Encoder is causal: group t ends after original frame 8*t.
    vend = (8 * torch.arange(f) + 1) / g["fps"]
    video = (vend <= g["real_duration_s"] + 1e-6).repeat_interleave(h*w)[None]
    # Conservative native 40ms audio tick + STFT lookahead margin.
    audio = ((torch.arange(audio_shape[2])+1)*.04 + audio_margin_s <= g["real_duration_s"]+1e-6)[None]
    return dict(video=video, audio=audio)


def align_audio_latents(latents, expected):
    """Centered mel adds an endpoint; causal stride-4 may retain one extra token.

    Drop only that documented endpoint beyond the renderer's duration budget,
    never waveform samples or an arbitrary mismatched latent sequence.
    """
    if tuple(latents.shape)==tuple(expected):
        return latents,0
    if (latents.shape[0],latents.shape[1],latents.shape[3])!=(expected[0],expected[1],expected[3]) or latents.shape[2]!=expected[2]+1:
        raise ValueError(f"Unexpected audio VAE geometry: {tuple(latents.shape)} != {tuple(expected)}")
    return latents[:,:,:expected[2]].contiguous(),1


@torch.no_grad()
def encode_vae(c, obs, target, device):
    import numpy as np
    import soundfile as sf
    from PIL import Image
    from ltx_core.model.audio_vae import AudioProcessor
    from ltx_core.types import Audio, AudioLatentShape
    from ltx_trainer.model_loader import load_video_vae_encoder, load_audio_vae_encoder
    from .features import sample_frames
    from .cache import cpu
    cfg, g = extractor_config(c), geometry(obs)
    pixels = sample_frames(target["target_video"], [i/g["fps"] for i in range(g["frames"])], (g["width"],g["height"]))
    ve = load_video_vae_encoder(cfg.ltx_path,device=device,dtype=torch.bfloat16).requires_grad_(False).eval()
    with Image.open(obs["listener_image"]) as im:
        if im.size != (g["width"],g["height"]):
            raise ValueError("Reference must share the already-cropped output geometry")
        first = np.asarray(im.convert("RGB")).copy()
    value = torch.from_numpy(pixels).permute(3,0,1,2)[None].to(device,torch.bfloat16)/127.5-1
    ref = torch.from_numpy(first).permute(2,0,1)[None,:,None].to(device,torch.bfloat16)/127.5-1
    with torch.autocast("cuda",dtype=torch.bfloat16):
        z = ve(value)
        z[:,:,:1] = ve(ref)
    if z.shape != (1,128,(g["frames"]-1)//8+1,g["height"]//32,g["width"]//32):
        raise ValueError("Video VAE geometry differs; silent crop is forbidden")
    del ve, value, ref, pixels
    unload()
    ae = load_audio_vae_encoder(cfg.ltx_path,device=device,dtype=torch.float32).requires_grad_(False).eval()
    if not getattr(ae,"is_causal",False):
        raise ValueError("Tail support policy requires causal audio encoder")
    ap = AudioProcessor(target_sample_rate=ae.sample_rate,mel_bins=ae.mel_bins,mel_hop_length=ae.mel_hop_length,n_fft=ae.n_fft).to(device)
    wave,rate = sf.read(target["target_audio"],dtype="float32",always_2d=True)
    count = round(g["duration_s"]*rate)
    wave = np.pad(wave,((0,max(0,count-len(wave))),(0,0)))[:count]
    value = torch.from_numpy(wave.T.copy())[None].to(device)
    if value.shape[1] == 1:
        value = value.repeat(1,2,1)
    az = ae(ap.waveform_to_mel(Audio(waveform=value,sampling_rate=rate)).float())
    expected = tuple(AudioLatentShape.from_duration(1,g["duration_s"]).to_torch_shape())
    if ae.sample_rate!=16000 or ae.mel_hop_length!=160:
        raise ValueError("Unsupported native audio time scale")
    az,endpoint_tokens=align_audio_latents(az,expected)
    margin = ae.n_fft / (2*ae.sample_rate) + ae.mel_hop_length/ae.sample_rate
    support = support_masks(g,z.shape,az.shape,margin)
    scalar = lambda x: torch.tensor([x])
    result = dict(av=dict(latents=dict(latents=z,num_frames=scalar(z.shape[2]),height=scalar(z.shape[3]),width=scalar(z.shape[4]),fps=scalar(g["fps"])),
                 audio_latents=dict(latents=az.to(torch.bfloat16),num_time_steps=scalar(az.shape[2]),duration=scalar(g["duration_s"]))),
                  support=support, geometry=g, audio_support_margin_s=margin,audio_endpoint_tokens_removed=endpoint_tokens)
    if not torch.isfinite(z).all() or not torch.isfinite(az).all():
        raise ValueError("Nonfinite VAE targets")
    result = cpu(result)
    del ae,ap,value,z,az
    unload()
    return result


def append_negative(negative, u):
    result = {}
    for mod in ("video","audio"):
        if u is None:
            result[mod] = negative[mod]
            result[mod+"_mask"] = negative["mask"]
        else:
            context = getattr(u,mod)
            result[mod] = torch.cat([negative[mod],context.to(negative[mod])],1)
            result[mod+"_mask"] = torch.cat([negative["mask"],getattr(u,mod+"_valid")],1)
    return result


@torch.no_grad()
def guided_sample(renderer, prepared, negative, steps, video_cfg, audio_cfg, sigmas=None):
    """U already appended to BOTH branches; modality masks may have different lengths."""
    if sigmas is None:
        from ltx_core.components.schedulers import LTX2Scheduler
        sigmas = LTX2Scheduler().execute(steps=steps)
    sigmas = sigmas.to(prepared.video.latent.device)
    if len(sigmas)!=steps+1 or abs(float(sigmas[0])-1)>1e-6 or abs(float(sigmas[-1]))>1e-7 or not (sigmas[:-1]>sigmas[1:]).all():
        raise ValueError("Invalid LTX schedule")
    renderer.eval()
    keep = ~prepared.video_loss_mask
    reference = prepared.video.latent.float()
    video = torch.where(keep[...,None],reference,torch.randn_like(reference))
    audio = torch.randn_like(prepared.audio.latent).float()
    for i in range(steps):
        sigma = torch.full_like(prepared.video.sigma,float(sigmas[i]))
        dt = float(sigmas[i+1]-sigmas[i])
        vi = replace(prepared.video,latent=video.to(prepared.video.latent.dtype),sigma=sigma,
                     timesteps=torch.where(keep,0,sigma[:,None]).to(prepared.video.timesteps.dtype))
        ai = replace(prepared.audio,latent=audio.to(prepared.audio.latent.dtype),sigma=sigma,
                     timesteps=sigma[:,None].expand_as(prepared.audio.timesteps))
        with torch.autocast("cuda",dtype=torch.bfloat16,enabled=video.is_cuda):
            pv,pa = renderer(video=vi,audio=ai,perturbations=None)
            if video_cfg!=1 or audio_cfg!=1:
                nv = replace(vi,context=negative["video"],context_mask=negative["video_mask"])
                na = replace(ai,context=negative["audio"],context_mask=negative["audio_mask"])
                uv,ua = renderer(video=nv,audio=na,perturbations=None)
                pv = uv.float()+video_cfg*(pv.float()-uv.float())
                pa = ua.float()+audio_cfg*(pa.float()-ua.float())
        video = torch.where(keep[...,None],reference,video+dt*pv.float())
        audio = audio+dt*pa.float()
        if not torch.isfinite(video).all() or not torch.isfinite(audio).all():
            raise FloatingPointError("Nonfinite generated AV")
    return dict(video=video,audio=audio)
