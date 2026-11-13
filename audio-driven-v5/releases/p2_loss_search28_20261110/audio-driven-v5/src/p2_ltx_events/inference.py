"""Generation needs Qwen/text/speaker caches only; no Event labels or teacher media."""
import torch


@torch.no_grad()
def sample(model,prepared,u,speaker,negative,*,steps=30,video_cfg=3.,audio_cfg=7.,sigmas=None):
    from p2_listen.runtime import guided_sample,append_negative
    model.eval()
    return guided_sample(model.renderer,model.prepare(prepared,u,speaker),append_negative(negative,u),
                         steps,video_cfg,audio_cfg,sigmas)


def load_model(path,device='cuda:0'):
    from p2_whisper_av.bootstrap import activate_ltx
    activate_ltx()
    from .data import data_config
    from p2_listen.config import extractor_config
    from p2_listen.features import weights_identity
    from .config import validate,source_identity
    from .checkpoint import checked
    from .model import LTXHiddenEvents as QwenWhisperEventFeature
    saved=checked(path);identity=saved['identity'];c=validate(identity['experiment'])
    if identity['source']!=source_identity():raise ValueError('Combined inference source changed')
    d=data_config(c)
    if identity['assets']!={k:weights_identity(d,k) for k in ['omni','ltx','gemma']}:
        raise ValueError('Combined inference pretrained assets differ')
    from omni_av_p2.ltx import load_renderer
    renderer,strategy,sampler=load_renderer(extractor_config(d),device)
    torch.manual_seed(c['training']['seed'])
    model=QwenWhisperEventFeature(c,extractor_config(d),renderer).to(device)
    model.load_state_dict(saved['model'],strict=True)
    model.eval().requires_grad_(False);renderer.eval().requires_grad_(False)
    renderer.set_gradient_checkpointing(False)
    return model,strategy,sampler
