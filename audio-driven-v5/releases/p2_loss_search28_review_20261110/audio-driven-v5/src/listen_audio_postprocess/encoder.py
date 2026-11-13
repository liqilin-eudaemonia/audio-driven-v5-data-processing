"""Persistent native audio encoder only; never load video VAE, Omni or Gemma."""
import numpy as np
import torch
from p2_listen.cache import cpu


class NativeAudioEncoder:
    def __init__(self,config,device='cuda:0'):
        from p2_listen.config import extractor_config
        cfg=extractor_config(config)
        from ltx_core.model.audio_vae import AudioProcessor
        from ltx_trainer.model_loader import load_audio_vae_encoder
        self.device=device
        self.encoder=load_audio_vae_encoder(cfg.ltx_path,device=device,dtype=torch.float32).requires_grad_(False).eval()
        ae=self.encoder
        if not getattr(ae,'is_causal',False) or ae.sample_rate!=16000 or ae.mel_hop_length!=160:
            raise ValueError('Unsupported audio VAE time contract')
        self.processor=AudioProcessor(target_sample_rate=ae.sample_rate,mel_bins=ae.mel_bins,
                                      mel_hop_length=ae.mel_hop_length,n_fft=ae.n_fft).to(device)

    @torch.no_grad()
    def encode(self,wave,rate,geometry,video_shape):
        from ltx_core.types import Audio,AudioLatentShape
        from p2_listen.runtime import align_audio_latents,support_masks
        count=round(geometry['duration_s']*rate)
        wave=np.pad(wave,((0,max(0,count-len(wave))),(0,0)))[:count]
        value=torch.from_numpy(wave.T.copy())[None].to(self.device)
        if value.shape[1]==1:value=value.repeat(1,2,1)
        z=self.encoder(self.processor.waveform_to_mel(Audio(waveform=value,sampling_rate=rate)).float())
        expected=tuple(AudioLatentShape.from_duration(1,geometry['duration_s']).to_torch_shape())
        z,endpoint=align_audio_latents(z,expected)
        if not torch.isfinite(z).all():raise ValueError('Nonfinite audio latent')
        ae=self.encoder;margin=ae.n_fft/(2*ae.sample_rate)+ae.mel_hop_length/ae.sample_rate
        support=support_masks(geometry,video_shape,z.shape,margin)
        return cpu(dict(audio_latents=dict(latents=z.to(torch.bfloat16),num_time_steps=torch.tensor([z.shape[2]]),duration=torch.tensor([geometry['duration_s']])),
                        audio_support=support['audio'],geometry=geometry,audio_support_margin_s=margin,audio_endpoint_tokens_removed=endpoint))
