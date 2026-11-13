"""Native causal audio length bounds; isolated from older shared decoder."""
import math
import torch
from omni_av_p2.data import unload

def audio_tail_padding(*, target_samples, decoded_samples, output_rate, latent_frames,
                       mel_frames, sample_rate, hop_length, downsample_factor, causal):
    """Bound tail padding by native nearest-tick rounding AND causal VAE crop."""
    causal_crop = downsample_factor - 1 if causal else 0
    expected_mel = max(1, latent_frames * downsample_factor - causal_crop)
    if mel_frames != expected_mel:
        raise ValueError(f'Unexpected decoded mel length: {mel_frames} != {expected_mel}')
    expected_wave = round(mel_frames * hop_length / sample_rate * output_rate)
    if abs(decoded_samples - expected_wave) > 1:
        raise ValueError(f'Unexpected vocoder waveform length: {decoded_samples} != {expected_wave}')
    # from_duration uses round(duration / tick): at most half a tick is lost.
    # A causal decoder then removes factor-1 mel frames (30ms for native LTX).
    bound = math.ceil((downsample_factor / 2 + causal_crop) * hop_length / sample_rate * output_rate) + 1
    padding = max(0, target_samples - decoded_samples)
    if padding > bound:
        raise ValueError(f'Decoded audio shorter than native rounding/causal bound: {padding} > {bound}')
    return padding, bound


@torch.no_grad()
def decode_and_save(cfg, video, audio, fps, path, device="cuda:0", raw_audio_path=None):
    from ltx_trainer.model_loader import load_audio_vae_decoder, load_video_vae_decoder, load_vocoder
    from ltx_trainer.video_utils import save_video
    if path.exists():
        raise FileExistsError(path)
    decoder = load_video_vae_decoder(cfg.ltx_path, device, torch.bfloat16).requires_grad_(False).eval()
    # A full 233-frame 720p decode exceeds H200 memory even after unloading DiT.
    # Use native overlapping spatial/temporal tiles and move each chunk to CPU.
    from ltx_core.model.video_vae.tiling import TilingConfig
    tiled = video.shape[2] > 10 or video.shape[3] > 24 or video.shape[4] > 24
    tiling = TilingConfig.default() if tiled else None
    chunks = []
    for chunk in decoder.decode_video(video.to(device, torch.bfloat16), tiling_config=tiling):
        if not torch.isfinite(chunk).all():
            raise FloatingPointError("Invalid decoded video")
        chunks.append(chunk.float().cpu())
    pixels = torch.cat(chunks, 0).permute(3, 0, 1, 2).contiguous()
    expected = ((video.shape[2] - 1) * 8 + 1, video.shape[3] * 32, video.shape[4] * 32)
    if tuple(pixels.shape[1:]) != expected:
        raise ValueError(f"Decoded video geometry mismatch: {pixels.shape[1:]} != {expected}")
    del chunks, chunk
    del decoder
    unload()
    decoder = load_audio_vae_decoder(cfg.ltx_path, device, torch.bfloat16).requires_grad_(False).eval()
    mel = decoder(audio.to(device, torch.bfloat16))
    from ltx_core.model.audio_vae.causality_axis import CausalityAxis
    audio_geometry = dict(latent_frames=audio.shape[2], mel_frames=mel.shape[2],
        sample_rate=decoder.sample_rate, hop_length=decoder.mel_hop_length,
        downsample_factor=decoder.patchifier.audio_latent_downsample_factor,
        causal=decoder.causality_axis != CausalityAxis.NONE)
    del decoder
    unload()
    vocoder = load_vocoder(cfg.ltx_path, device, torch.bfloat16).requires_grad_(False).eval()
    wave = vocoder(mel).squeeze(0).float().cpu()
    if not torch.isfinite(wave).all():
        raise FloatingPointError("Invalid decoded audio")
    rate = int(vocoder.output_sampling_rate)
    raw_samples = wave.shape[-1]
    del vocoder, mel
    unload()
    if raw_audio_path is not None:
        from omni_av_p2.checkpoint import atomic_save
        if not raw_audio_path.exists():atomic_save(raw_audio_path,dict(wave=wave,rate=rate,pre_clamp=True))
    count = round(pixels.shape[1] / fps * rate)
    padding, bound = audio_tail_padding(target_samples=count, decoded_samples=raw_samples,
                                       output_rate=rate, **audio_geometry)
    if padding:
        wave = torch.nn.functional.pad(wave, (0, padding))
    wave = wave[..., :count]
    save_video(pixels, path, fps=fps, audio=wave, audio_sample_rate=rate, video_format="CFHW")
    return dict(frames=pixels.shape[1], height=pixels.shape[2], width=pixels.shape[3], tiled_video_decode=tiled,
                audio_samples=wave.shape[-1], audio_rate=rate, rounding_padding_samples=padding,
                raw_audio_samples=raw_samples, audio_padding_bound_samples=bound,
                audio_decode_geometry=audio_geometry,
                duration_s=pixels.shape[1] / fps)
