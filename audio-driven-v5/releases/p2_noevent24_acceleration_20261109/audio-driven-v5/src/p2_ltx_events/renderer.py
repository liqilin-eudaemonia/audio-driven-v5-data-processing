"""Explicit readout before the native output projections; no hooks or detach.

This mirrors the private LTXModel.forward preparation and output path. Native
forward remains untouched, including all existing checkpoint/source identities.
Parity with native forward is tested on the actual LTX implementation.
"""
import dataclasses


def forward_with_hidden(renderer, video, audio, perturbations=None):
    if not renderer.model_type.is_video_enabled() and video is not None:
        raise ValueError('Video is not enabled for this model')
    if not renderer.model_type.is_audio_enabled() and audio is not None:
        raise ValueError('Audio is not enabled for this model')
    video = renderer._prepare_qwen_context(video, is_audio=False) if video is not None else None
    audio = renderer._prepare_qwen_context(audio, is_audio=True) if audio is not None else None
    if (video is not None and audio is not None and video.omni_context is not None
            and video.omni_context is audio.omni_context):
        video = renderer._resample_omni_context(video)
        audio = dataclasses.replace(audio, omni_context=video.omni_context,
            omni_context_mask=video.omni_context_mask, omni_field_mask=video.omni_field_mask)
    video = renderer._prepare_omni_context(video, is_audio=False) if video is not None else None
    audio = renderer._prepare_omni_context(audio, is_audio=True) if audio is not None else None
    va = renderer.video_args_preprocessor.prepare(video, audio) if video is not None else None
    aa = renderer.audio_args_preprocessor.prepare(audio, video) if audio is not None else None
    if audio is not None and audio.first_frame_latent is not None:
        aa = dataclasses.replace(aa, first_frame_context=renderer._prepare_first_frame_context(audio))
    # The native block loop retains its non-reentrant activation checkpointing.
    va, aa = renderer._process_transformer_blocks(va, aa, perturbations)
    vx = (renderer._process_output(renderer.scale_shift_table, renderer.norm_out,
          renderer.proj_out, va.x, va.embedded_timestep) if va is not None else None)
    ax = (renderer._process_output(renderer.audio_scale_shift_table, renderer.audio_norm_out,
          renderer.audio_proj_out, aa.x, aa.embedded_timestep) if aa is not None else None)
    hidden = {m: a.x for m, a in [('video', va), ('audio', aa)] if a is not None}
    return (vx, ax), hidden
