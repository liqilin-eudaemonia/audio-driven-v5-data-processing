"""Nested activation recomputation: preserve native blocks and all tensor values."""
from types import MethodType
from dataclasses import replace
from functools import partial
from contextlib import nullcontext
import torch
from torch.utils.checkpoint import checkpoint


def branch_learning_rate(base, name, step, stage_begin_step, warmup_steps):
    if name=='extractor' or warmup_steps==0:return base
    return base*min(1.,(step-stage_begin_step)/warmup_steps)


def tensor_block(vx, ax, *, block, video_template, audio_template, perturbations):
    v,a=block(video=replace(video_template,x=vx) if video_template is not None else None,
              audio=replace(audio_template,x=ax) if audio_template is not None else None,
              perturbations=perturbations)
    return v.x if v is not None else None, a.x if a is not None else None


def tensor_group(vx, ax, *, blocks, video_template, audio_template, perturbations, active):
    for block in blocks:
        call=partial(tensor_block,block=block,video_template=video_template,
                     audio_template=audio_template,perturbations=perturbations)
        vx,ax=checkpoint(call,vx,ax,use_reentrant=False) if active else call(vx,ax)
    return vx,ax


def grouped_blocks(self, video, audio, perturbations):
    size = self._residual_checkpoint_group_size
    threshold = getattr(self,'_residual_checkpoint_token_threshold',0)
    if video is not None and video.x.shape[1]<=threshold: size=1
    active = self._enable_gradient_checkpointing and self.training and torch.is_grad_enabled()
    # PyTorch's checkpoint input saver captures non-Tensor arguments by reference.
    # Passing dataclasses containing x to nested checkpoints pins EVERY block's
    # activation. Pass x as explicit Tensor inputs; templates must never retain x.
    vt=replace(video,x=None) if video is not None else None
    at=replace(audio,x=None) if audio is not None else None
    vx=video.x if video is not None else None
    ax=audio.x if audio is not None else None
    for start in range(0, len(self.transformer_blocks), size):
        call=partial(tensor_group,blocks=tuple(self.transformer_blocks[start:start+size]),
                     video_template=vt,audio_template=at,perturbations=perturbations,active=active)
        # Only outer boundary tensors are offloaded: the inner checkpoint hooks
        # suppress saving intermediate activations/weights. This leaves headroom
        # for Adam initialization and DDP's bucket rebuild on the second update.
        # Short clips still retain48 per-block inputs. With stage3 LoRA and
        # accumulated gradients these can exceed H200 memory on the final micro.
        # Offload those inputs too; no additional recomputation for short clips.
        offload=active and vx is not None and vx.is_cuda
        with torch.autograd.graph.save_on_cpu(pin_memory=True) if offload else nullcontext():
            vx,ax=checkpoint(call,vx,ax,use_reentrant=False) if active and size>1 else call(vx,ax)
    return replace(video,x=vx) if video is not None else None,replace(audio,x=ax) if audio is not None else None


def configure_grouped_checkpointing(renderer, group_size=1, token_threshold=32768):
    if group_size<1: raise ValueError('checkpoint group must be positive')
    renderer._residual_checkpoint_group_size=group_size
    renderer._residual_checkpoint_token_threshold=token_threshold
    renderer._process_transformer_blocks=MethodType(grouped_blocks,renderer)
    renderer.set_gradient_checkpointing(True)
