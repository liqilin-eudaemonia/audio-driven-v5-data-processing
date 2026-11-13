"""LTX-2.3 pretrained renderer, native Gemma conditions and paired VAE targets."""
import gc
import json
from dataclasses import replace
from pathlib import Path
from functools import lru_cache

import torch
from torch import nn

from .data import TextPlan
from .renderer import PlanningConnector, configure_renderer_lora, install_plan_attention


class NativeTextConditioner:
    """Load frozen Gemma once when preparing multiple text conditions."""

    def __init__(self, config, device="cuda:0"):
        self.encoder, self.processor = load_text_modules(config, device)

    def encode(self, plan):
        prompt = plan.render_prompt()
        if len(self.encoder.tokenizer.tokenizer(prompt, truncation=False)["input_ids"]) > 1024:
            raise ValueError("Text exceeds 1024 token native context; refusing truncation")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            hidden, mask = self.encoder.encode(prompt)
            result = self.processor.process_hidden_states(hidden, mask)
        output = dict(video=result.video_encoding.detach(), audio=result.audio_encoding.detach(),
                      mask=result.attention_mask.bool(), prompt=prompt)
        if output["video"].shape[-1] != 4096 or output["audio"].shape[-1] != 2048:
            raise ValueError("Unexpected LTX text context dimensions")
        return output

    def encode_batch(self, plans):
        """Native tokenization and per-example processors, with a batched Gemma forward."""
        if not plans:
            raise ValueError('Empty text batch')
        prompts = [plan.render_prompt() for plan in plans]
        tokenizer = self.encoder.tokenizer
        if any(len(tokenizer.tokenizer(p, truncation=False)['input_ids']) > 1024 for p in prompts):
            raise ValueError('Text exceeds 1024 token native context; refusing truncation')
        pairs = [tokenizer.tokenize_with_weights(p)['gemma'] for p in prompts]
        device = self.encoder.model.device
        ids = torch.tensor([[x[0] for x in row] for row in pairs], device=device)
        mask = torch.tensor([[x[1] for x in row] for row in pairs], device=device)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            encoded = self.encoder.model.model(input_ids=ids, attention_mask=mask,
                                               output_hidden_states=True)
            result = self.processor.process_hidden_states(encoded.hidden_states, mask)
        if result.video_encoding.shape[-1] != 4096 or result.audio_encoding.shape[-1] != 2048:
            raise ValueError('Unexpected batched LTX text context dimensions')
        return [dict(video=result.video_encoding[i:i+1].detach(),
                     audio=result.audio_encoding[i:i+1].detach(),
                     mask=result.attention_mask[i:i+1].bool(), prompt=prompt)
                for i, prompt in enumerate(prompts)]

    def close(self):
        self.encoder = None
        self.processor = None
        gc.collect()
        torch.cuda.empty_cache()


def load_text_modules(config, device):
    from ltx_core.text_encoders.gemma.encoders.base_encoder import GemmaTextEncoder
    from ltx_core.text_encoders.gemma.tokenizer import LTXVGemmaTokenizer
    from ltx_trainer.model_loader import load_embeddings_processor
    from transformers import Gemma3ForConditionalGeneration
    model = Gemma3ForConditionalGeneration.from_pretrained(config.gemma_path,
        dtype=torch.bfloat16, device_map={"": device}, local_files_only=True).requires_grad_(False).eval()
    encoder = GemmaTextEncoder(tokenizer=LTXVGemmaTokenizer(config.gemma_path, 1024),
                              model=model, dtype=torch.bfloat16)
    processor = load_embeddings_processor(config.ltx_path, device=device, dtype=torch.bfloat16)
    processor.requires_grad_(False).eval()
    return encoder, processor


def encode_text(config, plan, device="cuda:0"):
    runtime = NativeTextConditioner(config, device)
    try:
        return runtime.encode(plan)
    finally:
        runtime.close()


@lru_cache(maxsize=6)
def _vae_metadata(path, mtime_ns, size):
    rows = [json.loads(line) for line in Path(path).open()]
    result = {row["sample_id"]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError("Duplicate VAE metadata IDs")
    return result


def load_av_targets(root, sample_id, device="cuda:0", tiny=False, split="train", tensor_loader=None):
    if split not in {"train", "dev", "test"}:
        raise ValueError("Unknown dataset split")
    root = Path(root) / "ltx23_precomputed_720p_area_qualified_20260920"
    path = root / f"metadata_{split}.jsonl"
    stat = path.stat()
    meta = _vae_metadata(str(path.resolve()), stat.st_mtime_ns, stat.st_size).get(sample_id)
    if meta is None or meta["split"] != split:
        raise ValueError(f"Sample absent from {split} VAE metadata")
    result = {}
    sources = {}
    for key, folder in (("latents", "latents"), ("audio_latents", "audio_latents")):
        candidates = [root / folder / f"{sample_id}.pt", root / folder / split / f"{sample_id}.pt"]
        found = [p for p in candidates if p.is_file()]
        if len(found) != 1:
            raise ValueError(f"Expected one VAE cache: {candidates}")
        payload = (tensor_loader(found[0]) if tensor_loader is not None else
                   torch.load(found[0], map_location="cpu", weights_only=True))
        sources[key] = str(found[0])
        if not torch.isfinite(payload["latents"]).all():
            raise ValueError("Nonfinite VAE target")
        result[key] = {k: v[None].to(device, torch.bfloat16) if k == "latents"
                       else torch.tensor([v], device=device) for k, v in payload.items()}
    video, audio = result["latents"]["latents"], result["audio_latents"]["latents"]
    expected = (1, 128, (meta["target_frames"] - 1) // 8 + 1,
                meta["target_height"] // 32, meta["target_width"] // 32)
    if tuple(video.shape) != expected or audio.shape[1] != 8 or audio.shape[-1] != 16:
        raise ValueError("VAE metadata shape mismatch")
    duration = float(result["audio_latents"]["duration"].item())
    if abs(duration - meta["target_frames"] / meta["fps"]) > 1e-4:
        raise ValueError("VAE audio/video time mismatch")
    original = {k: list(v["latents"].shape) for k, v in result.items()}
    if tiny:
        result["latents"]["latents"] = video[:, :, :3, :4, :4].contiguous()
        for key, value in (("num_frames", 3), ("height", 4), ("width", 4)):
            result["latents"][key] = torch.tensor([value], device=device)
        result["audio_latents"]["latents"] = audio[:, :, :24].contiguous()
        result["audio_latents"]["num_time_steps"] = torch.tensor([24], device=device)
    return result, dict(metadata=meta, original_shapes=original, sources=sources, tiny=tiny)


class RendererBundle(nn.Module):
    def __init__(self, model, connector, planner=None):
        super().__init__()
        self.renderer, self.connector = model, connector
        self.planner = planner


def load_renderer(config, device="cuda:0", lora=True):
    from ltx_core.model.transformer.attention import Attention, PytorchAttention
    from ltx_trainer.model_loader import load_transformer
    from ltx_trainer.timestep_samplers import UniformTimestepSampler
    from ltx_trainer.training_strategies.text_to_video import TextToVideoConfig, TextToVideoStrategy
    model = load_transformer(config.ltx_path, device=device, dtype=torch.bfloat16)
    install_plan_attention(model)
    model.set_gradient_checkpointing(True)
    for module in model.modules():
        if isinstance(module, Attention):
            module.attention_function = PytorchAttention()
    if lora:
        model = configure_renderer_lora(model)
    connector = PlanningConnector(config.hidden_size).to(device)
    strategy = TextToVideoStrategy(TextToVideoConfig(with_audio=True, first_frame_conditioning_p=1.0,
                                                    i2av_only=True))
    sampler = UniformTimestepSampler()
    return RendererBundle(model, connector), strategy, sampler


def with_text(av, text):
    return dict(av, conditions=dict(video_prompt_embeds=text["video"], audio_prompt_embeds=text["audio"],
        prompt_attention_mask=text["mask"], video_prompt_attention_mask=text["mask"],
        audio_prompt_attention_mask=text["mask"]))


def audit_target_window(payload, metadata):
    """Accept only the documented terminal-frame trim of the inherited VAE cache."""
    meta = metadata["metadata"]
    full = float(payload["example"]["duration_s"])
    rendered = meta["target_frames"] / meta["fps"]
    trimmed = full - rendered
    if trimmed < -1 / meta["fps"] - 1e-4 or trimmed > 8 / meta["fps"] + 1e-4:
        raise ValueError("VAE window differs from target media beyond the known terminal-frame trim")
    return dict(planning_target_duration_s=full, renderer_target_duration_s=rendered,
                terminal_trim_s=trimmed, tiny=metadata["tiny"])


def render_loss(bundle, strategy, inputs):
    video, audio = bundle.renderer(video=inputs.video, audio=inputs.audio, perturbations=None)
    fp32 = replace(inputs, video_targets=inputs.video_targets.float(), audio_targets=inputs.audio_targets.float())
    loss = strategy.compute_loss(video.float(), audio.float(), fp32)
    return loss.loss.mean(), {k: v.mean() for k, v in loss.components.items()}


@torch.no_grad()
def sample_latents(bundle, prepared, steps=3):
    """Unconditional-noise start except the reference first frame; no GT tail use.

    Prepared carries shapes/positions and text+plan. Only first-frame latents survive.
    This short Euler gate verifies sampler plumbing, not production sampling quality.
    """
    bundle.eval()
    keep = ~prepared.video_loss_mask
    video = torch.randn_like(prepared.video.latent).float()
    reference = prepared.video.latent.float()
    video = torch.where(keep[..., None], reference, video)
    audio = torch.randn_like(prepared.audio.latent).float()
    for step in range(steps):
        sigma = torch.full_like(prepared.video.sigma, 1 - step / steps)
        vi = replace(prepared.video, latent=video.to(prepared.video.latent.dtype), sigma=sigma,
                     timesteps=torch.where(keep, 0, sigma[:, None]).to(prepared.video.timesteps.dtype))
        ai = replace(prepared.audio, latent=audio.to(prepared.audio.latent.dtype), sigma=sigma,
                     timesteps=sigma[:, None].expand_as(prepared.audio.timesteps))
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=video.is_cuda):
            pv, pa = bundle.renderer(video=vi, audio=ai, perturbations=None)
        video = torch.where(keep[..., None], reference, video - pv.float() / steps)
        audio = audio - pa.float() / steps
        if not torch.isfinite(video).all() or not torch.isfinite(audio).all():
            raise FloatingPointError("Nonfinite AV sampling")
    if not torch.equal(video[keep], reference[keep]):
        raise RuntimeError("First frame changed during sampling")
    return dict(video=video, audio=audio)
