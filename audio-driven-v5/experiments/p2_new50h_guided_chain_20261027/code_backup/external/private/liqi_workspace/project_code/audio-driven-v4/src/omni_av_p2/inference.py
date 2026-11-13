"""Deployment uses Connector only and never reads paired targets or feature caches."""
import json
import math
from dataclasses import asdict, replace

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from omni_av.data import Observation, TextPlan, save_new, write_json_new
from omni_av.ltx_runtime import NativeTextConditioner, with_text
from omni_av.omni import generate_reply, prepare_observation
from .checkpoint import load
from .connector import FusionConnector
from .data import contract, extract_condition, load_omni, unload
from .ltx import append_conditions, decode_and_save, load_renderer, sample, unpack_latents


def output_geometry(duration, fps, height, width):
    if not math.isfinite(fps) or fps <= 0 or height < 64 or width < 64 or height % 32 or width % 32:
        raise ValueError("Invalid output geometry; sizes must be >=64 and multiples of 32")
    frames = int((math.floor(duration * fps) - 1) // 8 * 8 + 1)
    if frames < 9:
        raise ValueError("Output budget too short")
    return dict(frames=frames, fps=fps, height=height, width=width, duration_s=frames / fps,
                external_budget_s=duration)


@torch.no_grad()
def reference_latents(cfg, obs, geometry, device="cuda:0"):
    from ltx_core.types import AudioLatentShape
    from ltx_trainer.model_loader import load_video_vae_encoder
    with Image.open(obs.listener_image) as source:
        image = torch.from_numpy(np.asarray(source.convert("RGB")).copy()).permute(2, 0, 1).float() / 255
    height, width = geometry["height"], geometry["width"]
    ih, iw = image.shape[1:]
    scale = max(height / ih, width / iw)
    rh, rw = max(height, round(ih * scale)), max(width, round(iw * scale))
    image = F.interpolate(image[None], size=(rh, rw), mode="bilinear", align_corners=False)
    top, left = (rh - height) // 2, (rw - width) // 2
    image = image[:, :, top:top + height, left:left + width, None].permute(0, 1, 4, 2, 3)
    encoder = load_video_vae_encoder(cfg.ltx_path, device, torch.bfloat16).requires_grad_(False).eval()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        first = encoder((image * 2 - 1).to(device))
    del encoder
    unload()
    f, h, w = (geometry["frames"] - 1) // 8 + 1, height // 32, width // 32
    if first.shape != (1, 128, 1, h, w):
        raise ValueError("First-frame VAE shape mismatch")
    video = torch.zeros(1, 128, f, h, w, device=device, dtype=torch.bfloat16)
    video[:, :, :1] = first
    ashape = AudioLatentShape.from_duration(1, geometry["duration_s"])
    audio = torch.zeros(ashape.to_torch_shape(), device=device, dtype=torch.bfloat16)
    make = lambda value: torch.tensor([value], device=device)
    return dict(latents=dict(latents=video, num_frames=make(f), height=make(h), width=make(w), fps=make(geometry["fps"])),
                audio_latents=dict(latents=audio, num_time_steps=make(ashape.frames)))


def run_inference(args, cfg):
    if args.observation is None or (args.checkpoint is None and not args.text_only):
        raise ValueError("Inference needs --observation and --checkpoint, or explicit --text-only baseline")
    if args.text_only and (args.command == "export-condition" or args.zero_condition):
        raise ValueError("Text-only baseline has no U to export or intervene on")
    if args.cache is not None or args.manifest is not None:
        raise ValueError("Inference refuses training cache/manifest inputs")
    obs = Observation(**json.loads(args.observation.read_text()))
    if not args.text_only:
        saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        if saved["identity"]["contract"] != contract(cfg):
            raise ValueError("Inference/checkpoint model configuration mismatch")
        identity = saved["identity"]
        del saved
    if not args.text_only or args.text_plan is None:
        runtime, processor = load_omni(cfg)
    if args.text_plan:
        plan = TextPlan(**json.loads(args.text_plan.read_text()))
        source = "provided_text"
    else:
        plan, _ = generate_reply(runtime, processor, prepare_observation(processor, obs, cfg))
        source = "autonomous_text"
    if not args.text_only:
        condition = extract_condition(runtime, processor, obs, plan)
    if not args.text_only or args.text_plan is None:
        del runtime, processor
        unload()
    u = None
    if not args.text_only:
        connector = FusionConnector(cfg).cuda().eval()
        load(args.checkpoint, connector, identity, connector_only=True)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            u = connector(condition)
        if args.zero_condition:
            u = replace(u, video=torch.zeros_like(u.video), audio=torch.zeros_like(u.audio))
        save_new(args.output / "condition.pt", dict(contract=contract(cfg), observation=asdict(obs),
            text_plan=asdict(plan), source=source, intervention="zero" if args.zero_condition else "none",
            u={k: v.cpu() for k, v in vars(u).items()}))
        del connector, condition
    write_json_new(args.output / "text_plan.json", asdict(plan))
    unload()
    if args.command == "export-condition":
        return dict(condition_exported=True, text_source=source, gt_access=False, decoder_loaded=False)
    geometry = output_geometry(obs.output_duration_s, args.fps, args.height, args.width)
    write_json_new(args.output / "geometry.json", geometry)
    text_runtime = NativeTextConditioner(cfg)
    try:
        text = text_runtime.encode(plan)
    finally:
        text_runtime.close()
    av = reference_latents(cfg, obs, geometry)
    renderer, strategy, sampler = load_renderer(cfg)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        prepared = strategy.prepare_training_inputs(with_text(av, text), sampler)
        inputs = append_conditions(prepared, u) if u is not None else prepared
        torch.manual_seed(args.seed)
        sampled = sample(renderer, inputs, args.sampling_steps)
        video, audio = unpack_latents(sampled, av)
    save_new(args.output / "sampled_latents.pt", dict(video=video.cpu(), audio=audio.cpu(), source=source))
    del renderer, inputs, prepared, sampled
    unload()
    decoded = decode_and_save(cfg, video, audio, args.fps, args.output / "generated.mp4")
    return dict(engineering_pass=True, quality_pass=None, gt_access=False, text_source=source,
                decoder_loaded=False, text_only=args.text_only, zero_condition=args.zero_condition,
                sampling="Euler without CFG; engineering baseline", decoded=decoded)
