"""Fixed-text inference and diagnostics. Pred/prefix/swap never load GT target tensors."""
from pathlib import Path
import json

import torch

from .common import digest, read, resolve, sha, source_contract, write
from .features import (cpu, inference_layout, load_native, load_text, observation, paths, save,
                       unload)
from .training import build, checked, restore_model


@torch.no_grad()
def generate(cfg, args):
    from omni_av.ltx_runtime import with_text
    from omni_av_p2.config import Config as RendererConfig
    from omni_av_p2.inference import reference_latents
    from omni_av_p2.ltx import load_renderer, append_conditions, unpack_latents
    from omni_av_p3.data import load_native as load_perception
    from omni_av_p3.model import Config as PlannerConfig
    from omni_av_p3_online.model import select_contexts
    from omni_av_p3_staged.sampling import guided_sample
    from omni_av_p3_staged.media import decode_and_save
    torch.set_num_threads(2)
    root = resolve(cfg["output"])
    manifest = read(root / "windows.json")
    matches = [r for r in manifest["windows"] if r["sample_id"] == args.sample_id and r["split"] == args.split]
    if len(matches) != 1:
        raise ValueError("Specify one --sample-id in the requested --split")
    row = matches[0]
    media = paths(root, row)
    marker = read(media["marker"])
    for key in ("speaker_audio", "listener_image"):
        if sha(media[key]) != marker["files"][key]:
            raise ValueError("Inference input media changed")
    if args.split == "test" and not (root / "test_release.json").exists():
        raise ValueError("Test is held out; record selected checkpoint/config in test_release.json first")
    if args.mode == "audio-swap" and args.swap_audio is None:
        raise ValueError("Audio swap requires --swap-audio with exact window duration")
    if args.mode != "audio-swap" and args.swap_audio:
        raise ValueError("Swap audio is only valid for audio-swap mode")
    saved = checked(args.checkpoint) if args.checkpoint else None
    if args.mode != "text" and (saved is None or saved["stage"] == "P"):
        raise ValueError("Prediction/diagnostics need a C0/A checkpoint")
    if saved:
        if saved["identity"]["config"] != cfg or saved["identity"]["manifest"] != digest(manifest):
            raise ValueError("Checkpoint config/data mismatch")
        if saved["identity"]["code"] != source_contract(cfg):
            raise ValueError("Checkpoint source code changed")
    if args.split == "test":
        release = read(root / "test_release.json")
        if release.get("config_sha256") != digest(cfg) or (args.mode != "text" and release.get("checkpoint_sha256") != sha(args.checkpoint)):
            raise ValueError("Test checkpoint/config was not locked")
    out = args.output or root / "generated" / args.mode / row["sample_id"] / str(args.seed)
    out = Path(out)
    if (out / "completed.json").exists():
        raise FileExistsError("Generation already exists; choose a new output")
    obs = observation(root, row, args.swap_audio)
    geometry = dict(frames=row["frames"], fps=row["fps"], height=row["height"], width=row["width"], duration_s=row["duration_s"])
    # Encode reference before allocating the renderer; no GT tail or GT VAE read.
    rcfg = RendererConfig(**cfg["renderer"])
    av = reference_latents(rcfg, obs, geometry, device="cuda:0")
    text, negative = load_text(root, cfg, "cuda:0"), load_text(root, cfg, "cuda:0", negative=True)
    u = None
    torch.manual_seed(args.seed)
    if args.mode == "text":
        renderer, strategy, sampler = load_renderer(rcfg, "cuda:0")
    else:
        layout = None
        if args.mode == "audio-swap":
            runtime, processor = load_perception(PlannerConfig(**cfg["planner"]), "cuda:0")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                layout = inference_layout(runtime, processor, obs)
            del runtime, processor
            unload()
        model, strategy, sampler = build(cfg, saved["stage"], saved["identity"]["steps"])
        restore_model(model, saved)
        del saved
        model.requires_grad_(False).eval()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if args.mode == "oracle":
                layout, targets = load_native(root, row, cfg, targets=True, device="cuda:0")
                mask = torch.zeros_like(layout.valid)
                prefix, hidden = model.planner.readout(layout, mask, targets)
                u = model.connector(select_contexts(prefix, hidden, layout, mask))
                del targets
            elif args.mode == "prefix":
                layout = load_native(root, row, cfg, device="cuda:0")
                # Read input prefix without any target slots, avoiding target-mediated context.
                outputs = model.planner.thinker.model(inputs_embeds=layout.prefix,
                    position_ids=layout.prefix_positions, attention_mask=layout.prefix_valid.long(),
                    output_hidden_states=True, return_dict=True, use_cache=False)
                prefix = outputs.hidden_states[-2]
                hidden = torch.zeros(1, layout.valid.shape[1], layout.prefix.shape[-1], device="cuda:0", dtype=prefix.dtype)
                u = model.connector(select_contexts(prefix, hidden, layout, layout.valid.clone()))
                del outputs
            else:
                layout = layout if layout is not None else load_native(root, row, cfg, device="cuda:0")
                u = model.predicted_conditions(layout, args.seed, cfg["planner"]["planning_steps"])
        model.planner.cpu()
        renderer = model.renderer
        del layout
        unload()
    torch.manual_seed(args.seed)
    prepared = strategy.prepare_training_inputs(with_text(av, text), sampler)
    if u is not None:
        prepared = append_conditions(prepared, u)
    g = cfg["generation"]
    torch.manual_seed(args.seed)
    sampled = guided_sample(renderer, prepared, negative, steps=g["steps"], video_cfg=g["video_cfg"], audio_cfg=g["audio_cfg"],
                            progress=lambda n, total: print(dict(sample_step=n, total=total), flush=True))
    video, audio = unpack_latents(sampled, av)
    video, audio = video.cpu(), audio.cpu()
    save(out / "latents.pt", dict(video=video, audio=audio))
    del renderer, prepared, sampled, av, u, text, negative
    if args.mode != "text":
        del model
    unload()
    result = decode_and_save(rcfg, video, audio, row["fps"], out / "generated.mp4", raw_audio_path=out / "raw_audio.pt")
    if (result["frames"], result["height"], result["width"]) != (row["frames"], row["height"], row["width"]):
        raise ValueError("Generated geometry mismatch")
    write(out / "completed.json", dict(sample_id=row["sample_id"], split=row["split"], mode=args.mode, seed=args.seed,
        checkpoint_sha256=sha(args.checkpoint) if args.checkpoint else None, decoded=result,
        gt_features_used=args.mode == "oracle", gt_tail_vae_used=False, fixed_text=True,
        swap_audio_sha256=sha(args.swap_audio) if args.swap_audio else None, video_sha256=sha(out / "generated.mp4")))
    print(json.dumps(result), flush=True)
