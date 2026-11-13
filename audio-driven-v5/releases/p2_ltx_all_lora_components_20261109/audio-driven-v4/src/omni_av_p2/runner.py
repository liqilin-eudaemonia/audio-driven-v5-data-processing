import json
import random
import traceback
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from omni_av.data import save_new, write_json_new
from omni_av.ltx_runtime import load_av_targets, with_text
from . import VERSION
from .checkpoint import export_connector, load, save
from .config import Config
from .data import contract, prepare_cache, read_sample, unload
from .ltx import append_conditions, av_loss, decode_and_save, load_renderer, sample, unpack_latents
from .model import Extractor


def event(name, **values):
    print(json.dumps(dict(event=name, **values), ensure_ascii=False), flush=True)


def build_config(args):
    cfg = Config(**json.loads(args.config.read_text())) if args.config else Config()
    if args.engineering_small:
        cfg = replace(cfg, fusion_width=128, fusion_depth=1, fusion_heads=4, decoder_width=128,
                      decoder_depth=2, diffusion_batch_mul=2)
    return cfg


def gradient_norms(loss, tensors):
    values = torch.autograd.grad(loss, tensors, retain_graph=True, allow_unused=True)
    return [0.0 if value is None else float(value.float().norm()) for value in values]


def run_training(args, cfg):
    if args.cache is None:
        raise ValueError("Training/gate requires a separately prepared P2 cache")
    if args.steps < 1 or args.save_every < 1:
        raise ValueError("Invalid step/save count")
    if args.command == "gate" and args.steps < 3:
        raise ValueError("Gate requires >=3 steps to audit zero-initialized branches after warmup")
    index = json.loads(args.cache.read_text())
    if index["contract"] != contract(cfg) or not index["entries"]:
        raise ValueError("P2 cache mismatch or empty dataset")
    model = Extractor(cfg, auxiliary=args.recipe == "joint").cuda()
    event("p2_loading_renderer", trainable=sum(p.numel() for p in model.parameters()))
    renderer, strategy, sampler = load_renderer(cfg)
    event("p2_renderer_loaded", parameters=sum(p.numel() for p in renderer.parameters()), frozen=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    identity = dict(contract=contract(cfg), recipe=args.recipe, entries=index["entries"],
                    tiny=args.tiny, seed=args.seed)
    step0 = load(args.resume, model, identity, optimizer)["step"] if args.resume else 0
    frozen = [(p, p._version) for p in renderer.parameters()]
    checkpoints, metrics = [], []
    last, saved = step0, step0
    with (args.output / "train.jsonl").open("x") as log:
        for step in range(step0 + 1, step0 + args.steps + 1):
            if (args.output / "STOP").exists():
                break
            # AV noise/data order independent of auxiliary-head initialization or noise draws.
            torch.manual_seed(args.seed + step)
            entry = index["entries"][(step - 1) % len(index["entries"])]
            condition, targets, text, payload = read_sample(entry, cfg)
            trim = payload["window"]["terminal_trim_s"]
            if abs(trim) > 1e-4 and not args.allow_terminal_trim:
                raise ValueError("Inherited VAE terminal trim requires explicit --allow-terminal-trim or aligned reprocessed caches")
            av, metadata = load_av_targets(args.data_root, payload["sample_id"], tiny=args.tiny)
            actual_duration = metadata["metadata"]["target_frames"] / metadata["metadata"]["fps"]
            if abs(actual_duration - payload["window"]["renderer_target_duration_s"]) > 1e-4:
                raise ValueError("Renderer cache window changed after feature extraction")
            model.train()
            renderer.train()  # Enables checkpointing; parameters remain frozen.
            optimizer.zero_grad(set_to_none=True)
            generator = torch.Generator(device="cuda:0").manual_seed(args.seed + 100000 + step)
            with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
                u = model(condition)
                prepared = strategy.prepare_training_inputs(with_text(av, text), sampler)
                prepared = append_conditions(prepared, u)
                lav, components = av_loss(renderer, strategy, prepared)
                feature = model.feature_losses(u, targets, generator) if args.recipe == "joint" else {}
                total = cfg.av_weight * lav
                if feature:
                    total = total + cfg.feature_video_weight * feature["video"] + cfg.feature_audio_weight * feature["audio"]
            grads = {}
            if args.command == "gate":
                norms = [norm.weight for norm in model.connector.memory.norms]
                # One DiT backward traversal audits both U tensors and every selected layer.
                values = gradient_norms(lav, [u.video, u.audio, *norms])
                grads.update(zip(("av_to_u_video", "av_to_u_audio"), values[:2]))
                grads.update({f"av_to_layer_{layer}": value for layer, value in zip(cfg.hidden_layers, values[2:])})
                for mod, loss in feature.items():
                    values = gradient_norms(loss, [getattr(u, mod), *norms])
                    grads["feature_to_u_" + mod] = values[0]
                    grads.update({f"feature_{mod}_to_layer_{layer}": value
                                  for layer, value in zip(cfg.hidden_layers, values[1:])})
                required = {k: v for k, v in grads.items()
                            if k.startswith("av_to_u_") or (k.startswith("av_to_layer_") and step > step0 + 1)
                            or step > step0 + 2}
                if not all(np.isfinite(v) and v > 0 for v in required.values()):
                    raise RuntimeError(f"Missing shared-condition gradients: {grads}")
            total.backward()
            if args.command == "gate":
                for layer, layer_norm in zip(cfg.hidden_layers, model.connector.memory.norms):
                    grad = layer_norm.weight.grad
                    value = float(grad.float().norm()) if grad is not None else 0.0
                    grads[f"total_to_layer_{layer}_norm"] = value
                    # The zero-initialized final Connector opens the upstream path after one update.
                    if step > step0 + 1 and (not np.isfinite(value) or value <= 0):
                        raise RuntimeError(f"Missing layer-fusion gradient at layer {layer}: {value}")
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
            if not torch.isfinite(total):
                raise FloatingPointError("Nonfinite loss")
            optimizer.step()
            if any(p.requires_grad or p.grad is not None or p._version != version for p, version in frozen):
                raise RuntimeError("Frozen renderer was updated")
            row = dict(step=step, loss=float(total.detach()), av=float(lav.detach()), feature={k: float(v.detach()) for k, v in feature.items()},
                       grad_norm=float(norm), gradients=grads, sample_id=payload["sample_id"])
            log.write(json.dumps(row) + "\n")
            log.flush()
            metrics.append(row)
            event("p2_step", **row)
            last = step
            if step % args.save_every == 0 or step == step0 + args.steps:
                path = args.output / f"extractor_step_{step:06d}.pt"
                save(path, model, optimizer, step, identity)
                export_connector(path, path.with_name(path.stem + "_connector.pt"))
                checkpoints.append(str(path))
                saved = step
    if last != saved:
        path = args.output / f"extractor_step_{last:06d}.pt"
        save(path, model, optimizer, last, identity)
        export_connector(path, path.with_name(path.stem + "_connector.pt"))
        checkpoints.append(str(path))
    if args.command != "gate":
        return dict(training_completed_steps=last - step0, checkpoints=checkpoints, quality_pass=None,
                    stopped=(args.output / "STOP").exists())
    if not metrics:
        raise ValueError("Gate stopped before an update")
    # Verify exact restore and connector-only deployment without touching existing outputs.
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        before = model(condition)
    load(checkpoints[-1], model, identity, optimizer)
    from .connector import FusionConnector
    deploy = FusionConnector(cfg).cuda().eval()
    checkpoint = Path(checkpoints[-1])
    load(checkpoint.with_name(checkpoint.stem + "_connector.pt"), deploy, identity, connector_only=True)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        after = deploy(condition)
        if not torch.equal(before.video, after.video) or not torch.equal(before.audio, after.audio):
            raise RuntimeError("Connector-only inference differs after checkpoint restore")
        inputs = append_conditions(strategy.prepare_training_inputs(with_text(av, text), sampler), after)
        sampled = sample(renderer, inputs, args.sampling_steps)
        video, audio = unpack_latents(sampled, av)
    save_new(args.output / "sampled_latents.pt", dict(video=video.cpu(), audio=audio.cpu(), source="gt_text_predicted_U"))
    peak = torch.cuda.max_memory_allocated() / 1e9
    # Release expensive model/graphs before VAE decode.
    del before, after, deploy, model, optimizer, renderer, inputs, prepared, u, total, lav, feature, components, frozen
    unload()
    decoded = decode_and_save(cfg, video, audio, float(av["latents"]["fps"][0]), args.output / "engineering_sample.mp4")
    return dict(engineering_pass=True, quality_pass=None, recipe=args.recipe, checkpoints=checkpoints,
                frozen_renderer=True, deployment_exact=True, metrics=metrics, decoded=decoded,
                tiny=args.tiny, pretrained_models=True, peak_allocated_gb=peak)


def run(args):
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    cfg = build_config(args)
    write_json_new(args.output / "run.json", dict(version=VERSION, config=cfg.to_dict(),
        args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}))
    try:
        if args.command == "prepare":
            index = prepare_cache(args, cfg)
            result = dict(prepared=len(index["entries"]), quality_pass=None)
        elif args.command in {"train", "gate"}:
            result = run_training(args, cfg)
        else:
            from .inference import run_inference
            result = run_inference(args, cfg)
        write_json_new(args.output / "completed.json", result)
        event("completed", **result)
    except Exception:
        write_json_new(args.output / "failed.json", dict(traceback=traceback.format_exc()))
        raise
