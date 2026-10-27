"""Single-GPU Stage II/III entry points with explicit paired GT supervision."""
import json

import torch

from .checkpoint import initialize_delta, restore_checkpoint, save_checkpoint
from .config import PlanConfig
from .data import TextPlan, save_new
from .ltx_runtime import NativeTextConditioner, audit_target_window, load_av_targets, load_renderer, render_loss, with_text
from .omni import load_planner
from .renderer import attach_plan
from .runner import contract, event, read_batch


def run_av_training(args):
    if args.cache is None or args.steps < 1 or args.save_every < 1:
        raise ValueError("AV training requires cache index, positive steps/save interval")
    cfg = PlanConfig()
    index = json.loads(args.cache.read_text())
    if index["contract"] != contract(cfg):
        raise ValueError("Cache contract mismatch")
    if not index["paths"]:
        raise ValueError("Empty training cache")
    joint = args.command == "train-joint"
    if joint and not args.resume and (args.planner_checkpoint is None or args.renderer_checkpoint is None):
        raise ValueError("Joint training requires completed planner and renderer initializations or --resume")
    # Encode text sequentially before keeping the AV model resident. Cache is immutable per run.
    text_paths = []
    text_runtime = NativeTextConditioner(cfg)
    try:
        for i, path in enumerate(index["paths"]):
            payload = torch.load(path, weights_only=True, map_location="cpu")
            if payload["contract"] != contract(cfg) or payload["split"] != "train":
                raise ValueError("Text cache sample contract mismatch")
            text = text_runtime.encode(TextPlan(**payload["example"]["text_plan"]))
            target = args.output / "text" / f"{i:06d}.pt"
            save_new(target, {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in text.items()})
            text_paths.append(target)
            del text
    finally:
        text_runtime.close()
    bundle, strategy, sampler = load_renderer(cfg)
    if joint:
        planner, _ = load_planner(cfg)
        bundle.planner = planner
        if not args.resume:
            initialize_delta(args.planner_checkpoint, planner, contract(cfg))
            renderer_names = {n for n, p in bundle.named_parameters()
                              if n.startswith("renderer.") and "lora_" in n and p.requires_grad}
            initialize_delta(args.renderer_checkpoint, bundle, contract(cfg), expected_names=renderer_names)
    else:
        # Stage II has no planning input or planning loss.
        bundle.connector.requires_grad_(False)
        for name, param in bundle.renderer.named_parameters():
            if "omni_attn2" in name:
                param.requires_grad_(False)
    params = [p for p in bundle.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=1e-5)
    run_contract = dict(model=contract(cfg), stage="joint" if joint else "renderer",
                        cache_paths=index["paths"], lr=1e-5, tiny=args.tiny)
    initial = restore_checkpoint(args.resume, bundle, optimizer, run_contract) if args.resume else 0
    frozen = [(p, p._version) for p in bundle.parameters() if not p.requires_grad]
    checkpoints = []
    last_step = initial
    saved_step = initial
    with (args.output / "train.jsonl").open("x") as log:
        for step in range(initial + 1, initial + args.steps + 1):
            if (args.output / "STOP").exists():
                break
            index_i = (step - 1) % len(index["paths"])
            data, payload = read_batch(index["paths"][index_i], cfg)
            text = torch.load(text_paths[index_i], map_location="cuda:0", weights_only=True)
            av, metadata = load_av_targets(args.data_root, payload["sample_id"], tiny=args.tiny)
            window = audit_target_window(payload, metadata)
            bundle.train()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
                prepared = strategy.prepare_training_inputs(with_text(av, text), sampler)
                planning = bundle.planner(data) if joint else None
                if planning is not None:
                    prepared = attach_plan(prepared, bundle.connector, planning["hidden"],
                                           planning["renderer_mask"])
                av_loss, components = render_loss(bundle, strategy, prepared)
                loss = av_loss + (planning["loss"] if planning else 0)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
            optimizer.step()
            last_step = step
            if any(p.grad is not None or p._version != version for p, version in frozen):
                raise RuntimeError("Frozen parameters changed")
            row = dict(step=step, loss=float(loss), av=float(av_loss), gradient_norm=float(norm),
                       planning=float(planning["loss"]) if planning else None, target_window=window)
            log.write(json.dumps(row) + "\n")
            log.flush()
            event("av_training", **row)
            if step % args.save_every == 0 or step == initial + args.steps or (args.output / "STOP").exists():
                path = args.output / f"{run_contract['stage']}_step_{step:06d}.pt"
                save_checkpoint(path, bundle, optimizer, step, run_contract)
                checkpoints.append(str(path))
                saved_step = step
    if last_step != saved_step:
        path = args.output / f"{run_contract['stage']}_step_{last_step:06d}.pt"
        save_checkpoint(path, bundle, optimizer, last_step, run_contract)
        checkpoints.append(str(path))
    return dict(engineering_pass=True, quality_pass=None, stage=run_contract["stage"],
                checkpoints=checkpoints, stopped=(args.output / "STOP").exists())
