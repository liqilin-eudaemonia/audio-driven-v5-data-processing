"""Single-GPU P/C0/A with stage transitions, strict resume, and real input contracts."""
from dataclasses import asdict
import math
from pathlib import Path
import random
import time

import torch

from .common import digest, read, resolve, sha, source_contract, write, write_once
from .features import (cache_dir, cpu, load_av, load_native, load_text, native_contract,
                       paths, renderer_contract, save, validate_part)
from .media import selection
from .prompts import contract


def build(cfg, stage, steps=None):
    from omni_av_p3.model import Config as PlannerConfig
    from omni_av_p3.data import load_native as load_model
    from omni_av_p2.config import Config as RendererConfig
    from omni_av_p2.ltx import load_renderer
    from omni_av_p21.config import LoRAConfig
    from omni_av_p21.residual_performance import configure_grouped_checkpointing
    from omni_av_p3_online.model import OnlinePlanner, OnlineTraining, OnlineConfig, GenerationConnector
    torch.manual_seed(cfg["seed"])
    pcfg = PlannerConfig(**cfg["planner"])
    runtime, processor = load_model(pcfg, "cuda:0")
    del processor
    planner = OnlinePlanner(runtime.thinker, pcfg, "lora")
    planner.connectors.cuda(); planner.decoders.cuda()
    planner.mask_token.data = planner.mask_token.data.cuda()
    planner.modality_embedding.data = planner.modality_embedding.data.cuda()
    planner.thinker.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    for name in ("visual", "audio_tower", "lm_head"):
        getattr(planner.thinker, name).cpu()
    if stage == "P":
        renderer, strategy, sampler = torch.nn.Identity(), None, None
    else:
        renderer, strategy, sampler = load_renderer(RendererConfig(**cfg["renderer"]), device="cuda:0")
        configure_grouped_checkpointing(renderer, 1)
    online = dict(cfg["online"])
    if steps is not None:
        online["planner_warmup_steps"] = min(50, max(1, math.floor(steps * .1)))
    t = cfg["training"]
    model = OnlineTraining(planner, renderer, strategy, stage, OnlineConfig(**online),
        None if stage == "P" else GenerationConnector(pcfg.hidden_size).cuda(),
        LoRAConfig(rank=t["lora_rank"], alpha=t["lora_alpha"]))
    return model, strategy, sampler


def checked(path):
    marker = read(Path(path).with_suffix(".complete.json"))
    if sha(path) != marker["sha256"]:
        raise ValueError("Checkpoint checksum mismatch")
    return torch.load(path, map_location="cpu", weights_only=True)


def restore_model(model, saved, transition=False):
    if tuple(saved["planner_names"]) != model.planner_names:
        raise ValueError("Planner trainable ordering changed")
    old = saved["model"]
    current = model.state_dict()
    if transition:
        if (saved["stage"], model.stage) not in (("P", "C0"), ("C0", "A")):
            raise ValueError("Only P→C0→A transitions are supported")
        allowed_new = ("connector.",) if model.stage == "C0" else ("adapters.",)
        if any(k not in old and not k.startswith(allowed_new) for k in current):
            raise ValueError("Unexpected new stage parameters")
    elif saved["stage"] != model.stage or set(old) != set(current):
        raise ValueError("Checkpoint stage mismatch")
    if any(k not in current or old[k].shape != current[k].shape or not torch.isfinite(old[k]).all() for k in old):
        raise ValueError("Invalid checkpoint tensors")
    model.load_state_dict(dict(current, **old), strict=True)


def identity(cfg, manifest, rows, stage, steps, s0, oracle_fit, initialization):
    return dict(version=cfg["version"], config=cfg, manifest=digest(manifest),
                rows=[r["sample_id"] for r in rows], stage=stage, steps=steps, s0=s0,
                oracle_fit=oracle_fit, initialization=initialization, prompts=contract(),
                native=digest(native_contract(cfg)), renderer=digest(renderer_contract(cfg)), code=source_contract(cfg))


def train(cfg, args):
    from omni_av.ltx_runtime import with_text
    from omni_av_p2.checkpoint import capture_rng, restore_rng
    torch.set_num_threads(2)
    if args.split != "train" or args.resume and args.checkpoint:
        raise ValueError("Train split only; resume and stage initialization are mutually exclusive")
    if args.oracle_fit and (not args.s0 or args.stage != "C0"):
        raise ValueError("Oracle fit is an S0 C0 diagnostic only")
    manifest, rows = selection(cfg, args)
    if not rows:
        raise ValueError("Empty training selection")
    t, root = cfg["training"], resolve(cfg["output"])
    budget = t[args.stage]
    steps = min(budget["steps"], math.ceil(len(rows) * budget["epochs"] / t["batch"]))
    if args.s0:
        steps = args.steps or 2
    elif args.steps is not None or args.limit:
        raise ValueError("Partial training/step overrides must be explicitly marked --s0")
    if steps < 1 or t["batch"] < 1:
        raise ValueError("Invalid budget")
    out = args.output or root / "runs" / (("s0_" if args.s0 else "") + args.stage)
    out = Path(out)
    source = checked(args.resume or args.checkpoint) if args.resume or args.checkpoint else None
    if args.stage != "P" and source is None:
        raise ValueError("C0/A require a previous-stage checkpoint")
    if args.stage == "P" and args.checkpoint:
        raise ValueError("P initializes from original pretrained Omni; use --resume for continuation")
    init_sha = source["identity"]["initialization"] if args.resume else sha(args.checkpoint) if args.checkpoint else None
    ident = identity(cfg, manifest, rows, args.stage, steps, args.s0, args.oracle_fit, init_sha)
    if source:
        for key in ("version", "config", "manifest", "native", "renderer", "prompts", "code"):
            if source["identity"][key] != ident[key]:
                raise ValueError(f"Checkpoint input/config/code mismatch: {key}")
        if not args.s0 and source["identity"]["s0"]:
            raise ValueError("S0 diagnostic cannot initialize a production stage")
        if source["identity"]["oracle_fit"] and not args.oracle_fit:
            raise ValueError("Oracle diagnostic cannot initialize the main experiment")
        if args.resume and source["identity"] != ident:
            raise ValueError("Resume identity differs")
    if (out / "run.json").exists() and not args.resume:
        raise FileExistsError("Run exists; resume its checkpoint or use a new output")
    # Preflight all selected caches before allocating models; targets remain CPU.
    for row in rows:
        load_native(root, row, cfg, targets=True)
        if args.stage != "P":
            load_av(root, row, cfg, "cpu")
    text = None if args.stage == "P" else load_text(root, cfg, "cuda:0")
    model, strategy, sampler = build(cfg, args.stage, steps)
    if source:
        restore_model(model, source, transition=not bool(args.resume))
    optimizer = torch.optim.AdamW(model.optimizer_groups(t["connector_lr"], t["dit_lr"],
        t["planner_lr"], t["heads_lr"]), weight_decay=0.)
    start = 0
    if args.resume:
        optimizer.load_state_dict(source["optimizer"])
        restore_rng(source["rng"])
        start = source["step"]
    del source
    write_once(out / "run.json", ident)
    model.train()
    frozen = [(p, p._version) for m in (model.planner, model.renderer) for p in m.parameters() if not p.requires_grad]
    trainable_before = [p.detach().clone() for p in model.parameters() if p.requires_grad] if args.s0 else []
    order = list(range(len(rows)))
    random.Random(cfg["seed"]).shuffle(order)
    base_lr = {g["name"]: g["lr"] for g in optimizer.param_groups}
    warmup = max(1, math.ceil(steps * t["warmup_fraction"]))
    def checkpoint(step):
        path = out / f"step_{step:06d}.pt"
        save(path, dict(stage=model.stage, step=step, identity=ident, planner_names=model.planner_names,
                        model=cpu(model.state_dict()), optimizer=optimizer.state_dict(), rng=capture_rng()))
        write(path.with_suffix(".complete.json"), dict(step=step, sha256=sha(path)))
        return path
    last = start
    with (out / "train.jsonl").open("a") as log:
        for step in range(start + 1, steps + 1):
            if (out / "STOP").exists() or args.stop_after is not None and step > args.stop_after:
                break
            begin = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            sums = {}
            for group in optimizer.param_groups:
                factor = min(1., step / warmup)
                if args.stage == "A" and group["name"] in ("omni_lora", "feature_heads"):
                    factor = max(0., min(1., (step - model.config.planner_warmup_steps) / warmup))
                group["lr"] = base_lr[group["name"]] * factor
            for micro in range(t["batch"]):
                ordinal = (step - 1) * t["batch"] + micro
                # A new deterministic shuffle per epoch, resumable from update number.
                epoch, offset = divmod(ordinal, len(rows))
                sequence = list(range(len(rows)))
                random.Random(cfg["seed"] + epoch).shuffle(sequence)
                row = rows[sequence[offset]]
                seed = cfg["seed"] + ordinal
                torch.manual_seed(seed)
                layout, targets = load_native(root, row, cfg, targets=True, device="cuda:0")
                prepared = None
                if args.stage != "P":
                    av = load_av(root, row, cfg, "cuda:0")
                    prepared = strategy.prepare_training_inputs(with_text(av, text), sampler)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    values = model(layout, targets, prepared, seed, step,
                                   torch.zeros_like(layout.valid) if args.oracle_fit else None)
                if not all(torch.isfinite(v).all() for v in values.values()):
                    raise FloatingPointError("Nonfinite training loss")
                (values["loss"] / t["batch"]).backward()
                for k, v in values.items():
                    sums[k] = sums.get(k, 0.) + float(v.detach()) / t["batch"]
                del layout, targets, prepared, values
            grads = {g["name"]: math.sqrt(sum(float(p.grad.float().square().sum()) for p in g["params"] if p.grad is not None))
                     for g in optimizer.param_groups}
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), t["grad_clip"], error_if_nonfinite=True)
            optimizer.step()
            model.assert_ownership()
            if any(p.grad is not None or p._version != version for p, version in frozen):
                raise RuntimeError("Frozen parameters changed")
            record = dict(step=step, loss=sums, grad_norm=float(norm), group_grads=grads,
                          seconds=time.monotonic() - begin, peak_memory_gb=torch.cuda.max_memory_allocated() / 1e9)
            import json
            log.write(json.dumps(record) + "\n"); log.flush()
            print(record, flush=True)
            last = step
            if step % t["save_every"] == 0 or step == steps or step == args.stop_after:
                checkpoint(step)
            if not args.s0 and step % min(t["validate_every"], max(1, math.ceil(len(rows) / t["batch"]))) == 0:
                evaluate(model, cfg, manifest, root, out, step, strategy, sampler, text)
    final = checkpoint(last)
    changed = None
    if args.s0:
        changed = any(not torch.equal(before, after) for before, after in zip(trainable_before, (p for p in model.parameters() if p.requires_grad)))
        if last > start and not changed:
            raise RuntimeError("No trainable parameter updated")
    write(out / "status.json", dict(step=last, target_steps=steps, stopped=last < steps,
                                    checkpoint=str(final), trainable_changed=changed, frozen_unchanged=True,
                                    s0=args.s0, oracle_fit=args.oracle_fit, quality_evaluated=False))


@torch.no_grad()
def evaluate(model, cfg, manifest, root, out, step, strategy, sampler, text):
    from omni_av.ltx_runtime import with_text
    from omni_av_p2.checkpoint import capture_rng, restore_rng
    rng = capture_rng()
    records = []
    model.eval()
    try:
        for i, row in enumerate(r for r in manifest["windows"] if r["split"] == "dev"):
            seed = cfg["seed"] + 100000 + i
            torch.manual_seed(seed)
            layout, targets = load_native(root, row, cfg, targets=True, device="cuda:0")
            prepared = None if model.stage == "P" else strategy.prepare_training_inputs(with_text(load_av(root, row, cfg, "cuda:0"), text), sampler)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                values = model(layout, targets, prepared, seed, step, layout.valid.clone())
            records.append(dict(sample_id=row["sample_id"], values={k: float(v) for k, v in values.items()}))
        write(out / f"dev_{step:06d}.json", dict(rows=records, note="All-mask objective only; use generation matrix for quality"))
    finally:
        restore_rng(rng)
        model.train()
