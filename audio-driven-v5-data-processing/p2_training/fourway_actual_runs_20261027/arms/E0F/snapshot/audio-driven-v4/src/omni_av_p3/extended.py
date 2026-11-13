"""Measured acceleration, exact-step continuation and small-set condition checks."""
from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np
import torch

from omni_av.data import file_hash, save_new, write_json_new
from .accelerated import FastPlanner, Resident, audit_gradients, execution_mode, optimizer_for, prepare_mask, resident
from .data import load_cache, load_native
from .diagnostics import restore
from .experiment import autocast, digest, evaluate, mean_teachers
from .model import Config


def load_experiment(cache, arm, device, checkpoint_path=None):
    cfg = Config(**json.loads(cache.read_text())["contract"]["config"])
    rows = load_cache(cache, cfg)
    torch.manual_seed(cfg.seed)
    runtime, _ = load_native(cfg, device)
    model = FastPlanner(runtime.thinker, cfg, arm)
    model.connectors.to(device)
    model.decoders.to(device)
    model.mask_token.data = model.mask_token.data.to(device)
    model.modality_embedding.data = model.modality_embedding.data.to(device)
    checkpoint = None
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if checkpoint["cache_sha256"] != file_hash(cache):
            raise ValueError("Resume cache mismatch")
        if checkpoint["config"] != json.loads(json.dumps(vars(cfg))):
            # Pilot saves tuples while JSON uses lists; compare normalized representations.
            if json.loads(json.dumps(checkpoint["config"])) != json.loads(json.dumps(vars(cfg))):
                raise ValueError("Resume model/training recipe mismatch")
        restore(model, checkpoint)
    return model, rows, checkpoint


def example_order(count, steps, seed):
    rng = np.random.default_rng(seed)
    order = []
    while len(order) < steps:
        order.extend(rng.permutation(count).tolist())
    return order[:steps]


@torch.no_grad()
def condition_check(model, data, device, seeds=(137, 251), probe=True):
    model.eval()
    records, probes = [], []
    for i, a in enumerate(data):
        b = data[(i + 1) % len(data)]
        wrong_layout = replace(a.layout, prefix=b.layout.prefix, prefix_valid=b.layout.prefix_valid,
                               prefix_positions=b.layout.prefix_positions)
        wrong = Resident(wrong_layout, a.targets, a.cpu_layout, a.sample_id)
        for seed in seeds:
            masked, ids = prepare_mask(a, model.cfg, seed, all_mask=True)
            with autocast(device):
                correct = model.prepared_loss(a, masked, ids, seed)
                shuffled = model.prepared_loss(wrong, masked, ids, seed)
                high = model.prepared_loss(a, masked, ids, seed, sigma_override=1.)
                high_wrong = model.prepared_loss(wrong, masked, ids, seed, sigma_override=1.)
            records.append(dict(sample_id=a.sample_id, shuffled_id=b.sample_id, seed=seed,
                correct={k: float(v) for k, v in correct.items()}, shuffled={k: float(v) for k, v in shuffled.items()},
                sigma1_correct={k: float(v) for k, v in high.items()}, sigma1_shuffled={k: float(v) for k, v in high_wrong.items()}))
        if probe and i < 2:
            with autocast(device):
                h = model.hidden(a.layout, a.layout.valid, a.targets)
                wh = model.hidden(wrong.layout, wrong.layout.valid, a.targets)
                for mod, name in enumerate(("video", "audio")):
                    take = a.layout.modalities == mod
                    c, wc = model.connectors[mod](h[take].float()), model.connectors[mod](wh[take].float())
                    noise = torch.randn(c.shape, device=c.device, generator=torch.Generator(device=c.device).manual_seed(39))
                    t = torch.full((len(c),), 1000., device=c.device, dtype=c.dtype)
                    u = model.decoders[mod].net(noise.to(c.dtype), t, c)
                    wu = model.decoders[mod].net(noise.to(c.dtype), t, wc)
                    rms = lambda x: float(x.float().square().mean().sqrt())
                    probes.append(dict(sample_id=a.sample_id, modality=name, hidden_rms=rms(h[take]),
                        hidden_change_rms=rms(h[take] - wh[take]), condition_rms=rms(c),
                        condition_change_rms=rms(c - wc), velocity_rms=rms(u), velocity_change_rms=rms(u - wu)))
    summary = {}
    for name in ("video", "audio"):
        summary[name] = {}
        for prefix in ("", "sigma1_"):
            correct = np.mean([r[prefix + "correct"][name] for r in records])
            wrong = np.mean([r[prefix + "shuffled"][name] for r in records])
            summary[name][prefix + "correct"] = float(correct)
            summary[name][prefix + "shuffled"] = float(wrong)
            summary[name][prefix + "shuffle_increase_percent"] = float(100 * (wrong - correct) / correct)
    model._geometry.clear()
    return dict(summary=summary, records=records, probes=probes)


def benchmark(args):
    args.output.mkdir(parents=True, exist_ok=False)
    model, rows, checkpoint = load_experiment(args.cache, args.arm, args.device, args.resume)
    if checkpoint is None:
        raise ValueError("Benchmark requires a trained checkpoint, exercising nonzero upstream gradients")
    data = [resident(row, args.device) for row in rows["train"]]
    start = checkpoint["step"]
    total = args.benchmark_steps + 5
    order = example_order(len(data), start + total, model.cfg.seed)[start:]
    masks = [prepare_mask(data[index], model.cfg, model.cfg.seed + (start + i) * 1009)
             for i, index in enumerate(order)]
    timings, losses, final = {}, {}, {}
    for mode in ("reference", "accelerated"):
        restore(model, checkpoint)
        fast = mode == "accelerated"
        execution_mode(model, fast)
        opt, parameters = optimizer_for(model, checkpoint, fused=fast)
        model.train()
        torch.cuda.reset_peak_memory_stats()
        sequence, milliseconds = [], []
        for i, index in enumerate(order):
            step = start + i
            seed = model.cfg.seed + step * 1009
            opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            begin = time.perf_counter()
            with autocast(args.device):
                if fast:
                    loss = model.prepared_loss(data[index], *masks[i], seed)
                else:
                    # Original per-step transfers/checks/checkpointing and gradient scans.
                    model.train()
                    from .model import Layout, Planner
                    row = rows["train"][index]
                    original_hidden = model.hidden
                    model.hidden = lambda *a, **kw: Planner.hidden(model, *a, **kw)
                    try:
                        loss = model.loss(Layout(**row["layout"]).to(args.device), row["targets"].to(args.device), seed)
                    finally:
                        model.hidden = original_hidden
            sum(loss.values()).backward()
            if not fast:
                for p in model.parameters():
                    if p.grad is not None:
                        if not torch.isfinite(p.grad).all():
                            raise RuntimeError("Nonfinite reference gradient")
                        float(p.grad.abs().max())
            torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
            opt.step()
            torch.cuda.synchronize()
            elapsed = 1000 * (time.perf_counter() - begin)
            if i >= 5:
                milliseconds.append(elapsed)
            sequence.append({k: float(v.detach()) for k, v in loss.items()})
        timings[mode] = dict(mean_ms=float(np.mean(milliseconds)), median_ms=float(np.median(milliseconds)),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(), measured_updates=len(milliseconds), warmup_updates=5)
        losses[mode] = sequence
        final[mode] = model.trainable_state()
        del opt
    deltas = [float((final["reference"][k] - final["accelerated"][k]).abs().max()) for k in final["reference"]]
    paired_loss_differences = [abs(a[k] - b[k]) for a, b in zip(losses["reference"], losses["accelerated"]) for k in a]
    report = dict(arm=args.arm, timings=timings,
        speedup=timings["reference"]["mean_ms"] / timings["accelerated"]["mean_ms"],
        max_absolute_parameter_difference=max(deltas), max_absolute_loss_difference=max(paired_loss_differences),
        loss_traces=losses, same_samples_noise_and_optimizer_start=True,
        note="Fused optimizer changes floating point operation order; not a bitwise-equivalence claim.")
    if not np.isfinite(max(deltas)) or not np.isfinite(max(paired_loss_differences)):
        raise RuntimeError("Nonfinite speed comparison")
    write_json_new(args.output / "completed.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "loss_traces"}), flush=True)


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    model, rows, checkpoint = load_experiment(args.cache, args.arm, args.device, args.resume)
    start = checkpoint["step"] if checkpoint else 0
    if args.total_steps <= start:
        raise ValueError("Total steps must exceed restored step")
    if args.train_limit and checkpoint:
        raise ValueError("Do not silently change dataset while resuming")
    train_rows = rows["train"][:args.train_limit] if args.train_limit else rows["train"]
    if checkpoint and "train_ids" in checkpoint and checkpoint["train_ids"] != [r["sample_id"] for r in train_rows]:
        raise ValueError("Resume training identities changed")
    if len(train_rows) < 2:
        raise ValueError("Condition checks need two different examples")
    data = [resident(row, args.device) for row in train_rows]
    dev = [resident(row, args.device) for row in rows["dev"]]
    execution_mode(model, True)
    optimizer, parameters = optimizer_for(model, checkpoint, fused=True)
    common = {k: v for k, v in model.trainable_state().items() if not k.startswith("thinker.")}
    write_json_new(args.output / "protocol.json", dict(arm=args.arm, model_config=vars(model.cfg),
        total_steps=args.total_steps, start_step=start, train_ids=[r["sample_id"] for r in train_rows],
        dev_ids=[r["sample_id"] for r in rows["dev"]], cache_sha256=file_hash(args.cache),
        resume_sha256=file_hash(args.resume) if args.resume else None, starting_common_sha256=digest(common),
        optimizations=["resident_features", "prepared_mask_indices", "cached_attention_positions",
                       "checkpointing_disabled", "fused_adamw", "periodic_vector_gradient_audit"],
        batch_size=1, validation_every=args.validate_every, seed=model.cfg.seed))
    del common
    order = example_order(len(data), args.total_steps, model.cfg.seed)
    masks = [prepare_mask(data[order[step]], model.cfg, model.cfg.seed + step * 1009)
             for step in range(start, args.total_steps)]
    for item in data:
        model.geometry(item.layout, item.targets.dtype)
    checkpoints = sorted(set([start] + list(range(args.validate_every, args.total_steps + 1, args.validate_every)) + [args.total_steps]))
    begin_all = time.monotonic()
    initial = dict(step=start, train=condition_check(model, data, args.device),
                   dev=condition_check(model, dev, args.device))
    write_json_new(args.output / f"conditions_{start:06d}.json", initial)
    gradients, update_seconds = {}, 0.
    model.train()
    with (args.output / "train.jsonl").open("x") as log:
        for step in range(start, args.total_steps):
            index = order[step]
            seed = model.cfg.seed + step * 1009
            begin = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            with autocast(args.device):
                loss = model.prepared_loss(data[index], *masks[step - start], seed)
                total = sum(loss.values())
            total.backward()
            if step < start + 4 or (step + 1) % 100 == 0:
                audit = audit_gradients(model)
                gradients = {k: max(gradients.get(k, 0.), v) for k, v in audit.items()}
            norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
            optimizer.step()
            # A single packed transfer records losses/norm and completes this update's timing.
            scalar = torch.stack([loss["video"].detach(), loss["audio"].detach(), norm.detach()]).cpu().tolist()
            update_seconds += time.perf_counter() - begin
            log.write(json.dumps(dict(step=step + 1, sample_id=data[index].sample_id, seed=seed,
                video=scalar[0], audio=scalar[1], grad_norm=scalar[2], update_seconds=update_seconds)) + "\n")
            if (step + 1) % 25 == 0:
                log.flush()
                print(json.dumps(dict(arm=args.arm, samples=len(data), step=step + 1,
                    video=scalar[0], audio=scalar[1], update_seconds=update_seconds)), flush=True)
            if step + 1 in checkpoints:
                save_new(args.output / f"step_{step + 1:06d}.pt", dict(config=vars(model.cfg), arm=args.arm,
                    step=step + 1, cache_sha256=file_hash(args.cache), state=model.trainable_state(),
                    optimizer=optimizer.state_dict(), train_ids=[r["sample_id"] for r in train_rows]))
                result = dict(step=step + 1, train=condition_check(model, data, args.device),
                              dev=condition_check(model, dev, args.device))
                write_json_new(args.output / f"conditions_{step + 1:06d}.json", result)
                model.train()
    if any(gradients[k] <= 0 for k in ("mask", "connector", "decoder")) or (args.arm == "lora" and gradients["lora"] <= 0):
        raise RuntimeError(f"Gradient route did not open: {gradients}")
    after = evaluate(model, rows["dev"], mean_teachers(train_rows), args.device, full=True)
    write_json_new(args.output / "after.json", after)
    write_json_new(args.output / "completed.json", dict(arm=args.arm, start_step=start, step=args.total_steps,
        train_samples=len(data), update_seconds=update_seconds, elapsed_seconds=time.monotonic() - begin_all,
        gradients=gradients, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        conclusion="Planning-only diagnostic; evaluate condition dependence before choosing renderer work."))
