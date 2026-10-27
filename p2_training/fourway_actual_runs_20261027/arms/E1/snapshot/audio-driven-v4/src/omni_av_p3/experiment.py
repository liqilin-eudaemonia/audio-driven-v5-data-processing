"""Paired fixed-GT planning pilot. Feature metrics are not AV quality metrics."""
from contextlib import nullcontext
from dataclasses import asdict, replace
import hashlib
import json
import time

import numpy as np
import torch
from torch.nn import functional as F

from omni_av.data import file_hash, save_new, write_json_new
from .data import load_cache, load_native
from .model import Layout, Planner


def autocast(device):
    return torch.autocast("cuda", dtype=torch.bfloat16) if str(device).startswith("cuda") else nullcontext()


def digest(state):
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        h.update(name.encode())
        h.update(value.contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def mean_teachers(rows):
    means = []
    for mod in (0, 1):
        # Train only, token-weighted channel means.
        count, total = 0, None
        for row in rows:
            chosen = row["layout"]["valid"] & (row["layout"]["modalities"] == mod)
            features = row["targets"][chosen].double()
            count += len(features)
            total = features.sum(0) if total is None else total + features.sum(0)
        means.append((total / count).float())
    return torch.stack(means)


def metrics(predicted, targets, layout):
    result = {}
    for mod, name in enumerate(("video", "audio")):
        selected = layout.valid & (layout.modalities == mod)
        a, b = predicted[selected].float(), targets[selected].float()
        result[name] = dict(mse=float((a - b).square().mean()),
                           cosine=float(F.cosine_similarity(a, b, dim=-1).mean()),
                           predicted_rms=float(a.square().mean().sqrt()),
                           target_rms=float(b.square().mean().sqrt()))
    return result


@torch.no_grad()
def evaluate(model, rows, means, device, full=True):
    model.eval()
    records = []
    for i, row in enumerate(rows):
        layout = Layout(**row["layout"]).to(device)
        targets = row["targets"].to(device)
        baseline = means.to(device)[layout.modalities]
        base_metrics = metrics(baseline, targets, layout)
        for seed in model.cfg.eval_seeds:
            with autocast(device):
                partial = model.loss(layout, targets, seed, all_mask=False)
                full_loss = model.loss(layout, targets, seed, all_mask=True)
                record = dict(sample_id=row["sample_id"], seed=seed,
                    partial_fm={k: float(v) for k, v in partial.items()},
                    full_mask_fm={k: float(v) for k, v in full_loss.items()}, mean_baseline=base_metrics)
                if full:
                    one, _ = model.sample(layout, seed, steps=1)
                    pred, remaining = model.sample(layout, seed)
                    # Mismatch only the prefix; preserve target layout, target, mask order and noise.
                    other = Layout(**rows[(i + 1) % len(rows)]["layout"]).to(device)
                    wrong_layout = replace(layout, prefix=other.prefix, prefix_valid=other.prefix_valid,
                                           prefix_positions=other.prefix_positions)
                    wrong, _ = model.sample(wrong_layout, seed)
                    record.update(one_round=metrics(one, targets, layout),
                                  iterative=metrics(pred, targets, layout),
                                  shuffled_condition=metrics(wrong, targets, layout), remaining=remaining,
                                  shuffled_sample_id=rows[(i + 1) % len(rows)]["sample_id"])
            records.append(record)
        print(json.dumps(dict(evaluated=row["sample_id"], arm=model.arm)), flush=True)
    return records


def train_arm(cfg, cache_path, output, arm, device):
    output.mkdir(parents=True, exist_ok=False)
    rows = load_cache(cache_path, cfg)
    if len(rows["dev"]) < 2:
        raise ValueError("Condition permutation requires at least two Dev samples")
    write_json_new(output / "protocol.json", dict(config=asdict(cfg), arm=arm,
        cache_path=str(cache_path.resolve()), cache_sha256=file_hash(cache_path),
        torch=torch.__version__, device=str(device)))
    torch.manual_seed(cfg.seed)
    runtime, _ = load_native(cfg, device)
    model = Planner(runtime.thinker, cfg, arm)
    model.connectors.to(device)
    model.decoders.to(device)
    model.mask_token.data = model.mask_token.data.to(device)
    model.modality_embedding.data = model.modality_embedding.data.to(device)
    model.thinker.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    heads, adapters = [], []
    groups = {"heads": 0., "lora": 0., "mask": 0.}
    for name, p in model.named_parameters():
        if p.requires_grad:
            (adapters if name.startswith("thinker.") else heads).append(p)
        elif p.grad is not None:
            raise RuntimeError("Frozen parameter already has a gradient")
    common = {k: v for k, v in model.trainable_state().items() if not k.startswith("thinker.")}
    initial = dict(common_sha256=digest(common), common_parameters=sum(p.numel() for p in heads),
                   lora_parameters=sum(p.numel() for p in adapters))
    write_json_new(output / "initialization.json", initial)
    del common
    optimizer = torch.optim.AdamW([dict(params=heads, lr=cfg.heads_lr),
                                   dict(params=adapters, lr=cfg.lora_lr)], weight_decay=0)
    means = mean_teachers(rows["train"])
    before = evaluate(model, rows["dev"], means, device, full=False)
    write_json_new(output / "before.json", before)
    rng = np.random.default_rng(cfg.seed)
    order = []
    while len(order) < cfg.steps:
        order.extend(rng.permutation(len(rows["train"])).tolist())
    started = time.monotonic()
    with (output / "train.jsonl").open("x") as log:
        for step, index in enumerate(order[:cfg.steps]):
            row = rows["train"][index]
            layout = Layout(**row["layout"]).to(device)
            targets = row["targets"].to(device)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            with autocast(device):
                losses = model.loss(layout, targets, cfg.seed + step * 1009)
                total = sum(losses.values())
            if not torch.isfinite(total):
                raise RuntimeError("Nonfinite loss")
            total.backward()
            for name, p in model.named_parameters():
                if not p.requires_grad and p.grad is not None:
                    raise RuntimeError(f"Frozen gradient: {name}")
                if p.grad is not None:
                    if not torch.isfinite(p.grad).all():
                        raise RuntimeError(f"Nonfinite gradient: {name}")
                    group = "lora" if name.startswith("thinker.") else "mask" if name == "mask_token" else "heads"
                    groups[group] = max(groups[group], float(p.grad.detach().abs().max()))
            norm = torch.nn.utils.clip_grad_norm_(heads + adapters, 1., error_if_nonfinite=True)
            optimizer.step()
            record = dict(step=step + 1, sample_id=row["sample_id"], seed=cfg.seed + step * 1009,
                **{k: float(v.detach()) for k, v in losses.items()}, grad_norm=float(norm),
                elapsed_s=time.monotonic() - started)
            log.write(json.dumps(record) + "\n")
            log.flush()
            if step % 10 == 0 or step + 1 == cfg.steps:
                print(json.dumps(dict(arm=arm, **record)), flush=True)
    train_seconds = time.monotonic() - started
    save_new(output / "final.pt", dict(config=asdict(cfg), arm=arm, step=cfg.steps,
        cache_sha256=file_hash(cache_path), state=model.trainable_state(), optimizer=optimizer.state_dict()))
    if groups["heads"] == 0 or groups["mask"] == 0 or (arm == "lora" and groups["lora"] == 0):
        raise RuntimeError(f"Training gradient route did not open: {groups}")
    after = evaluate(model, rows["dev"], means, device, full=True)
    write_json_new(output / "after.json", after)
    write_json_new(output / "completed.json", dict(arm=arm, steps=cfg.steps,
        train_seconds=train_seconds, gradient_maxima=groups, initialization=initial,
        peak_allocated_bytes=torch.cuda.max_memory_allocated() if str(device).startswith("cuda") else None,
        conclusion="Engineering run completed; compare paired held-out metrics before choosing next phase."))


def compare(frozen, lora, output):
    protocols = [json.loads((p / "protocol.json").read_text()) for p in (frozen, lora)]
    inits = [json.loads((p / "initialization.json").read_text()) for p in (frozen, lora)]
    for p in (frozen, lora):
        if not (p / "completed.json").exists():
            raise ValueError("Both runs must complete")
    if protocols[0]["arm"] != "frozen" or protocols[1]["arm"] != "lora":
        raise ValueError("Comparison arms reversed/invalid")
    if protocols[0]["config"] != protocols[1]["config"] or protocols[0]["cache_sha256"] != protocols[1]["cache_sha256"]:
        raise ValueError("Paired protocol mismatch")
    if inits[0]["common_sha256"] != inits[1]["common_sha256"]:
        raise ValueError("Common head initialization mismatch")
    logs = [[json.loads(line) for line in (p / "train.jsonl").read_text().splitlines()] for p in (frozen, lora)]
    if [(x["sample_id"], x["seed"]) for x in logs[0]] != [(x["sample_id"], x["seed"]) for x in logs[1]]:
        raise ValueError("Training examples/noise differ")
    data = [json.loads((p / "after.json").read_text()) for p in (frozen, lora)]
    if [(r["sample_id"], r["seed"]) for r in data[0]] != [(r["sample_id"], r["seed"]) for r in data[1]]:
        raise ValueError("Evaluation pairs differ")
    summary = dict(protocol=protocols[0]["config"], common_initialization=inits[0]["common_sha256"],
                   cache_sha256=protocols[0]["cache_sha256"], arms={}, paired={})
    for name, records in zip(("frozen", "lora"), data):
        summary["arms"][name] = {}
        for mod in ("video", "audio"):
            summary["arms"][name][mod] = {
                key: float(np.mean([r[key][mod]["mse"] for r in records]))
                for key in ("one_round", "iterative", "shuffled_condition", "mean_baseline")}
            summary["arms"][name][mod]["full_mask_fm"] = float(np.mean([r["full_mask_fm"][mod] for r in records]))
            summary["arms"][name][mod]["cosine"] = float(np.mean([r["iterative"][mod]["cosine"] for r in records]))
    for mod in ("video", "audio"):
        by_sample = {}
        for a, b in zip(*data):
            by_sample.setdefault(a["sample_id"], []).append(b["iterative"][mod]["mse"] - a["iterative"][mod]["mse"])
        differences = np.array([np.mean(v) for v in by_sample.values()])
        rng = np.random.default_rng(29)
        boot = rng.choice(differences, (4000, len(differences)), replace=True).mean(1)
        summary["paired"][mod] = dict(lora_minus_frozen_mse=float(differences.mean()),
            sample_bootstrap_95_percent=np.quantile(boot, [.025, .975]).tolist(),
            independent_dev_samples=len(differences))
    summary["limitations"] = ["Small single-training-seed pilot; bootstrap is exploratory.",
        "GT reply/timing supplied to both arms; autonomous content/duration not evaluated.",
        "Native feature point errors do not establish spoken content, expression or AV quality.",
        "No renderer trained or evaluated; equal updates/exposure, not equal FLOPs."]
    write_json_new(output, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary
