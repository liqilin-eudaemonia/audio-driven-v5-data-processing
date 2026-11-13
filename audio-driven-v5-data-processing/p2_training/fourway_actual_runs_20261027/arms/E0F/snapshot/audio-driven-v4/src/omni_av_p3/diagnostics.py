"""Post-pilot condition-use check with identical target, sigma and noise."""
from dataclasses import replace
import json

import numpy as np
import torch

from omni_av.data import file_hash, write_json_new
from .data import load_cache, load_native
from .experiment import autocast, digest
from .model import Config, Layout, Planner


def restore(model, checkpoint):
    required = {name: p for name, p in model.named_parameters() if p.requires_grad}
    if set(checkpoint["state"]) != set(required):
        raise ValueError("Checkpoint trainable parameter set mismatch")
    if checkpoint["arm"] != model.arm:
        raise ValueError("Checkpoint arm mismatch")
    with torch.no_grad():
        for name, p in required.items():
            if p.shape != checkpoint["state"][name].shape:
                raise ValueError(f"Checkpoint shape mismatch: {name}")
            p.copy_(checkpoint["state"][name].to(p))


@torch.no_grad()
def diagnose(run, cache_path, output, device):
    protocol = json.loads((run / "protocol.json").read_text())
    if protocol["cache_sha256"] != file_hash(cache_path):
        raise ValueError("Cache changed since training")
    cfg = Config(**protocol["config"])
    rows = load_cache(cache_path, cfg)["dev"]
    runtime, _ = load_native(cfg, device)
    model = Planner(runtime.thinker, cfg, protocol["arm"])
    model.connectors.to(device)
    model.decoders.to(device)
    model.mask_token.data = model.mask_token.data.to(device)
    model.modality_embedding.data = model.modality_embedding.data.to(device)
    checkpoint = torch.load(run / "final.pt", map_location="cpu", weights_only=True)
    if checkpoint["cache_sha256"] != protocol["cache_sha256"]:
        raise ValueError("Checkpoint/cache mismatch")
    restore(model, checkpoint)
    if digest(model.trainable_state()) != digest(checkpoint["state"]):
        raise RuntimeError("Checkpoint restoration changed trainable values")
    model.eval()
    records = []
    for i, row in enumerate(rows):
        layout = Layout(**row["layout"]).to(device)
        other = Layout(**rows[(i + 1) % len(rows)]["layout"]).to(device)
        wrong = replace(layout, prefix=other.prefix, prefix_positions=other.prefix_positions,
                        prefix_valid=other.prefix_valid)
        targets = row["targets"].to(device)
        for seed in cfg.eval_seeds:
            with autocast(device):
                correct_loss = model.loss(layout, targets, seed, all_mask=True)
                wrong_loss = model.loss(wrong, targets, seed, all_mask=True)
            records.append(dict(sample_id=row["sample_id"], seed=seed,
                correct={k: float(v) for k, v in correct_loss.items()},
                shuffled={k: float(v) for k, v in wrong_loss.items()}))
    summary = {}
    for mod in ("video", "audio"):
        correct = np.mean([r["correct"][mod] for r in records])
        wrong = np.mean([r["shuffled"][mod] for r in records])
        per_sample = [np.mean([r["shuffled"][mod] - r["correct"][mod]
                              for r in records if r["sample_id"] == row["sample_id"]]) for row in rows]
        summary[mod] = dict(correct=float(correct), shuffled=float(wrong),
            shuffled_minus_correct=float(wrong - correct), relative_increase_percent=float(100 * (wrong - correct) / correct),
            sample_differences=per_sample)
    result = dict(arm=model.arm, checkpoint_sha256=file_hash(run / "final.pt"),
                  checkpoint_restore_exact=True, summary=summary, records=records,
                  note="Post-pilot diagnostic, no extra training or hyperparameter selection.")
    write_json_new(output, result)
    print(json.dumps(dict(arm=model.arm, summary=summary)), flush=True)
