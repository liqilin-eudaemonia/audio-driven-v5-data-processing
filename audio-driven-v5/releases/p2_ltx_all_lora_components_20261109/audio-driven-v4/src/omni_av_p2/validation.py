"""Fixed-noise held-out loss evaluation; no padding or repeated Dev examples."""
import random
import numpy as np
import torch
import torch.distributed as dist
from omni_av.ltx_runtime import with_text
from .ltx import append_conditions, av_loss


def check_indexes(train, dev, expected_contract):
    """Old engineering Train indexes remain usable without a Dev index."""
    for index, split in ((train, "train"), (dev, "dev")):
        if index is None:
            continue
        if index["contract"] != expected_contract or not index["entries"]:
            raise ValueError("Cache contract mismatch or empty dataset")
        if index.get("split", "train") != split:
            raise ValueError(f"Expected {split} cache")
        if dev is not None:
            entries = index["entries"]
            if any(e.get("split") != split or not e.get("sample_id") for e in entries):
                raise ValueError("Validation requires indexes with split and sample_id; reindex legacy caches")
            ids = [e["sample_id"] for e in entries]
            if len(set(ids)) != len(ids):
                raise ValueError("Duplicate cache IDs")
    if dev and {e["sample_id"] for e in train["entries"]} & {e["sample_id"] for e in dev["entries"]}:
        raise ValueError("Train/Dev overlap")


def distributed_mean(sums, count, device):
    value = torch.tensor([*sums, count], dtype=torch.float64, device=device)
    dist.all_reduce(value)
    if value[-1] <= 0:
        raise ValueError("Empty validation set")
    return (value[:-1] / value[-1]).tolist(), int(value[-1])


@torch.no_grad()
def evaluate(raw, renderer, strategy, sampler, samples, cfg, recipe, rank, world, device, seed):
    # Dev noise is independent of checkpoint step and never advances training RNG.
    py_rng, np_rng = random.getstate(), np.random.get_state()
    modes = raw.training, renderer.training
    sums, count = [0.] * 6, 0
    try:
        raw.eval()
        renderer.eval()
        with torch.random.fork_rng(devices=[device.index] if device.type == "cuda" else []):
            for ordinal in range(rank, len(samples.entries), world):
                random.seed(seed + ordinal)
                np.random.seed((seed + ordinal) % 2**32)
                torch.random.default_generator.manual_seed(seed + ordinal)
                if device.type == "cuda":
                    torch.cuda.default_generators[device.index].manual_seed(seed + ordinal)
                condition, targets, text, _, av = samples[ordinal]
                generator = torch.Generator(device=device).manual_seed(seed + ordinal + 100000)
                with torch.autocast(device.type, dtype=torch.bfloat16, cache_enabled=False, enabled=device.type == "cuda"):
                    if recipe == "joint":
                        u, losses = raw(condition, targets, generator)
                    else:
                        u, losses = raw(condition), {}
                    inputs = append_conditions(strategy.prepare_training_inputs(with_text(av, text), sampler), u)
                    lav, parts = av_loss(renderer, strategy, inputs)
                    lv, la = losses.get("video", lav.new_zeros(())), losses.get("audio", lav.new_zeros(()))
                    total = cfg.av_weight * lav + cfg.feature_video_weight * lv + cfg.feature_audio_weight * la
                values = [total, lav, lv, la, parts["video_loss"], parts["audio_loss"]]
                if not all(torch.isfinite(v) for v in values):
                    raise FloatingPointError("Nonfinite Dev loss")
                sums = [s + float(v) for s, v in zip(sums, values)]
                count += 1
    finally:
        raw.train(modes[0])
        renderer.train(modes[1])
        random.setstate(py_rng)
        np.random.set_state(np_rng)
    means, count = distributed_mean(sums, count, device)
    return dict(zip(("loss", "av", "video_feature", "audio_feature", "av_video", "av_audio"), means),
                samples=count, noise_seed=seed, protocol="paired_gt_text_dev_loss")
