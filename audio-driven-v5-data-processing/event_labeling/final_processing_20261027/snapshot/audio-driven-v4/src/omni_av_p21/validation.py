"""Fixed-noise Dev losses include the active text-attention adapters."""
import random
import numpy as np
import torch

from omni_av.ltx_runtime import with_text
from omni_av_p2.validation import distributed_mean


@torch.no_grad()
def evaluate(raw, renderer, strategy, sampler, samples, cfg, recipe, rank, world, device, seed):
    py_rng, np_rng = random.getstate(), np.random.get_state()
    mode = raw.training
    sums, count = [0.] * 6, 0
    try:
        raw.eval()
        with torch.random.fork_rng(devices=[device.index] if device.type == "cuda" else []):
            for ordinal in range(rank, len(samples.entries), world):
                random.seed(seed + ordinal)
                np.random.seed((seed + ordinal) % 2**32)
                torch.manual_seed(seed + ordinal)
                condition, targets, text, _, av = samples[ordinal]
                generator = torch.Generator(device=device).manual_seed(seed + ordinal + 100000)
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    prepared = strategy.prepare_training_inputs(with_text(av, text), sampler)
                    values = raw(condition, targets if recipe == "joint" else None, prepared, generator)
                if not all(torch.isfinite(v) for v in values):
                    raise FloatingPointError("Nonfinite P2.1 Dev loss")
                sums = [s + float(v) for s, v in zip(sums, values)]
                count += 1
    finally:
        raw.train(mode)
        random.setstate(py_rng)
        np.random.set_state(np_rng)
    means, count = distributed_mean(sums, count, device)
    return dict(zip(("loss", "av", "video_feature", "audio_feature", "av_video", "av_audio"), means),
                samples=count, noise_seed=seed, protocol="p21_paired_gt_text_dev_loss")
