"""Source-only layouts and support-aware reductions in seconds."""
import math
import torch


def geometry(obs):
    g = obs["geometry"]
    real = obs["requested_duration_s"]
    frames = 8 * math.ceil((math.ceil(real * g["fps"] - 1e-7) - 1) / 8) + 1
    return dict(g, frames=frames, duration_s=frames / g["fps"], real_duration_s=real,
                padding_s=frames / g["fps"] - real)


def make_layout(duration, grid_hw, vision_fps=8, audio_hz=8):
    if duration <= 0 or min(grid_hw) < 1 or vision_fps != 8 or audio_hz != 8:
        raise ValueError("Invalid layout")
    result = {}
    for mod, hz in (("video", vision_fps / 2), ("audio", audio_hz)):
        count = math.ceil(duration * hz - 1e-7)
        intervals = torch.tensor([[i / hz, (i + 1) / hz] for i in range(count)], dtype=torch.float32)
        # Visual centers match paired frames at i*2/fps and (i*2+1)/fps.
        offset = .5 / vision_fps if mod == "video" else .5 / audio_hz
        h, w = grid_hw if mod == "video" else (1, 1)
        coords = [[i / hz + offset, y, x] for i in range(count) for y in range(h) for x in range(w)]
        result[mod + "_coordinates"] = torch.tensor([coords], dtype=torch.float32)
        result[mod + "_groups"] = torch.arange(count).repeat_interleave(h * w)[None]
        result[mod + "_intervals"] = intervals[None]
        result[mod + "_valid"] = torch.ones(1, len(coords), dtype=torch.bool)
        result[mod + "_target_valid"] = (intervals[:, 1] <= duration + 1e-6).repeat_interleave(h * w)[None]
    return result


def pool_by_time(values, groups, valid, count):
    if values.shape[:2] != groups.shape or groups.shape != valid.shape:
        raise ValueError("Time group shape mismatch")
    out = values.new_zeros(values.shape[0], count, values.shape[-1])
    counts = values.new_zeros(values.shape[0], count, 1)
    out.scatter_add_(1, groups[..., None].expand_as(values), values * valid[..., None])
    counts.scatter_add_(1, groups[..., None], valid[..., None].to(values.dtype))
    return out / counts.clamp_min(1), counts[..., 0] > 0


def aggregate_audio(features, intervals, native_tick=.04):
    """Overlap-weighted native tokens; no adaptive pooling of padded sequences."""
    n = features.shape[1]
    a = torch.arange(n, device=features.device).float() * native_tick
    b = a + native_tick
    bins = intervals.to(features.device)[0]
    weights = (torch.minimum(bins[:, 1, None], b) - torch.maximum(bins[:, 0, None], a)).clamp_min(0)
    valid = weights.sum(-1) >= (bins[:, 1] - bins[:, 0]) - 1e-6
    z = torch.einsum("tn,bnd->btd", weights / weights.sum(-1, keepdim=True).clamp_min(1e-8), features.float())
    return z.to(features.dtype), valid[None]
