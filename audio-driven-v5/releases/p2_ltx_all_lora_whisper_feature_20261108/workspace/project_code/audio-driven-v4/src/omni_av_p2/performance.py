"""Execution options that do not change the model or the training objective."""
from collections import OrderedDict

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from omni_av.ltx_runtime import load_av_targets
from .data import read_sample


class CheckpointBlock(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, *args, **kwargs):
        if self.training and torch.is_grad_enabled():
            return checkpoint(self.block, *args, use_reentrant=False, **kwargs)
        return self.block(*args, **kwargs)


def configure_checkpointing(renderer, extractor, every=1, decoder=True):
    """1: all blocks; 2: alternate blocks; 0: no recomputation. Set once."""
    if every < 0:
        raise ValueError("checkpoint_every must be nonnegative")
    renderer.set_gradient_checkpointing(every == 1)
    if every > 1:
        for i, block in enumerate(renderer.transformer_blocks):
            if i % every == 0:
                renderer.transformer_blocks[i] = CheckpointBlock(block)
    for name in ("video_decoder", "audio_decoder"):
        if hasattr(extractor, name):
            getattr(extractor, name).net.grad_checkpointing = decoder


class TrainingSamples:
    """Bounded device cache; static input files are verified when loaded.

    Each cached sample is immutable for the life of one training process. The
    training strategy creates fresh noisy inputs, never mutating these targets.
    """
    def __init__(self, entries, cfg, root, device, tiny=False, allow_trim=False, capacity=2, split="train"):
        self.entries, self.cfg, self.root, self.device = entries, cfg, root, device
        self.tiny, self.allow_trim, self.capacity = tiny, allow_trim, capacity
        self.split = split
        self.cache = OrderedDict()

    def __getitem__(self, index):
        index %= len(self.entries)
        if index in self.cache:
            self.cache.move_to_end(index)
            return self.cache[index]
        condition, targets, text, payload = read_sample(self.entries[index], self.cfg, self.device, self.split)
        if abs(payload["window"]["terminal_trim_s"]) > 1e-4 and not self.allow_trim:
            raise ValueError("Inherited VAE trim requires --allow-terminal-trim")
        av, meta = load_av_targets(self.root, payload["sample_id"], device=self.device, tiny=self.tiny, split=self.split)
        example = payload["example"]
        if not self.tiny:
            height, width = av["latents"]["latents"].shape[-2:]
            if (height * 32, width * 32) != (example.get("target_height", meta["metadata"]["target_height"]),
                                           example.get("target_width", meta["metadata"]["target_width"])):
                raise ValueError("Spatial shape changed: no automatic resize or square crop allowed")
        duration = meta["metadata"]["target_frames"] / meta["metadata"]["fps"]
        if abs(duration - payload["window"]["renderer_target_duration_s"]) > 1e-4:
            raise ValueError("Renderer cache window changed after feature extraction")
        result = condition, targets, text, payload, av
        if self.capacity:
            self.cache[index] = result
            while len(self.cache) > self.capacity:
                self.cache.popitem(last=False)
        return result


def global_sample_index(step, micro, rank, world_size, accumulation):
    """Identical data/noise assignment for DDP and serial accumulation."""
    return (step - 1) * world_size * accumulation + micro * world_size + rank


def sample_costs(entries, root):
    """Self-attention work scales approximately with squared video token count."""
    import json
    from pathlib import Path
    path = Path(root) / 'ltx23_precomputed_720p_area_qualified_20260920/metadata_train.jsonl'
    metadata = {row['sample_id']: row for row in map(json.loads, path.open())}
    costs = []
    for entry in entries:
        row = metadata[entry['sample_id']]
        tokens = ((row['target_frames'] - 1) // 8 + 1) * (row['target_height'] // 32) * (row['target_width'] // 32)
        costs.append(tokens * tokens)
    return costs


def balanced_sample_indices(step, world, accumulation, costs):
    """Redistribute the SAME global sample ordinals; preserve their noise seeds.

    Greedy longest-job-first assignment with a fixed microbatch quota per rank.
    Only assignment and accumulation order change, not sample weights or membership.
    """
    import heapq
    if min(step, world, accumulation) < 1 or not costs or min(costs) <= 0:
        raise ValueError('Invalid workload schedule')
    base = (step - 1) * world * accumulation
    jobs = sorted(range(base, base + world * accumulation), key=lambda x: (-costs[x % len(costs)], x))
    queue = [(0, rank) for rank in range(world)]
    heapq.heapify(queue)
    result = [[] for _ in range(world)]
    for ordinal in jobs:
        cost, rank = heapq.heappop(queue)
        result[rank].append(ordinal)
        if len(result[rank]) < accumulation:
            heapq.heappush(queue, (cost + costs[ordinal % len(costs)], rank))
    return result
