"""Length-bucket sampler aligned with UniAV ``BucketSampler`` (ti2av_dataset.py).

Design goals (same as UniAV):
  - Each micro-batch is drawn from a single length bucket (same ``n_frames``).
  - With ``sync_bucket_across_ranks=True``, every rank picks the **same** bucket on a
    given step (seed excludes rank) so FSDP ranks do not wait on mismatched seq lens.
  - Every eligible sample is visited exactly once before minimal bucket-tail padding.
  - A global batch is formed from one shuffled bucket and sharded across ranks.
  - Bucket tails are padded to ``world_size * batch_size`` so every rank has the
    same step count without dropping data.

This module is opt-in and independent of the legacy ``DataLoader(shuffle=True)`` path.
"""

from __future__ import annotations

from collections import Counter
from typing import Iterator, Protocol, Sequence

import numpy as np
from torch.utils.data import Sampler

from ltx_trainer import logger


class BucketDataset(Protocol):
    """Minimal dataset API required by ``LengthBucketSampler``."""

    def get_bucket_keys(self) -> Sequence[int]: ...

    def get_bucket_size(self, bucket_key: int) -> int: ...


class LengthBucketSampler(Sampler[tuple[int, int]]):
    """Yield ``(bucket_key, sample_idx)`` tuples for length-bucketed training.

    Intended usage with ``batch_size=1``::

        sampler = LengthBucketSampler(dataset, batch_size=1, ...)
        DataLoader(dataset, sampler=sampler, batch_size=1, ...)
    """

    def __init__(
        self,
        dataset: BucketDataset,
        *,
        batch_size: int = 1,
        num_replicas: int = 1,
        rank: int = 0,
        seed: int = 42,
        sync_bucket_across_ranks: bool = True,
        max_batches_per_epoch: int | None = None,
        min_bucket_size: int | None = None,
    ) -> None:
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        if num_replicas < 1:
            raise ValueError(f"num_replicas must be >= 1, got {num_replicas}")
        if not 0 <= rank < num_replicas:
            raise ValueError(f"rank must be in [0, {num_replicas}), got {rank}")

        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        self.sync_bucket_across_ranks = bool(sync_bucket_across_ranks)
        self.max_batches_per_epoch = max_batches_per_epoch
        self.epoch = 0
        self.start_step_index = 0

        # Small buckets are safe: their tail is padded across the global batch. An
        # explicit min_bucket_size remains available for intentional filtering.
        if min_bucket_size is None:
            min_bucket_size = 1
        self.min_bucket_size = int(min_bucket_size)

        self.bucket_keys = sorted(int(k) for k in dataset.get_bucket_keys())
        self.bucket_sizes: dict[int, int] = {}
        dropped: list[tuple[int, int]] = []
        for bucket_key in self.bucket_keys:
            bucket_size = int(dataset.get_bucket_size(bucket_key))
            if bucket_size >= self.min_bucket_size:
                self.bucket_sizes[bucket_key] = bucket_size
            else:
                dropped.append((bucket_key, bucket_size))

        self.valid_bucket_keys = [k for k in self.bucket_keys if k in self.bucket_sizes]
        if not self.valid_bucket_keys:
            raise ValueError(
                f"No valid length buckets: all buckets have fewer than "
                f"min_bucket_size={self.min_bucket_size} samples "
                f"(batch_size={self.batch_size} * num_replicas={self.num_replicas}). "
                f"Dropped: {dropped}"
            )
        if dropped and rank == 0:
            logger.warning(
                f"LengthBucketSampler: dropped {len(dropped)} undersized bucket(s) "
                f"(need >= {self.min_bucket_size}): {dropped}"
            )

        total_samples = sum(self.bucket_sizes.values())

        self._generate_epoch_indices()
        if rank == 0:
            stats = [(k, self.bucket_sizes[k]) for k in self.valid_bucket_keys]
            logger.info(
                f"LengthBucketSampler: {total_samples:,} samples in {len(self.valid_bucket_keys)} "
                f"buckets, batch_size={self.batch_size}, world={self.num_replicas}, "
                f"sync_bucket={self.sync_bucket_across_ranks}, buckets={stats}"
            )

    def _generate_epoch_indices(self) -> None:
        global_batch_size = self.batch_size * self.num_replicas
        full_bucket_steps = {
            key: (self.bucket_sizes[key] + global_batch_size - 1) // global_batch_size
            for key in self.valid_bucket_keys
        }
        schedule = [key for key in self.valid_bucket_keys for _ in range(full_bucket_steps[key])]
        schedule_seed = self.seed + self.epoch * 1_000_003
        if not self.sync_bucket_across_ranks:
            schedule_seed += self.rank * 10_007
        schedule_rng = np.random.default_rng(schedule_seed)
        schedule_rng.shuffle(schedule)

        full_num_batches = len(schedule)
        if self.max_batches_per_epoch and full_num_batches > self.max_batches_per_epoch:
            if self.rank == 0:
                logger.warning(
                    "LengthBucketSampler: capping epoch batches "
                    f"{full_num_batches} -> {self.max_batches_per_epoch}; "
                    "the capped epoch cannot cover every sample"
                )
            schedule = schedule[: int(self.max_batches_per_epoch)]

        self.epoch_bucket_schedule = tuple(schedule)
        self.num_batches_per_epoch = len(schedule)
        self.num_batches_for_this_rank = len(schedule)
        self.batch_start_idx = 0
        self.batch_end_idx = len(schedule)
        scheduled_bucket_steps = Counter(schedule)

        # Build one deterministic global permutation per bucket. Every rank sees the
        # same tensor and takes its own rank slice, so non-padding samples cannot
        # collide across ranks or steps.
        self._rank_batches: dict[int, np.ndarray] = {}
        for key, num_steps in scheduled_bucket_steps.items():
            bucket_size = self.bucket_sizes[key]
            permutation_rng = np.random.default_rng(self.seed + self.epoch * 1_000_003 + key * 10_009)
            permutation = permutation_rng.permutation(bucket_size)
            padded_size = num_steps * global_batch_size
            if padded_size <= bucket_size:
                padded = permutation[:padded_size]
            else:
                repeats = (padded_size + bucket_size - 1) // bucket_size
                padded = np.tile(permutation, repeats)[:padded_size]
            self._rank_batches[key] = padded.reshape(num_steps, self.num_replicas, self.batch_size)

        if self.rank == 0:
            total_samples = sum(self.bucket_sizes.values())
            padded_samples = len(schedule) * global_batch_size
            logger.info(
                f"LengthBucketSampler epoch={self.epoch}: global_steps={len(schedule)}, "
                f"batches_per_rank={self.num_batches_for_this_rank}, samples={total_samples}, "
                f"tail_padding={max(0, padded_samples - total_samples)}"
            )

    def __iter__(self) -> Iterator[tuple[int, int]]:
        bucket_offsets = Counter(self.epoch_bucket_schedule[: self.start_step_index])
        for bucket_key in self.epoch_bucket_schedule[self.start_step_index :]:
            bucket_step = bucket_offsets[bucket_key]
            selected = self._rank_batches[bucket_key][bucket_step, self.rank]
            bucket_offsets[bucket_key] += 1
            for sample_idx in selected:
                yield bucket_key, int(sample_idx)

    def __len__(self) -> int:
        remaining_batches = max(0, self.num_batches_for_this_rank - self.start_step_index)
        return remaining_batches * self.batch_size

    def set_epoch(self, epoch: int, start_step_index: int = 0) -> None:
        """Reset RNG schedule for a new epoch; optional resume offset in batch units."""
        self.epoch = int(epoch)
        self.start_step_index = int(start_step_index)
        if self.start_step_index < 0:
            raise ValueError(f"start_step_index must be >= 0, got {self.start_step_index}")
        if self.start_step_index > self.num_batches_for_this_rank:
            # Clamp after regenerating sizes for the new epoch.
            pass
        self._generate_epoch_indices()
        if self.start_step_index > self.num_batches_for_this_rank:
            logger.warning(
                f"LengthBucketSampler: start_step_index={self.start_step_index} exceeds "
                f"batches_per_rank={self.num_batches_for_this_rank}; clamping to 0"
            )
            self.start_step_index = 0
