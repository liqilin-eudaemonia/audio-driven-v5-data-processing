# Copyright (c) 2026 Omni-LTX Team
"""Context-Parallel Distributed Sampler for coordinated sequence parallel training."""

import math
from collections.abc import Iterator, Sized
import torch
from torch.utils.data import Sampler


class ContextParallelDistributedSampler(Sampler[int]):
    """Distributed sampler that shards data across data-parallel replicas while ensuring
    all sequence-parallel (CP) ranks within the same group receive identical sample indices.

    Args:
        dataset: Dataset to sample from.
        num_replicas: Total number of processes (world_size).
        rank: Global rank of current process.
        sp_size: Sequence parallel / Context parallel degree.
        shuffle: Whether to shuffle dataset each epoch.
        seed: Random seed for deterministic shuffling.
        drop_last: Whether to drop trailing samples not divisible by dp_world_size.
    """

    def __init__(
        self,
        dataset: Sized,
        num_replicas: int,
        rank: int,
        sp_size: int = 1,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.sp_size = max(1, sp_size)

        if num_replicas % self.sp_size != 0:
            raise ValueError(
                f"num_replicas ({num_replicas}) must be divisible by sp_size ({self.sp_size})"
            )

        self.dp_world_size = num_replicas // self.sp_size
        self.dp_rank = rank // self.sp_size
        self.sp_rank = rank % self.sp_size

        self.epoch = 0
        self.start_index = 0
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed

        # Number of samples per data-parallel replica
        if self.drop_last and len(self.dataset) % self.dp_world_size != 0:
            self.num_samples = math.ceil(
                (len(self.dataset) - self.dp_world_size) / self.dp_world_size
            )
        else:
            self.num_samples = math.ceil(len(self.dataset) / self.dp_world_size)
        self.total_size = self.num_samples * self.dp_world_size

    def __iter__(self) -> Iterator[int]:
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        if not self.drop_last:
            # Add padding indices to make evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # Subsample to exact multiple
            indices = indices[: self.total_size]
        assert len(indices) == self.total_size

        # Subsample for this data-parallel rank
        indices = indices[self.dp_rank : self.total_size : self.dp_world_size]
        assert len(indices) == self.num_samples

        return iter(indices[self.start_index :])

    def __len__(self) -> int:
        return self.num_samples - self.start_index

    def set_epoch(self, epoch: int, start_index: int = 0) -> None:
        if not 0 <= start_index <= self.num_samples:
            raise ValueError(
                f"start_index must be in [0, {self.num_samples}], got {start_index}"
            )
        self.epoch = epoch
        self.start_index = start_index
