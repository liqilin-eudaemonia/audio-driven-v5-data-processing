"""Partition native length-grouped global batches without adding tail samples."""
import array
import hashlib
import math
from pathlib import Path
from torch.utils.data import Sampler


def sequence_sha(values):
    return hashlib.sha256(array.array('q', values).tobytes()).hexdigest()


class FixedGlobalBatchShard(Sampler):
    def __init__(self, sampler, global_batch_size, world_size, rank, audit_dir=None):
        if global_batch_size < world_size or not 0 <= rank < world_size:
            raise ValueError('Invalid global batch partition')
        self.sampler = sampler
        self.global_batch_size = global_batch_size
        self.world_size = world_size
        self.rank = rank
        self.audit_dir = audit_dir
        self.batch_size = math.ceil(global_batch_size / world_size)
        self.drop_last = False

    def __len__(self):
        return math.ceil(len(self.sampler) / self.global_batch_size)

    def __iter__(self):
        import torch
        from p2_listen.contracts import atomic_json
        before = hashlib.sha256(torch.random.get_rng_state().numpy().tobytes()).hexdigest()
        indices = list(self.sampler)
        if len(indices) < self.world_size or len(set(indices)) != len(indices):
            raise ValueError('Expected each training index exactly once')
        order_sha = sequence_sha(indices)
        if self.audit_dir is not None:
            if torch.distributed.is_initialized():
                hashes = [None] * self.world_size
                torch.distributed.all_gather_object(hashes, order_sha)
                if len(set(hashes)) != 1:
                    raise ValueError('Ranks reconstructed different global training orders')
            atomic_json(Path(self.audit_dir) / f'rank-{self.rank:02d}.json', dict(
                rank=self.rank, world_size=self.world_size, examples=len(indices),
                global_batch=self.global_batch_size, batches=len(self),
                order_sha256=order_sha, sampler_rng_sha256=before,
                checkpoint_prefix_examples=1500 * self.global_batch_size,
                checkpoint_prefix_sha256=sequence_sha(indices[:1500*self.global_batch_size]),
                first_indices=indices[:8], tail_size=len(indices) % self.global_batch_size,
                tail_padding=False))
        for offset in range(0, len(indices), self.global_batch_size):
            batch = indices[offset:offset+self.global_batch_size]
            n = len(batch)
            local = batch[n*self.rank//self.world_size:n*(self.rank+1)//self.world_size]
            if not local:
                raise ValueError('Empty rank at final optimizer step')
            yield local


def loader(trainer):
    from functools import partial
    from accelerate.data_loader import DataLoaderShard
    from transformers.trainer_utils import seed_worker
    args = trainer.args
    batches = FixedGlobalBatchShard(trainer._get_train_sampler(), trainer.fixed_micro_batch,
        args.world_size, args.process_index, Path(args.output_dir)/'sampler_audit')
    return DataLoaderShard(trainer.train_dataset, device=trainer.accelerator.device, rng_types=[],
        batch_sampler=batches, collate_fn=trainer.data_collator,
        num_workers=args.dataloader_num_workers, pin_memory=args.dataloader_pin_memory,
        persistent_workers=args.dataloader_persistent_workers,
        prefetch_factor=args.dataloader_prefetch_factor,
        worker_init_fn=partial(seed_worker, num_workers=args.dataloader_num_workers, rank=args.process_index))
