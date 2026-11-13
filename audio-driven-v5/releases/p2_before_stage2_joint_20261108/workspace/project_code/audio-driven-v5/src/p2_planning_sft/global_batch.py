"""Keep original global sample batches when changing from 32 to 24 ranks."""
import math
from torch.utils.data import Sampler


class FixedGlobalBatchShard(Sampler):
    def __init__(self,sampler,global_batch_size,world_size,rank):
        if global_batch_size<world_size or not 0<=rank<world_size:raise ValueError('Invalid global batch partition')
        self.sampler=sampler;self.global_batch_size=global_batch_size;self.world_size=world_size;self.rank=rank
        self.batch_size=math.ceil(global_batch_size/world_size);self.drop_last=False
    def __len__(self):return math.ceil(len(self.sampler)/self.global_batch_size)
    def __iter__(self):
        initial=[];batch=[];g=self.global_batch_size
        for index in self.sampler:
            if len(initial)<g:initial.append(index)
            batch.append(index)
            if len(batch)==g:
                yield batch[g*self.rank//self.world_size:g*(self.rank+1)//self.world_size];batch=[]
        if batch:
            # Match Accelerate's original even_batches=True tail padding.
            while len(initial)<g:initial+=initial
            batch+=initial[:g-len(batch)]
            yield batch[g*self.rank//self.world_size:g*(self.rank+1)//self.world_size]


def loader(trainer):
    from functools import partial
    from accelerate.data_loader import DataLoaderShard
    from transformers.trainer_utils import seed_worker
    args=trainer.args
    sampler=trainer._get_train_sampler()
    batches=FixedGlobalBatchShard(sampler,trainer.fixed_global_batch,args.world_size,args.process_index)
    return DataLoaderShard(trainer.train_dataset,device=trainer.accelerator.device,rng_types=[],
        batch_sampler=batches,collate_fn=trainer.data_collator,num_workers=args.dataloader_num_workers,
        pin_memory=args.dataloader_pin_memory,persistent_workers=args.dataloader_persistent_workers,
        prefetch_factor=args.dataloader_prefetch_factor,
        worker_init_fn=partial(seed_worker,num_workers=args.dataloader_num_workers,rank=args.process_index))
