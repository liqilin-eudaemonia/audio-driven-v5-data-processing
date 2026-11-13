"""Verify every global batch under native Accelerate32 and new24 partitions."""
import sys,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import torch,numpy as np
from torch.utils.data import BatchSampler
from transformers.trainer_pt_utils import LengthGroupedSampler
from accelerate.data_loader import BatchSamplerShard
from p2_planning_sft.reference_data import ReferencePlanningDataset
from p2_planning_sft.global_batch import FixedGlobalBatchShard
from p2_listen.contracts import read,sha,atomic_json,digest
c=read(ROOT/'configs/p2_planning_sft_filtered_3b_32gpu_20261102.json');d=ReferencePlanningDataset(c['data']['seal'],'train');torch.set_num_threads(2)
torch.manual_seed(c['seed']);torch.empty((),dtype=torch.int64).random_() # DataLoader's original worker base-seed draw.
order=list(LengthGroupedSampler(batch_size=8,lengths=d.lengths))
old=[]
for rank in range(32):old.append(np.asarray(list(BatchSamplerShard(BatchSampler(order,8,False),num_processes=32,process_index=rank)),dtype=np.int32))
old=np.concatenate(old,axis=1)
new=[]
for rank in range(24):new.append(np.asarray(list(FixedGlobalBatchShard(order,256,24,rank)),dtype=np.int32))
new=np.concatenate(new,axis=1)
assert np.array_equal(old,new) and old.shape==(2697,256)
assert len(set(old[:500].reshape(-1).tolist()))==128000
assert set(old[:500].reshape(-1).tolist()).isdisjoint(set(old[500:-1].reshape(-1).tolist()))
report=dict(success=True,rows=len(d),global_batch=256,steps=len(old),completed_step=500,completed_unique_samples=128000,
    every_global_batch_equal=True,tail_padding_equal=True,code_sha256=sha(ROOT/'src/p2_planning_sft/global_batch.py'),
    plan_sha256=__import__('hashlib').sha256(old.tobytes()).hexdigest(),local_batches=[len(x) for x in next(iter(zip(*[iter(FixedGlobalBatchShard(order,256,24,r)) for r in range(24)])))])
atomic_json(ROOT/'experiments/p2_planning_sft_filtered_3b_32gpu_20261102/performance/batch_partition24.json',report);print(json.dumps(report),flush=True)
