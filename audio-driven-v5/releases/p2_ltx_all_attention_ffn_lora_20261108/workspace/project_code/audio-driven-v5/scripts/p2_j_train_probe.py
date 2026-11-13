"""CPU-only 32-rank connectivity probe; safe while cache GPUs are occupied."""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import socket
import torch
import torch.distributed as dist

parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args()
torch.set_num_threads(1)
dist.init_process_group('gloo',timeout=timedelta(seconds=180))
try:
    rank=dist.get_rank();world=dist.get_world_size();assert world==32
    value=torch.tensor(rank+1,dtype=torch.int64);dist.all_reduce(value)
    assert value.item()==world*(world+1)//2
    rows=[None]*world;dist.all_gather_object(rows,dict(rank=rank,host=socket.gethostname(),local_rank=int(os.environ['LOCAL_RANK'])))
    if rank==0:
        assert len({r['host'] for r in rows})==4
        output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
        (output/'RESULT.json').write_text(json.dumps(dict(success=True,backend='gloo',gpu_tested=False,world=world,ranks=rows),indent=2))
finally:dist.destroy_process_group()
