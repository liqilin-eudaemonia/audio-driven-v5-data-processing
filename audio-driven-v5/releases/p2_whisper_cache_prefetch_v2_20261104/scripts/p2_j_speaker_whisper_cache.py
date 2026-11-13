"""Cache only paired SPEAKER audio with frozen Whisper; never launch training."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

from p2_listen.contracts import read
from p2_whisper_av.config import DEFAULT, load

CLUSTER = ROOT/'configs/p2_j_qwen_whisper_av_32gpu_20261104.json'


def cache_commands(cluster, config, *, batch_size=16, read_workers=4, cpu_threads=2, master_port=29844):
    """Print one torchrun command per node; ranks shard the same frozen list."""
    nodes = cluster['nodes']
    if not nodes or any(not node['gpus'] for node in nodes):
        raise ValueError('Every selected cache node must have explicit GPUs')
    sizes = {len(node['gpus']) for node in nodes}
    if len(sizes) != 1 or any(len(set(node['gpus'])) != len(node['gpus']) for node in nodes):
        raise ValueError('Cache torchrun requires equally sized, unique local GPU lists')
    if (any(type(v) is not int or v < 1 for v in (batch_size, read_workers, cpu_threads))
            or type(master_port) is not int or not 1 <= master_port <= 65535):
        raise ValueError('Invalid cache batch / CPU / rendezvous configuration')
    return [[
        'env', 'CUDA_VISIBLE_DEVICES='+','.join(map(str,node['gpus'])),
        'TOKENIZERS_PARALLELISM=false', 'PYTHONUNBUFFERED=1',
        'OMP_NUM_THREADS='+str(cpu_threads), 'OPENBLAS_NUM_THREADS=1',
        cluster['python'], '-m', 'torch.distributed.run', '--max_restarts=0',
        '--nnodes='+str(len(nodes)), '--nproc_per_node='+str(len(node['gpus'])),
        '--node_rank='+str(rank), '--master_addr='+nodes[0]['host'], '--master_port='+str(master_port),
        str(Path(__file__).resolve()), 'extract', '--config', str(Path(config).resolve()),
        '--batch-size', str(batch_size), '--read-workers', str(read_workers),
        '--cpu-threads', str(cpu_threads),
    ] for rank,node in enumerate(nodes)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('prepare', 'extract', 'finalize', 'commands'))
    p.add_argument('--config', type=Path, default=DEFAULT)
    p.add_argument('--cluster', type=Path, default=CLUSTER)
    p.add_argument('--sample-ids', type=Path, help='JSON array; use a separate pilot cache directory')
    p.add_argument('--rank', type=int, default=int(os.environ.get('RANK',0)))
    p.add_argument('--world-size', type=int, default=int(os.environ.get('WORLD_SIZE',1)))
    p.add_argument('--device')
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--read-workers', type=int, default=4)
    p.add_argument('--cpu-threads', type=int, default=2)
    p.add_argument('--master-port', type=int, default=29844)
    p.add_argument('--limit', type=int)
    a = p.parse_args(); c = load(a.config)
    if a.command == 'commands':
        result = dict(audio_source='speaker', training_started=False,
            nodes=[node['name'] for node in read(a.cluster)['nodes']],
            commands=cache_commands(read(a.cluster),a.config,batch_size=a.batch_size,
                read_workers=a.read_workers,cpu_threads=a.cpu_threads,master_port=a.master_port))
    else:
        import torch
        if a.cpu_threads < 1:
            p.error('--cpu-threads must be positive')
        torch.set_num_threads(a.cpu_threads)
        from p2_whisper_av import cache
        if a.command == 'prepare':
            result = cache.prepare(c,sample_ids=None if a.sample_ids is None else read(a.sample_ids))
        elif a.command == 'finalize':
            result = cache.finalize(c,read_workers=a.read_workers)
        else:
            device = a.device or ('cuda:'+os.environ.get('LOCAL_RANK','0') if torch.cuda.is_available() else 'cpu')
            if device.startswith('cuda'):
                torch.cuda.set_device(torch.device(device))
            result = cache.extract(c,rank=a.rank,world=a.world_size,device=device,
                batch_size=a.batch_size,read_workers=a.read_workers,limit=a.limit)
        result = dict(result,audio_source='speaker',training_started=False)
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__ == '__main__':
    main()
