"""Existing four-node launch and native training; stop at absolute step7000."""
from pathlib import Path
import argparse, os, subprocess, sys, time
import run_runtime as R
from p2_listen.contracts import read, sha, atomic_json

def node(index, resume):
    cluster = read(R.OUT/'cluster.json')
    n = cluster['nodes'][index]
    offset = sum(len(x['gpus']) for x in cluster['nodes'][:index])
    children, streams = [], []
    try:
        for local, gpu in enumerate(n['gpus']):
            rank = offset+local
            env = dict(os.environ, RANK=str(rank), LOCAL_RANK=str(local), WORLD_SIZE='28',
                       LOCAL_WORLD_SIZE=str(len(n['gpus'])), MASTER_ADDR=cluster['nodes'][0]['host'],
                       MASTER_PORT=str(cluster['master_port']), P2_TRAIN_MEMORY_GIB='134')
            stream = (R.OUT/'rank_logs'/f'rank_{rank:02d}.log').open('a')
            streams.append(stream)
            children.append(subprocess.Popen([cluster['python'], '-u', str(Path(__file__)),
                                              'train', '--resume', resume], env=env,
                                             stdout=stream, stderr=subprocess.STDOUT))
        while any(p.poll() is None for p in children):
            if any(p.poll() not in (None, 0) for p in children):
                raise RuntimeError('Full continuation rank failed; inspect rank logs')
            time.sleep(1)
        assert all(p.returncode == 0 for p in children)
    finally:
        for p in children:
            if p.poll() is None: p.terminate()
        for p in children:
            try: p.wait(timeout=20)
            except subprocess.TimeoutExpired: p.kill(); p.wait()
        for stream in streams: stream.close()

def train(resume):
    import torch
    torch.set_num_threads(2)
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    m = R.load()
    c = R.validate(read(R.OUT/'config.json'))
    ready = read(R.W/'READY.json')
    assert ready['source'] == R.source() and ready['config_sha256'] == sha(R.OUT/'config.json')
    rank = int(os.environ['RANK'])
    assert int(os.environ['WORLD_SIZE']) == 28
    atomic_json(R.OUT/'rank_logs'/f'placement_{rank:02d}.json',
                dict(pid=os.getpid(), rank=rank, host=__import__('socket').gethostname(),
                     visible=os.environ['CUDA_VISIBLE_DEVICES'], local=int(os.environ['LOCAL_RANK']), target=7000))
    print(m.train(c, Path(resume), stop_after=7000), flush=True)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['node', 'train'])
    p.add_argument('--index', type=int)
    p.add_argument('--resume', required=True)
    a = p.parse_args()
    if a.action == 'node': node(a.index, a.resume)
    else: train(a.resume)
