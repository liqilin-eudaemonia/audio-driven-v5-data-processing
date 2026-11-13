#!/usr/bin/env python3
"""Durable three-node launcher for the frozen, authorized E1 run."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
PYTHON = ROOT.parent / 'Omni-LTX-stage2/.venv/bin/python'


def write(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(tmp, path)


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--node-rank', type=int, choices=range(3), required=True)
    p.add_argument('--detach', action='store_true')
    p.add_argument('--resume', type=Path)
    a = p.parse_args()
    ready = json.loads((OUT / 'READY.json').read_text())
    assert hashlib.sha256((OUT / 'config.json').read_bytes()).hexdigest() == ready['config_sha256']
    release = json.loads((OUT / 'training_release.json').read_text())
    assert hashlib.sha256((OUT / 'training_release.json').read_bytes()).hexdigest() == ready['release_sha256']
    for path, expected in release['source_identity'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, path
    status_path = OUT / f'node{a.node_rank}.json'
    if a.detach:
        if status_path.exists() and json.loads(status_path.read_text()).get('state') == 'running':
            raise RuntimeError('Node already marked running; check process before relaunch')
        command = [str(PYTHON), str(Path(__file__).resolve()), '--node-rank', str(a.node_rank)]
        if a.resume:
            command += ['--resume', str(a.resume.resolve())]
        with (OUT / f'node{a.node_rank}.log').open('a') as log:
            child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
        print(json.dumps(dict(node_rank=a.node_rank, supervisor_pid=child.pid, hostname=socket.gethostname())))
        return
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'], text=True)
    memory = [int(line.split(',')[1]) for line in gpu.strip().splitlines()]
    assert len(memory) == 8 and max(memory) < 1024, f'Expected eight idle GPUs, got {gpu}'
    command = [str(PYTHON), '-u', '-m', 'torch.distributed.run', '--nnodes=3', '--nproc-per-node=8',
               f'--node-rank={a.node_rank}', '--master-addr=172.29.252.11', '--master-port=29746',
               '--max-restarts=0', '--log-dir', str(OUT / 'worker_logs' / f'node{a.node_rank}'), '--tee=3',
               'scripts/p2_listen.py', 'train', '--config', str(OUT / 'config.json'),
               '--arm', 'E1', '--train-seed', '20261023']
    if a.resume:
        command += ['--resume', str(a.resume.resolve())]
    script = 'source ' + shlex.quote(str(ROOT / 'configs/p2_listen_h200_multinode.env')) + '\nexec ' + shlex.join(command)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7', PYTHONUNBUFFERED='1')
    record = dict(state='running', node_rank=a.node_rank, hostname=socket.gethostname(),
                  supervisor_pid=os.getpid(), started_at=now(), command=command,
                  communication_config=str(ROOT / 'configs/p2_listen_h200_multinode.env'),
                  config_sha256=ready['config_sha256'], source_verified=True,
                  resume=str(a.resume) if a.resume else None)
    child = subprocess.Popen(['/bin/bash', '-c', script], cwd=ROOT, env=env, stdin=subprocess.DEVNULL)
    record['torchrun_pid'] = child.pid
    write(status_path, record)
    print(json.dumps(record), flush=True)
    rc = child.wait()
    record.update(state='finished' if rc == 0 else 'failed', returncode=rc, finished_at=now())
    write(status_path, record)
    print(json.dumps(record), flush=True)
    sys.exit(rc)


if __name__ == '__main__':
    main()
