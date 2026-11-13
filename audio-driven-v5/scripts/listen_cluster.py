"""Three-node launcher. Owns its workers; shared storage, explicit rank mapping."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / 'configs/p3_listen_3nodes.json'


def transport(cfg, node, command):
    local = {'127.0.0.1', 'localhost', socket.gethostname(),
             *subprocess.check_output(['hostname', '-I'], text=True).split()}
    shell = 'cd ' + shlex.quote(str(ROOT.parent)) + ' && exec ' + shlex.join(list(map(str, command)))
    if node['host'] in local:
        return ['bash', '-c', shell]
    return ['ssh', '-i', node.get('ssh_key',cfg['ssh_key']), '-p', str(node['ssh_port']),
            *(['-o','UserKnownHostsFile='+cfg['known_hosts']] if cfg.get('known_hosts') else []),
            '-o', 'IdentitiesOnly=yes', '-o', 'BatchMode=yes',
            '-o', 'StrictHostKeyChecking=yes', '-o', 'ConnectTimeout=15',
            '-o', 'ServerAliveInterval=20', '-o', 'ServerAliveCountMax=3',
            node['ssh_user'] + '@' + node['host'], shell]


def launch(cfg, commands, out, cancel_event=None):
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    processes, logs, transports = [], [], []
    try:
        for node, cmd in zip(cfg['nodes'], commands):
            # Existing guard kills only its process group if SSH/stdin disappears.
            guard = ROOT.parent / 'audio-driven-v4-p3-autoselect-20261013/scripts/pipeline3_cluster.py'
            command = ['env', 'OPENBLAS_NUM_THREADS=1', 'OMP_NUM_THREADS=2', cfg['python'], str(guard), '--guard', *cmd]
            log = (out / (node['name'] + '.log')).open('x'); logs.append(log)
            transports.append(transport(cfg, node, command))
            processes.append(subprocess.Popen(transports[-1], stdin=subprocess.PIPE,
                                              stdout=log, stderr=subprocess.STDOUT))
        retries = [0] * len(processes)
        while any(p.poll() is None for p in processes):
            if cancel_event is not None and cancel_event.is_set():
                raise RuntimeError('Cluster stage cancelled because a peer stage failed')
            for i,p in enumerate(processes):
                # This cluster intermittently drops SSH during key exchange; no worker started.
                if p.poll()==255 and retries[i]<6:
                    content=(out/(cfg['nodes'][i]['name']+'.log')).read_text()
                    if len(content)<3000 and 'kex_exchange_identification' in content:
                        p.stdin.close();retries[i]+=1
                        time.sleep(2)
                        processes[i]=subprocess.Popen(transports[i],stdin=subprocess.PIPE,stdout=logs[i],stderr=subprocess.STDOUT)
            if any(p.poll() not in (None, 0) for p in processes):
                raise RuntimeError('Cluster worker failed; see ' + str(out))
            time.sleep(2)
        if any(p.returncode for p in processes):
            raise RuntimeError('Cluster failed: ' + str(out))
        (out / 'completed.json').write_text(json.dumps({'returncodes': [p.returncode for p in processes]}))
    finally:
        for p in processes:
            if p.stdin: p.stdin.close()
        for p in processes:
            try: p.wait(timeout=30)
            except subprocess.TimeoutExpired: p.terminate()
        for log in logs: log.close()


def distributed_commands(cfg, script, extra):
    world = sum(len(n['gpus']) for n in cfg['nodes'])
    if world != 24 or len(cfg['nodes']) != 3 or cfg['nodes'][0]['role'] != 'master':
        raise ValueError('Expected master first, three nodes, 24 GPUs')
    return [['env', 'CUDA_VISIBLE_DEVICES=' + ','.join(map(str, n['gpus'])),
        'OMP_NUM_THREADS=2', 'OPENBLAS_NUM_THREADS=1', 'TOKENIZERS_PARALLELISM=false',
        'PYTHONUNBUFFERED=1', *[k+'='+str(v) for k,v in cfg['network'].items()], cfg['python'],
        '-m', 'torch.distributed.run', '--max_restarts=0', '--nnodes=3', '--nproc_per_node=8',
        '--node_rank='+str(i), '--master_addr='+cfg['nodes'][0]['host'],
        '--master_port='+str(cfg['master_port']), str(script), *extra] for i,n in enumerate(cfg['nodes'])]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cluster',type=Path,default=DEFAULT)
    p.add_argument('--logs',type=Path,required=True)
    p.add_argument('script',type=Path)
    p.add_argument('arguments',nargs=argparse.REMAINDER)
    a=p.parse_args();cfg=json.loads(a.cluster.read_text())
    launch(cfg,distributed_commands(cfg,a.script.resolve(),a.arguments),a.logs)


if __name__=='__main__':main()
