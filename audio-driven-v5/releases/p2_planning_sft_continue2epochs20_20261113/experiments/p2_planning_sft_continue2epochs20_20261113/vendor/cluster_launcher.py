"""Render or explicitly launch three/four-node DDP; never stop unrelated GPU jobs."""
import argparse
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time

ROOT = Path('/private/liqi_workspace/project_code/audio-driven-v5')
sys.path.insert(0, str(ROOT/'src'))
from p2_listen.contracts import read, atomic_json


def ssh(node, cluster):
    return ['ssh', '-i', node.get('ssh_key', cluster['ssh_key']), '-o', 'IdentitiesOnly=yes',
            '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-o',
            'UserKnownHostsFile='+cluster['known_hosts'], '-p', str(node['ssh_port']),
            node['ssh_user']+'@'+node['host']]


def command(cluster, config, rank, resume=None, probe=False):
    node = cluster['nodes'][rank]
    env = dict(cluster['network'], CUDA_VISIBLE_DEVICES=','.join(map(str,node['gpus'])),
               TOKENIZERS_PARALLELISM='false', PYTHONUNBUFFERED='1', HF_HUB_OFFLINE='1')
    cmd = ['env'] + [k+'='+str(v) for k,v in env.items()] + [cluster['python'], '-m',
        'torch.distributed.run', '--nnodes='+str(len(cluster['nodes'])), '--nproc_per_node=8', '--node_rank='+str(rank),
        '--master_addr='+cluster['nodes'][0]['host'], '--master_port='+str(cluster['master_port'])]
    if probe:
        cmd += [str(ROOT/'scripts/p2_j_train_probe.py'), '--output', cluster['probe_output']]
    else:
        cmd += ['/private/liqi_workspace/project_code/audio-driven-v5/experiments/p2_planning_sft_upright_3b_32gpu_resume_20261106/vendor/train_entry.py', '--config', str(Path(config).resolve())]
        if cluster.get('stop_after') is not None:
            cmd += ['--stop-after', str(cluster['stop_after'])]
        if resume:
            cmd += ['--resume', str(Path(resume).resolve())]
    return shlex.join(cmd)


def node_run(cmd, pid_file):
    child = subprocess.Popen(shlex.split(cmd), cwd=ROOT, start_new_session=True)
    atomic_json(pid_file, dict(pid=child.pid, supervisor=os.getpid(), command=cmd))
    def stop(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGHUP, stop)
    try:
        return child.wait()
    finally:
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default=str(ROOT/'configs/p2_planning_sft_filtered_3b_24gpu_inputfast_isolated_20261102.json'))
    p.add_argument('--cluster', default=str(ROOT/'configs/p2_planning_cluster_filtered_24gpu_20261102.json'))
    p.add_argument('--resume')
    p.add_argument('--launch', action='store_true', help='Without this flag, only render commands')
    p.add_argument('--probe', action='store_true', help='32-rank CPU-only connectivity test')
    p.add_argument('--node-command', help=argparse.SUPPRESS)
    p.add_argument('--pid-file', help=argparse.SUPPRESS)
    a = p.parse_args()
    if a.node_command:
        sys.exit(node_run(a.node_command, a.pid_file))
    cluster, config = read(a.cluster), read(a.config)
    if len(cluster['nodes']) not in (3,4) or any(len(n['gpus']) != 8 for n in cluster['nodes']):
        raise ValueError('Expected three or four nodes with eight GPUs each')
    commands = [command(cluster, a.config, i, a.resume, a.probe) for i in range(len(cluster['nodes']))]
    if not a.launch:
        print(json.dumps(dict(launch=False, gpu_training=not a.probe, nodes=[dict(name=n['name'],command=c)
              for n,c in zip(cluster['nodes'],commands)]), indent=2))
        return
    root = Path(cluster['probe_output'] if a.probe else config['output'])
    # Separate each invocation so cleanup cannot read a previous launch's PID.
    logs = root/'launch_logs'/(str(time.time_ns())+'-'+str(os.getpid()))
    if not a.probe:
        for node in cluster['nodes']:
            out = subprocess.check_output(ssh(node,cluster)+['nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits'], text=True)
            free = {int(row.split(',')[0]):float(row.split(',')[1])/1024 for row in out.splitlines()}
            if any(free[g] < cluster['minimum_free_gib'] for g in node['gpus']):
                raise RuntimeError(node['name']+' GPUs are occupied; no jobs were stopped or launched')
    logs.mkdir(parents=True, exist_ok=True)
    children = []
    try:
        for node, cmd in zip(cluster['nodes'],commands):
            pid = logs/(node['name']+'.pid.json')
            wrapper = shlex.join([cluster['python'], str(Path(__file__).resolve()),
                                 '--node-command', cmd, '--pid-file', str(pid)])
            with (logs/(node['name']+'.log')).open('a') as stream:
                children.append(subprocess.Popen(ssh(node,cluster)+[wrapper], stdout=stream, stderr=subprocess.STDOUT))
        while True:
            states = [p.poll() for p in children]
            if any(s not in (None,0) for s in states):
                raise RuntimeError('A node exited unsuccessfully; see '+str(logs))
            if all(s == 0 for s in states):
                break
            time.sleep(2)
    finally:
        for node, child in zip(cluster['nodes'],children):
            if child.poll() is None:
                pid_path = logs/(node['name']+'.pid.json')
                if pid_path.exists():
                    pid = int(read(pid_path)['supervisor'])
                    subprocess.run(ssh(node,cluster)+['kill -TERM '+str(pid)], check=False)
                child.terminate()


if __name__ == '__main__':
    main()
