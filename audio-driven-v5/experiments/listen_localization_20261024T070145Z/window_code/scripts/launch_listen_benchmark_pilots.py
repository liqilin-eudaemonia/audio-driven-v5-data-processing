#!/usr/bin/env python3
"""Start registered one-sample GPU pilots and record resource usage on one node."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    p = argparse.ArgumentParser(); p.add_argument('--plan', required=True); p.add_argument('--node', required=True)
    a = p.parse_args(); plan = json.loads(Path(a.plan).read_text()); root = Path(plan['root'])
    cfg = json.loads(Path(plan['config']).read_text()); systems = {s['name']: s for s in cfg['systems']}
    folder = root / 'pilot' / a.node; folder.mkdir(parents=True, exist_ok=True)
    processes = []
    for gpu, name in enumerate(plan['nodes'][a.node][1]):
        s = systems[name]; work = folder / f'gpu{gpu}'; work.mkdir(exist_ok=True)
        command = [s['python'], str(Path(plan['code'])/'scripts/benchmark_listen_labeling.py'),
                   '--config', plan['config'], 'worker', '--manifest', plan['manifest'],
                   '--output', str(work), '--limit', '1', '--system', name, '--device', 'cuda:0']
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
                   HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
                   PYTHONUNBUFFERED='1', HF_ENABLE_PARALLEL_LOADING='true', HF_PARALLEL_LOADING_WORKERS='4')
        with (work/'worker.log').open('w') as log:
            proc = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        processes.append((proc, work, name, gpu))
        (work/'process.json').write_text(json.dumps(dict(pid=proc.pid, command=command, node=a.node, gpu=gpu, system=name)))
    previous_cpu = None
    with (folder/'resources.jsonl').open('a', buffering=1) as log:
        while True:
            gpu = subprocess.run(['nvidia-smi','--query-gpu=index,memory.used,memory.total,utilization.gpu,utilization.memory,power.draw',
                                  '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=15)
            cpu = list(map(int, Path('/proc/stat').read_text().splitlines()[0].split()[1:]))
            total, idle = sum(cpu[:8]), cpu[3]+cpu[4]
            busy = None if previous_cpu is None else 100*(1-(idle-previous_cpu[1])/max(1,total-previous_cpu[0]))
            previous_cpu=(total,idle)
            statuses=[dict(pid=p.pid,system=s,gpu=g,exit_code=p.poll()) for p,w,s,g in processes]
            value=dict(utc=datetime.now(timezone.utc).isoformat(),node=a.node,gpus=gpu.stdout.strip(),cpu_percent=busy,
                       loadavg=Path('/proc/loadavg').read_text().strip(),workers=statuses)
            log.write(json.dumps(value)+'\n'); (folder/'latest.json').write_text(json.dumps(value,indent=2)+'\n')
            if all(x['exit_code'] is not None for x in statuses): break
            time.sleep(10)


if __name__ == '__main__': main()
