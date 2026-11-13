#!/usr/bin/env python3
"""Launch a node's registered replicas, monitor resources, and retain exit status."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--node',required=True);a=p.parse_args()
    plan=json.loads(Path(a.plan).read_text());root=Path(plan['root']);node=plan['nodes'][a.node]
    systems={s['name']:s for s in json.loads(Path(plan['config']).read_text())['systems']}
    folder=root/'nodes'/a.node;folder.mkdir(parents=True,exist_ok=True);children=[]
    pending=list(node['jobs'])
    def launch(job):
        name=job['systems'][0];system=systems[name];worker_id=f"{a.node}-g{job['gpu']}-r{job['replica']}"
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(job['gpu']),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
                 HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',PYTHONUNBUFFERED='1',
                 HF_ENABLE_PARALLEL_LOADING='true',HF_PARALLEL_LOADING_WORKERS='2')
        cmd=[system['python'],str(Path(plan['code'])/'scripts/run_listen_benchmark_queue.py'),
             '--config',plan['config'],'--manifest',plan['manifest'],'--output',str(root/'results'),
             '--systems',','.join(job['systems']),'--worker-id',worker_id,'--start-offset',str(job.get('offset',0))]
        if system['backend']=='mediapipe':cmd+=['--device','cpu']
        with (folder/(worker_id+'.log')).open('w') as log:
            proc=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        children.append((proc,worker_id,cmd,job))
        (folder/'processes.json').write_text(json.dumps([dict(pid=p.pid,worker_id=w,command=c,job=j) for p,w,c,j in children],indent=2)+'\n')
    previous=None
    with (folder/'resources.jsonl').open('a',buffering=1) as log:
        while True:
            for job in pending[:]:
                pilot=Path(plan.get('pilot_root','/nonexistent'))/a.node/f"gpu{job['gpu']}"/'process.json'
                if pilot.exists():
                    record=json.loads(pilot.read_text());cmdline=Path('/proc')/str(record['pid'])/'cmdline'
                    if cmdline.exists() and b'benchmark_listen_labeling.py' in cmdline.read_bytes():continue
                if systems[job['systems'][0]]['backend']=='captioner':
                    path=Path(systems[job['systems'][0]]['model_path'])
                    if not all((path/f).is_file() for f in ('model.safetensors.index.json','tokenizer_config.json','vocab.json','preprocessor_config.json')):continue
                    index=json.loads((path/'model.safetensors.index.json').read_text())
                    if not all((path/f).is_file() for f in set(index['weight_map'].values())):continue
                launch(job);pending.remove(job)
            query=subprocess.run(['nvidia-smi','--query-gpu=index,memory.used,memory.total,utilization.gpu,utilization.memory,power.draw',
                                  '--format=csv,noheader,nounits'],text=True,capture_output=True,timeout=15)
            cpu=list(map(int,Path('/proc/stat').read_text().splitlines()[0].split()[1:]));total,idle=sum(cpu[:8]),cpu[3]+cpu[4]
            busy=None if previous is None else 100*(1-(idle-previous[1])/max(1,total-previous[0]));previous=(total,idle)
            value=dict(utc=datetime.now(timezone.utc).isoformat(),node=a.node,gpus=query.stdout.strip(),cpu_percent=busy,
                       loadavg=Path('/proc/loadavg').read_text().strip(),
                       pending_jobs=len(pending),workers=[dict(worker_id=w,pid=p.pid,exit_code=p.poll()) for p,w,c,j in children])
            log.write(json.dumps(value)+'\n')
            tmp=folder/'latest.json.tmp';tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(folder/'latest.json')
            if not pending and all(x['exit_code'] is not None for x in value['workers']):break
            time.sleep(10)


if __name__=='__main__':main()
