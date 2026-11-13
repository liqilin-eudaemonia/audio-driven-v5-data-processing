#!/usr/bin/env python3
"""Explicit, bounded single-host 8-GPU annotation benchmark. No remote launching."""
import argparse
import csv
import io
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
PYTHON='/private/software/conda/envs/vllm/bin/python'


def save(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)


def gpu_stats():
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw',
                                 '--format=csv,noheader,nounits'],text=True)
    return [dict(zip(('gpu','utilization','memory_used_mib','memory_total_mib','power_w'),
                     [float(v.strip()) for v in row])) for row in csv.reader(io.StringIO(raw))]


def cpu_counters():
    return list(map(int,Path('/proc/stat').read_text().splitlines()[0].split()[1:9]))


def summarize(out):
    results=[]
    for path in sorted((out/'results').glob('*.json')):
        r=json.loads(path.read_text());p=r['performance'];records=r['records']
        results.append({'sample_id':r['sample_id'],'duration_s':r['duration_s'],
            'wall_s':p['prefetch_wait_s']+p['process_s'],'generate_s':p['generate_s'],
            'decode_s':p['decode_s'],'prepare_s':p['prepare_s'],
            'requests':len(records),'valid':sum(x['status']=='valid' for x in records),
            'truncated':sum(x['finish_reason']=='length' for x in records),
            'prompt_tokens':sum(x['prompt_tokens'] for x in records),
            'output_tokens':sum(x['completion_tokens'] for x in records),
            'events':sum(len(x.get('parsed',{}).get('events',[])) for x in records)})
    seconds=sum(r['wall_s'] for r in results)
    return {'samples':len(results),'active_wall_s':seconds,
            'clips_per_s':len(results)/seconds if seconds else None,
            'valid_requests_per_s':sum(r['valid'] for r in results)/seconds if seconds else None,
            'invalid_requests':sum(r['requests']-r['valid'] for r in results),
            'truncated_requests':sum(r['truncated'] for r in results),'samples_detail':results}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--matrix',type=Path,required=True)
    ap.add_argument('--timeout-minutes',type=float,default=40)
    ap.add_argument('--prepared-media-cache',type=Path)
    ap.add_argument('--model-loaders',type=int,default=2)
    ap.add_argument('--allow-shared',action='store_true',help='Explicit operator authorization required')
    args=ap.parse_args()
    if args.output.exists():raise SystemExit('Use a fresh output directory')
    devices=gpu_stats()
    if len(devices)!=8 or (not args.allow_shared and any(d['memory_used_mib']>1000 or d['utilization']>5 for d in devices)):
        raise SystemExit('This benchmark requires eight idle GPUs; no other process will be stopped')
    jobs=json.loads(args.matrix.read_text())
    if sorted(j['gpu'] for j in jobs)!=list(range(8)):
        raise SystemExit('Exactly one worker per GPU is required')
    if args.allow_shared:
        for j in jobs:
            d=devices[j['gpu']]
            if d['memory_total_mib']-d['memory_used_mib'] < d['memory_total_mib']*j.get('memory',.85)+12288:
                raise SystemExit(f"GPU {j['gpu']} lacks requested reservation plus 12 GiB headroom")
    args.output.mkdir(parents=True)
    locks=args.output/'resource_locks';locks.mkdir()
    procs=[];handles=[];start=time.time()
    if args.prepared_media_cache:
        from concurrent.futures import ThreadPoolExecutor
        sys.path.insert(0,str(ROOT/'src'))
        from listen_benchmark.visual_native8 import prepare_media
        from listen_benchmark.visual_media_cache import cached_media
        def prepare(row):
            def decode():
                tick=time.monotonic();whole=prepare_media(row['source'])
                return whole,{'decode_s':time.monotonic()-tick,'decode_lock_wait_s':0.}
            _,metrics=cached_media(row['source'],args.prepared_media_cache,decode)
            return {'sample_id':row['sample_id'],**metrics}
        with ThreadPoolExecutor(max_workers=4) as pool:
            cache_metrics=list(pool.map(prepare,json.loads(args.manifest.read_text())))
        save(args.output/'cache_build.json',{'wall_s':time.time()-start,'samples':cache_metrics,
            'directory':str(args.prepared_media_cache),'not_included_in_inference_throughput':True})
    save(args.output/'protocol.json',{'host':socket.gethostname(),'started_at':start,
        'manifest':str(args.manifest),'matrix':jobs,'initial_gpus':devices,
        'decode_readers_per_host':max(j.get('decode_slots',1) for j in jobs),
        'model_loaders':args.model_loaders,'other_tasks_stopped':False,'shared_gpu_authorized':args.allow_shared})
    try:
        for job in jobs:
            out=args.output/job['name'];(out/'locks').mkdir(parents=True)
            slots=job.get('decode_slots',1)
            lock_pairs=[('decode-'+socket.gethostname()+(f'-{i}' if slots>1 else ''),f'decode-{i}') for i in range(slots)]
            lock_pairs.append(('model-init-'+socket.gethostname(),f"model-init-{job['gpu']%args.model_loaders}"))
            for name,target in lock_pairs:
                (out/'locks'/name).symlink_to(locks/target)
            cmd=[PYTHON,str(ROOT/'scripts/listen_visual_native8_nod.py'),'vlm',
                 '--manifest',str(args.manifest),'--output',str(out),'--memory',str(job.get('memory',.85)),
                 '--batch-size',str(job['batch']),'--prefetch',str(job['prefetch']),
                 '--max-num-batched-tokens',str(job['tokens']),
                 '--batch-frame-budget',str(job.get('frames',256)),'--scheduling','balanced',
                 '--clip-batch-size',str(job.get('clips',1)),'--decode-slots',str(slots),
                 '--decode-workers',str(job.get('decode_workers',1)),
                 '--rank',str(job.get('rank',0)),'--shards',str(job.get('shards',1))]
            if args.prepared_media_cache:cmd+=['--prepared-media-cache',str(args.prepared_media_cache)]
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(job['gpu']),OMP_NUM_THREADS='2',
                     OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',
                     LD_PRELOAD='/private/software/conda/envs/vllm/lib/libstdc++.so.6')
            f=(out/'worker.log').open('w');handles.append(f)
            proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            procs.append((job,proc,out))
            save(args.output/'workers.json',[dict(j,pid=p.pid,output=str(o)) for j,p,o in procs])
        previous_cpu=cpu_counters()
        with (args.output/'resources.jsonl').open('a',buffering=1) as log:
            while any(p.poll() is None for _,p,_ in procs):
                current_cpu=cpu_counters();delta=[b-a for a,b in zip(previous_cpu,current_cpu)];previous_cpu=current_cpu
                state={'time':time.time(),'gpus':gpu_stats(),
                       'host_cpu_busy_pct':100*(sum(delta)-delta[3]-delta[4])/max(1,sum(delta)),
                       'host_loadavg':Path('/proc/loadavg').read_text().strip(),
                       'workers':[{'name':j['name'],'pid':p.pid,'exit_code':p.poll()} for j,p,_ in procs]}
                log.write(json.dumps(state)+'\n');save(args.output/'state.json',state)
                if time.time()-start>args.timeout_minutes*60:raise TimeoutError('Benchmark time budget exceeded')
                time.sleep(10)
        report={j['name']:dict(summarize(out),gpu=j['gpu'],exit_code=p.returncode,settings=j)
                for j,p,out in procs}
        save(args.output/'summary.json',report)
        print(json.dumps({k:{x:v[x] for x in ('samples','clips_per_s','invalid_requests','truncated_requests','exit_code')}
                          for k,v in report.items()}),flush=True)
    finally:
        for _,p,_ in procs:
            if p.poll() is None:
                os.killpg(p.pid,signal.SIGTERM)
        for _,p,_ in procs:
            try:p.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid,signal.SIGKILL);p.wait()
        for f in handles:f.close()


if __name__=='__main__':main()
