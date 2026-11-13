#!/usr/bin/env python3
"""Prepare, cache and verify selected final-crop J clips on local GPUs."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from p2_listen.contracts import atomic_json,read


def gpu_snapshot():
    fields='index,memory.used,utilization.gpu,utilization.memory,power.draw'
    result=subprocess.run(['nvidia-smi','--query-gpu='+fields,'--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    rows=[]
    for line in result.stdout.splitlines():
        values=[v.strip() for v in line.split(',')]
        rows.append(dict(gpu=int(values[0]),memory_mib=float(values[1]),gpu_util=float(values[2]),
                         memory_util=float(values[3]),power_w=float(values[4])))
    return dict(time=time.time(),gpus=rows)


def run(work,gpus,text_batch_size):
    from p2_j_cache.data import open_manifest,sample
    work=Path(work);m=open_manifest(work)
    if len(set(gpus))!=len(gpus) or 0 not in gpus or len(gpus)>len(m['sample_ids']):
        raise ValueError('Use unique GPUs including GPU0, with at least one sample per worker')
    items=sorted([sample(work,m,sid) for sid in m['sample_ids']],key=lambda x:x['source']['duration_s'])
    groups={gpu:[] for gpu in gpus}
    # Keep the small review run balanced across all available GPUs.
    per=len(items)//len(gpus)
    for item in items[:per]:groups[0].append(item['sample_id'])
    rest=gpus[1:]
    if not rest:
        groups[0]=m['sample_ids']
    else:
        loads={gpu:0. for gpu in rest}
        for item in reversed(items[per:]):
            gpu=min(rest,key=lambda x:loads[x]);groups[gpu].append(item['sample_id']);loads[gpu]+=item['source']['duration_s']
    baseline=gpu_snapshot();atomic_json(work/'gpu_baseline.json',baseline)
    atomic_json(work/'assignments.json',{str(k):v for k,v in groups.items()})
    processes={};logs={};begin=time.monotonic();measurements=[]
    try:
        for gpu in gpus:
            path=work/'logs'/f'gpu{gpu}.log';path.parent.mkdir(parents=True,exist_ok=True)
            logs[gpu]=path.open('w')
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',
                OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',HF_ENABLE_PARALLEL_LOADING='true',
                HF_PARALLEL_LOADING_WORKERS='4',PYTHONUNBUFFERED='1')
            cmd=[sys.executable,str(Path(__file__).resolve()),'worker','--work',str(work),'--gpu',str(gpu),
                 '--text-batch-size',str(text_batch_size)]
            processes[gpu]=subprocess.Popen(cmd,stdout=logs[gpu],stderr=subprocess.STDOUT,env=env,cwd=ROOT)
        atomic_json(work/'processes.json',{str(k):v.pid for k,v in processes.items()})
        last=0.
        with (work/'gpu_samples.jsonl').open('w') as stream:
            while True:
                snap=gpu_snapshot();measurements.append(snap);stream.write(json.dumps(snap)+'\n');stream.flush()
                states={gpu:p.poll() for gpu,p in processes.items()}
                failed={gpu:rc for gpu,rc in states.items() if rc not in (None,0)}
                if failed:raise RuntimeError('GPU workers failed; see logs: '+str(failed))
                if all(rc==0 for rc in states.values()):break
                if time.monotonic()-last>=20:
                    print(json.dumps(dict(elapsed_s=round(time.monotonic()-begin,1),running=[g for g,r in states.items() if r is None],
                        gpu_util={r['gpu']:r['gpu_util'] for r in snap['gpus']})),flush=True);last=time.monotonic()
                time.sleep(1)
        from p2_j_cache.runtime import verify
        result=verify(work)
        resources={}
        for gpu in gpus:
            rows=[s for m in measurements for s in m['gpus'] if s['gpu']==gpu]
            active=[r['gpu_util'] for r in rows if r['gpu_util']>0]
            resources[str(gpu)]=dict(mean_util=sum(r['gpu_util'] for r in rows)/len(rows),
                active_mean_util=sum(active)/len(active) if active else 0.,peak_util=max(r['gpu_util'] for r in rows),
                peak_memory_mib=max(r['memory_mib'] for r in rows),samples=len(rows),nonzero_util_samples=len(active))
        result=dict(result,gpus=gpus,elapsed_s=time.monotonic()-begin,resources=resources,
                    resource_note='Model-loading intervals are included in device metrics.')
        atomic_json(work/'result.json',result)
        print(json.dumps({k:v for k,v in result.items() if k!='results'},ensure_ascii=False,indent=2),flush=True)
        return result
    finally:
        for p in processes.values():
            if p.poll() is None:p.terminate()
        for p in processes.values():
            try:p.wait(timeout=20)
            except subprocess.TimeoutExpired:p.kill();p.wait()
        for log in logs.values():log.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    prep=sub.add_parser('prepare');prep.add_argument('--work',required=True)
    prep.add_argument('--config',default=str(ROOT/'experiments/p2_train50h_postprocessed_reselected_20261027/config.json'))
    prep.add_argument('--sample-list',required=True);prep.add_argument('--selection');prep.add_argument('--review')
    launch=sub.add_parser('run');launch.add_argument('--work',required=True)
    launch.add_argument('--gpus',type=int,nargs='+',default=list(range(8)));launch.add_argument('--text-batch-size',type=int,default=5)
    w=sub.add_parser('worker');w.add_argument('--work',required=True);w.add_argument('--gpu',type=int,required=True)
    w.add_argument('--text-batch-size',type=int,default=5);w.add_argument('--memory-gib',type=float)
    check=sub.add_parser('verify');check.add_argument('--work',required=True)
    args=parser.parse_args()
    if args.command=='prepare':
        from p2_j_cache.data import prepare
        value=prepare(args.config,args.sample_list,args.work,args.selection,args.review)
        print(json.dumps(dict(work=args.work,samples=len(value['sample_ids']),splits={k:len(v) for k,v in value['splits'].items()})),flush=True)
    elif args.command=='run':run(args.work,args.gpus,args.text_batch_size)
    elif args.command=='worker':
        from p2_j_cache.runtime import worker
        ids=read(Path(args.work)/'assignments.json')[str(args.gpu)]
        worker(args.work,args.gpu,ids,args.text_batch_size,args.memory_gib)
    else:
        from p2_j_cache.runtime import verify
        value=verify(args.work);print(json.dumps({k:v for k,v in value.items() if k!='results'}),flush=True)


if __name__=='__main__':main()
