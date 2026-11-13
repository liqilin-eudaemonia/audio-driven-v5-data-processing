"""Scheduling-only V7 acceleration: shared work queue and concurrent CPU preparation.

The original frozen planner, normalization, audio models and decisions are used
unchanged. Existing completed results are preserved.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import run_listen_v7_bulk as bulk

OUT=bulk.OUT
SCHED=OUT/'scheduling_v2'
PREP_WORKERS=4
WORLD=24


def init():
    original=bulk.verify_contract()
    SCHED.mkdir(parents=True,exist_ok=True)
    if (SCHED/'contract.json').exists():
        raise ValueError('Queue already initialized; resume workers, do not overwrite assignments')
    rows=bulk.read(OUT/'catalog.json')['rows']
    done={p.stem for p in (OUT/'results').glob('*.json') if bulk.read(p).get('complete')}
    pending=[r['pair'] for r in rows if r['pair'] not in done]
    bulk.write(SCHED/'queue.json',dict(pairs=pending))
    bulk.write(SCHED/'state.json',dict(cursor=0,total=len(pending)))
    contract=dict(utc=bulk.now(),code_sha256=bulk.sha(__file__),
        original_contract_sha256=bulk.sha(OUT/'contract.json'),queue_sha256=bulk.sha(SCHED/'queue.json'),
        preparation_workers_per_gpu=PREP_WORKERS,gpu_workers=WORLD,
        existing_completed_pairs=len(done),pending_pairs=len(pending),
        policy='Scheduling only: original frozen prepare/normalize/scan; no threshold, model or batch change',
        original_revision=original['revision'])
    bulk.write_once(SCHED/'contract.json',contract)
    print(json.dumps(contract),flush=True)


def verify():
    original=bulk.verify_contract();contract=bulk.read(SCHED/'contract.json')
    assert contract['code_sha256']==bulk.sha(__file__)
    assert contract['original_contract_sha256']==bulk.sha(OUT/'contract.json')
    assert contract['queue_sha256']==bulk.sha(SCHED/'queue.json')
    assert bulk.sha(bulk.ROOT/'data_reviews/listener_noise_rejections.json')==bulk.sha(OUT/'human_rejections.json')
    return original


def claim(queue,rank):
    with (SCHED/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        state=bulk.read(SCHED/'state.json');index=state['cursor']
        if index>=len(queue):return None
        pair=queue[index]
        bulk.write(SCHED/'claims'/(pair+'.json'),dict(pair=pair,rank=rank,pid=os.getpid(),utc=bulk.now()))
        state['cursor']=index+1;bulk.write(SCHED/'state.json',state)
        return pair


def prepare(item):
    start=time.perf_counter()
    audit,norm=bulk.prepare(item)
    return audit,norm,time.perf_counter()-start


def worker(rank):
    import torch
    torch.set_num_threads(2)
    original=verify();bulk.preview.OUT=OUT;bulk.subturns.OUT=OUT
    lookup={r['pair']:r for r in bulk.read(OUT/'catalog.json')['rows']}
    queue=bulk.read(SCHED/'queue.json')['pairs']
    rejects=bulk.read(OUT/'human_rejections.json')['rows'];positives=bulk.read(OUT/'human_retained.json')['rows']
    counts=Counter();hours=Counter();timing=Counter();started=time.perf_counter()
    def report(stage,**kw):
        value=dict(utc=bulk.now(),rank=rank,pid=os.getpid(),stage=stage,scheduling='dynamic_v2',
            total_pairs=len(queue),counts=dict(counts),candidate_hours=dict(hours),timing_seconds=dict(timing),
            wall_seconds=time.perf_counter()-started,**kw)
        bulk.write(OUT/'progress'/f'rank_{rank:02d}.json',value)
        print(json.dumps(value),flush=True)
    report('loading_models')
    ast=bulk.EventModel(bulk.MODEL,batch_size=original['ast_batch'])
    panns=bulk.PannsModel(batch_size=original['panns_batch'])
    failures=0
    with ThreadPoolExecutor(max_workers=PREP_WORKERS) as pool:
        pending={}
        def fill():
            while len(pending)<PREP_WORKERS:
                pair=claim(queue,rank)
                if pair is None:break
                pending[pool.submit(prepare,lookup[pair])]=pair
                counts['claimed_pairs']+=1
        fill()
        while pending:
            before=time.perf_counter()
            completed,_=wait(pending,return_when=FIRST_COMPLETED)
            timing['waiting_for_preparation']+=time.perf_counter()-before
            for future in completed:
                pair=pending.pop(future);item=lookup[pair]
                # Submit a replacement before screening to keep CPU slots supplied.
                fill();report('screening_pair',pair=pair,inflight_prepare=len(pending))
                try:
                    audit,norm,seconds=future.result();timing['preparation_work_sum']+=seconds
                    if not audit['ok']:
                        reasons=[r.get('reason','') for r in audit['rejected']]
                        allowed=('source_orientation_not_upright','orientation_ambiguous',
                            'orientation_insufficient_landmarks','orientation_invalid_keypoint_shape',
                            'Source AV/keypoint length disagreement','Missing transcript')
                        if any(not str(r).startswith(allowed) for r in reasons):raise RuntimeError(str(reasons))
                    before=time.perf_counter()
                    results=[bulk.scan(row,norm,ast,panns,rejects,positives) for row in audit['clips']]
                    scan_seconds=time.perf_counter()-before;timing['audio_screening']+=scan_seconds
                    bulk.write(OUT/'results'/(pair+'.json'),dict(pair=pair,split=item['split'],group_id=item['group_id'],
                        complete=True,utc=bulk.now(),clips=results,rejected=audit['rejected'],source_ok=audit['ok'],
                        source_seconds=audit.get('source_seconds'),parent_candidates=audit.get('parent_candidates',0),
                        scheduling_revision=2,performance=dict(preparation_seconds=seconds,audio_scan_seconds=scan_seconds)))
                    for row in results:
                        key=item['split']+'/'+row['effective_decision'];counts[key]+=1;hours[key]+=row['unique_seconds']/3600
                    counts['complete_pairs']+=1;failures=0
                except Exception as e:
                    counts['error_pairs']+=1;failures+=1
                    bulk.write(OUT/'errors'/(pair+'.json'),dict(pair=pair,utc=bulk.now(),
                        error=type(e).__name__+': '+str(e),traceback=traceback.format_exc()))
                    if failures>=10:raise RuntimeError('Ten consecutive processing failures') from e
                report('pair_finished',pair=pair,inflight_prepare=len(pending))
    report('finished_with_errors' if counts['error_pairs'] else 'finished')


def launch_node(node):
    verify()
    with (OUT/f'node_{node}.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        children=[]
        for gpu in range(8):
            rank=node*8+gpu
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',
                MKL_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
            with (OUT/'logs'/f'fast_rank_{rank:02d}.log').open('ab',buffering=0) as log:
                child=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'worker','--rank',str(rank)],
                    stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=env)
            children.append((rank,child))
        bulk.write(OUT/f'node_{node}.json',dict(node=node,utc=bulk.now(),state='running',scheduling='dynamic_v2',
            supervisor_pid=os.getpid(),workers=[dict(rank=r,pid=p.pid) for r,p in children]))
        codes={str(r):p.wait() for r,p in children}
        bulk.write(OUT/f'node_{node}.json',dict(node=node,utc=bulk.now(),state='finished',scheduling='dynamic_v2',exit_codes=codes))
        if any(codes.values()):raise SystemExit(1)


def detach(node):
    cmd=[sys.executable,'-u',str(Path(__file__).resolve()),'launch-node','--node',str(node)]
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    with (OUT/'logs'/f'fast_node_{node}.log').open('ab',buffering=0) as log:
        child=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
            env=env,start_new_session=True)
    value=dict(node=node,pid=child.pid,utc=bulk.now(),command=cmd)
    bulk.write(SCHED/f'launch_{node}.json',value);print(json.dumps(value),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['init','worker','launch-node','detach'])
    p.add_argument('--node',type=int,choices=range(3),default=0);p.add_argument('--rank',type=int,choices=range(24),default=0)
    a=p.parse_args()
    if a.stage=='init':init()
    elif a.stage=='worker':worker(a.rank)
    elif a.stage=='launch-node':launch_node(a.node)
    else:detach(a.node)
