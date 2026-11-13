"""Bounded, independent 128-pair QC4 validation. No full-run or training action."""
import argparse
from collections import Counter,defaultdict
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import run_listen_v8_full as full
import listen_audio_quality_v4 as qc4
from p3_listen.common import read,write,sha

ROOT=full.ROOT
PARENT=ROOT/'data_corpora/seamless_listen_v8_full'
OUT=ROOT/'data_corpora/seamless_listen_v8_qc4_sample'
EXP=ROOT/'experiments/p3_listen_v8_qc4_sample'
SEED=2026102302


def init():
    if (OUT/'contract.json').exists():
        full.OUT=OUT;full.verify();return
    assert full.OUT==PARENT
    parent=full.verify()
    assert read(PARENT/'review_hold.json')['state']=='awaiting_user_review'
    previous=read(ROOT/'data_previews/seamless_listen_v8_repair/samples.json')['samples']
    previous+=read(ROOT/'experiments/p3_listen_v8_boundary_validation/selection.json')['rows']
    seen_groups={r['group_id'] for r in previous}
    catalog=read(PARENT/'catalog.json');buckets=defaultdict(list)
    for row in catalog['rows']:
        if row['split']=='train' and row['group_id'] not in seen_groups and not (PARENT/'results'/(row['pair']+'.json')).exists():
            buckets[row['group_id']].append(row)
    rng=random.Random(SEED);groups=sorted(buckets);rng.shuffle(groups)
    assert len(groups)>=128
    selected=[rng.choice(buckets[group]) for group in groups[:128]]
    assert len({r['group_id'] for r in selected})==128
    OUT.mkdir(parents=True,exist_ok=True);EXP.mkdir(parents=True,exist_ok=True)
    for name in ('logs','progress','results','errors','audit','normalized_pairs','telemetry','claims'):(OUT/name).mkdir(exist_ok=True)
    for name in parent['input_sha256']:
        if name!='catalog.json':shutil.copyfile(PARENT/name,OUT/name)
    write(OUT/'catalog.json',dict(catalog,rows=selected,scope='128 previously unprocessed train pairs, one per group; disjoint from prior two displayed reviews'))
    shutil.copyfile(ROOT/'data_reviews/listener_v8_repair_feedback.json',OUT/'qc4_human_feedback.json')
    write(OUT/'selection.json',dict(seed=SEED,selected_pairs=[r['pair'] for r in selected],
        excluded_review_groups=sorted(seen_groups),distinct_groups=128,parent_catalog_sha256=sha(PARENT/'catalog.json')))
    write(OUT/'queue.json',dict(pairs=[r['pair'] for r in selected]));write(OUT/'queue_state.json',dict(cursor=0,total=128))
    write(OUT/'review_hold.json',dict(state='awaiting_user_review',full_run_authorized=False))
    contract=dict(parent,revision='v8-qc4-independent-sample-1',utc=full.now(),
        parent_contract_sha256=sha(PARENT/'contract.json'),additional_policy=qc4.POLICY,
        code_sha256=dict(parent['code_sha256']),input_sha256=dict(parent['input_sha256']),
        sample_only=True,source_pairs=128,workers=8,prep_per_worker=1,scope='Independent bounded validation, not a full run')
    for name in ('scripts/listen_audio_quality_v4.py','scripts/run_listen_v8_qc4_sample.py'):
        contract['code_sha256'][name]=sha(ROOT/name)
    for name in ('catalog.json','selection.json','qc4_human_feedback.json'):contract['input_sha256'][name]=sha(OUT/name)
    write(OUT/'contract.json',contract);full.OUT=OUT;full.verify()


def worker(rank):
    full.OUT=OUT
    original_screen=full.screen
    feedback=read(OUT/'qc4_human_feedback.json')['rows']
    def screen(*args,**kwargs):
        result=original_screen(*args,**kwargs)
        waves={role:full.legacy.read_wave(result['normalized_audio'][role]['path'],result['start'],result['duration_s']) for role in ('speaker','listener')}
        result['qc4']=qc4.assess(result,waves['speaker'],waves['listener'],feedback)
        return result
    full.screen=screen
    full.worker(rank,1)


def summary():
    counts=Counter();hours=Counter();source_bad=Counter();complete=0
    for p in (OUT/'results').glob('*.json'):
        r=read(p);assert r['complete'];complete+=1
        if not r['source_ok']:source_bad.update(x['reason'].split(':',1)[0] for x in r['rejected'])
        for row in r['clips']:
            status=row['qc4']['status'];counts[status]+=1;hours[status]+=row['duration_s']/3600
    value=dict(utc=full.now(),source_pairs=128,complete_pairs=complete,error_pairs=len(list((OUT/'errors').glob('*.json'))),
        candidate_counts=dict(counts),candidate_hours=dict(hours),source_failures=dict(source_bad),
        scope='Fresh group-disjoint sources; not reviewed/qualified training data',training_ready=False)
    write(OUT/'qc4_summary.json',value);return value


def run():
    full.OUT=OUT;full.verify()
    assert read(OUT/'queue_state.json')['cursor']==0,'Use a new frozen sample for another trial'
    children=[]
    for rank in range(8):
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(rank),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',
            MKL_NUM_THREADS='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
        with (OUT/'logs'/f'rank_{rank:03d}.log').open('ab',buffering=0) as log:
            children.append(subprocess.Popen([sys.executable,'-u',__file__,'worker','--rank',str(rank)],
                stdin=subprocess.DEVNULL,stdout=log,stderr=log,env=env))
    write(OUT/'runtime.json',dict(state='screening_small_batch',utc=full.now(),supervisor_pid=os.getpid(),
        pids=[p.pid for p in children],workers=8,total_preparation_slots=8,source_pairs=128,full_run=False))
    previous=full.cpu_sample()
    while any(p.poll() is None for p in children):
        time.sleep(15)
        previous=full.telemetry(previous);summary()
    codes=[p.wait() for p in children];s=summary()
    write(OUT/'runtime.json',dict(state='awaiting_user_review' if not any(codes) else 'worker_failure',utc=full.now(),
        workers=0,pids=[],exit_codes=codes,summary=s,full_run=False,training_started=False))
    if any(codes):raise RuntimeError('Small-batch worker failed; inspect logs')
    print(json.dumps(s),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['init','run','worker','summary'])
    parser.add_argument('--rank',type=int,default=0);args=parser.parse_args()
    if args.stage=='init':init()
    elif args.stage=='worker':worker(args.rank)
    elif args.stage=='run':run()
    else:print(json.dumps(summary()),flush=True)
