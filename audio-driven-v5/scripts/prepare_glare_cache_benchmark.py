#!/usr/bin/env python3
"""Prepare four isolated, identical-cohort GPU test arms. Never starts a GPU job."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shlex
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import glare_cache_cluster as cluster


def build_cohort(base,directory,lia_pairs=640,prosody_chunks=4096,seed=137):
    directory=Path(directory).resolve();directory.mkdir(parents=True,exist_ok=True)
    cohort_file=directory/'cohort.json'
    plan=Path(base['source_plan']);old=Path(base['shared_lia'])
    p=cluster.read(plan/'protocol.json');g=cluster.read(old/'protocol.json')
    def order(key):return hashlib.sha256((str(seed)+':'+str(key)).encode()).hexdigest()
    with cluster.ro(old/'groups.sqlite') as db:groups=list(db.execute('SELECT gid,pairs FROM groups'))
    gids=[];pairs=0
    for gid,n in sorted(groups,key=lambda x:order(x[0])):
        gids.append(gid);pairs+=n
        if pairs>=lia_pairs:break
    with cluster.ro(plan/'index.sqlite') as db:tracks=list(db.execute('SELECT track_id,COUNT(*) FROM chunks GROUP BY track_id'))
    tids=[];chunks=0
    for tid,n in sorted(tracks,key=lambda x:order(x[0])):
        tids.append(tid);chunks+=n
        if chunks>=prosody_chunks:break
    cohort=dict(seed=seed,track_ids=tids,group_ids=gids,pairs=pairs,chunks=chunks,
                source_index_sha256=p['index_sha256'],groups_sha256=g['groups_sha256'])
    if cohort_file.exists():assert cluster.read(cohort_file)==cohort,'Different cohort; choose another test directory'
    else:cluster.save(cohort_file,cohort)
    return cohort_file,cohort


def prepare(base,directory,lia_pairs=640,prosody_chunks=4096,seed=137):
    directory=Path(directory).resolve()
    cohort_file,cohort=build_cohort(base,directory,lia_pairs,prosody_chunks,seed)
    arms=[]
    for workers,async_ in [(1,False),(2,False),(1,True),(2,True)]:
        name=str(workers)+'_'+('async' if async_ else 'sync');cfg=copy.deepcopy(base)
        cfg['work']=str(directory/name/'run')
        cfg['execution']=dict(lia_workers_per_gpu=workers,async_lia=async_,async_prosody=async_,
                              lia_write_operations=16,prosody_write_operations=128)
        cfg['benchmark']=dict(cohort=str(cohort_file),output_root=str(directory/name/'run/outputs'),
                              tasks=dict(prosody=8,lia=40),force_lia_recompute=True)
        config_file=directory/(name+'.json');cluster.save(config_file,cfg)
        m=cluster.prepare(cluster.config(config_file))
        # The user's stop remains effective for every newly prepared arm.
        cluster.save(Path(cfg['work'])/'STOP',dict(reason='Prepared only; wait for explicit GPU test instruction'))
        command=[base['python'],str(ROOT/'scripts/run_glare_cache_cluster.py'),'--config',str(config_file),'start']
        arms.append(dict(name=name,config=str(config_file),run_id=m['run_id'],assigned_units=m['assigned_units'],
                         start_command=shlex.join(command),gpu_started=False))
    result=dict(cohort=str(cohort_file),same_samples=True,separate_output_caches=True,
                warmup_note='Report end-to-end time including model/JIT startup; repeat in reverse order for timing confirmation',arms=arms)
    cluster.save(directory/'TEST_PLAN.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,default=cluster.DEFAULT)
    p.add_argument('--out',type=Path,default=ROOT/'experiments/glare_cache_28gpu_20261110/benchmark_ready')
    p.add_argument('--lia-pairs',type=int,default=640);p.add_argument('--prosody-chunks',type=int,default=4096)
    a=p.parse_args();assert a.lia_pairs>=40 and a.prosody_chunks>=512
    print(json.dumps(prepare(cluster.config(a.config),a.out,a.lia_pairs,a.prosody_chunks)))
