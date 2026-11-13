#!/usr/bin/env python3
"""Read actual completed test arms and compare timing, tensors and raw replies."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import glare_cache_cluster as cluster


def timing(cfg,m):
    stages={}
    for stage in cluster.STAGES:
        queue=cluster.TaskQueue(cfg['work'],m['run_id'],stage);rows=[]
        for task in queue.tasks:
            path=queue.folder(task)/'done.json'
            assert path.exists(),'Arm not complete: '+str(path)
            rows.append((cluster.read(queue.folder(task)/'owner.json'),cluster.read(path)))
        first=min(r[0]['claimed'] for r in rows);last=max(r[1]['finished'] for r in rows)
        units=sum(r[1]['task']['units'] for r in rows)
        stages[stage]=dict(units=units,elapsed_s=last-first,units_per_s=units/(last-first),
                           first_claim=first,last_commit=last,
                           restarts_or_retries=sum(len(cluster.read(queue.folder(t)/'attempts.json')['attempts'])
                             for t in queue.tasks if (queue.folder(t)/'attempts.json').exists()))
    return dict(stages=stages,wall_seconds=max(s['last_commit'] for s in stages.values())-min(s['first_claim'] for s in stages.values()))


def path_for(cfg,sid,kind):
    from hashlib import sha256
    root=Path(cfg['benchmark']['output_root'])/'nonprosody/cache'/sha256(sid.encode()).hexdigest()[:2]
    return root/(sid+'.'+kind)


def compare(left,right,cohort,manifest):
    import numpy as np
    import torch
    from glare_prosody_subset import ChunkCache
    sids=[]
    with cluster.ro(Path(left['shared_lia'])/'groups.sqlite') as groups,cluster.ro(Path(manifest['parent'])/'jobs.sqlite') as jobs:
        for gid in cohort['group_ids']:
            ordinals=json.loads(groups.execute('SELECT ordinals FROM groups WHERE gid=?',(gid,)).fetchone()[0])
            sids.extend(jobs.execute('SELECT sid FROM jobs WHERE ordinal=?',(o,)).fetchone()[0] for o in ordinals)
    result={k:dict(exact_pairs=0,nonexact_pairs=0,max_abs=0.,examples=[]) for k in ['lia','audio','statistics','reaction']}
    for sid in sids:
        for kind in ['lia','audio','statistics']:
            a=torch.load(path_for(left,sid,kind+'.pt'),map_location='cpu',weights_only=True)
            b=torch.load(path_for(right,sid,kind+'.pt'),map_location='cpu',weights_only=True)
            assert set(a)==set(b),'Tensor keys differ'
            exact=True;error=0.
            for k in a:
                assert a[k].shape==b[k].shape and a[k].dtype==b[k].dtype,'Tensor layout differs'
                exact&=torch.equal(a[k],b[k]);error=max(error,float((a[k]-b[k]).abs().max()) if a[k].numel() else 0.)
            row=result[kind];row['exact_pairs' if exact else 'nonexact_pairs']+=1;row['max_abs']=max(row['max_abs'],error)
            if not exact and len(row['examples'])<5:row['examples'].append(sid)
        with np.load(path_for(left,sid,'reaction.npz')) as a,np.load(path_for(right,sid,'reaction.npz')) as b:
            assert set(a.files)==set(b.files),'Reaction fields differ'
            exact=True;error=0.
            for k in a.files:
                assert a[k].shape==b[k].shape,'Reaction layout differs'
                numeric=a[k].dtype.kind in 'fc'
                exact&=np.array_equal(a[k],b[k],equal_nan=True) if numeric else np.array_equal(a[k],b[k])
                if numeric:
                    mask=np.isfinite(a[k])&np.isfinite(b[k])
                    error=max(error,float(np.max(np.abs(a[k][mask]-b[k][mask]))) if mask.any() else 0.)
            row=result['reaction'];row['exact_pairs' if exact else 'nonexact_pairs']+=1;row['max_abs']=max(row['max_abs'],error)
            if not exact and len(row['examples'])<5:row['examples'].append(sid)
    caches=[ChunkCache(Path(c['benchmark']['output_root'])/'prosody',manifest['cache_contract']) for c in [left,right]]
    prosody=dict(chunks=0,raw_equal=0,official_normalized_equal=0,examples=[])
    with cluster.ro(Path(left['source_plan'])/'index.sqlite') as db:
        for tid in cohort['track_ids']:
            track=json.loads(db.execute('SELECT data FROM tracks WHERE track_id=?',(tid,)).fetchone()[0])
            for (i,) in db.execute('SELECT chunk_index FROM chunks WHERE track_id=?',(tid,)):
                a,b=[c.load(track,i) for c in caches];assert a is not None and b is not None
                assert a['identity']==b['identity']
                raw_equal=a['analysis']['raw_text']==b['analysis']['raw_text']
                native_equal=a['official_normalized_text']==b['official_normalized_text']
                prosody['chunks']+=1;prosody['raw_equal']+=raw_equal;prosody['official_normalized_equal']+=native_equal
                if not native_equal and len(prosody['examples'])<5:prosody['examples'].append(dict(track=tid,chunk=i))
    return dict(pair_features=result,prosody=prosody)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--plan',type=Path,default=ROOT/'experiments/glare_cache_28gpu_20261110/benchmark_ready/TEST_PLAN.json')
    ap.add_argument('--compare-values',action='store_true');a=ap.parse_args();plan=cluster.read(a.plan)
    rows=[];configs=[];m=None
    for arm in plan['arms']:
        # Archived results retain their sealed config/input identities even if
        # the current execution-only adapter has since been optimized.
        cfg=cluster.config(arm['config']);m=cluster.load_run(cfg,verify=False);value=timing(cfg,m)
        configs.append(cfg);rows.append(dict(name=arm['name'],**value))
    baseline=rows[0]
    for row in rows:
        row['lia_speedup']=row['stages']['lia']['units_per_s']/baseline['stages']['lia']['units_per_s']
        row['prosody_speedup']=row['stages']['prosody']['units_per_s']/baseline['stages']['prosody']['units_per_s']
        row['wall_speedup']=baseline['wall_seconds']/row['wall_seconds']
    result=dict(arms=rows,timing_includes_model_startup=True,
                caveats=['Short cohort; CPU detector startup and stage-tail scheduling affect timings; not full-corpus speedup.'])
    if a.compare_values:
        cohort=cluster.read(plan['cohort'])
        result['content_comparisons']={row['name']:compare(configs[0],cfg,cohort,m) for row,cfg in zip(rows[1:],configs[1:])}
    target=a.plan.parent/'COMPARISON.json';cluster.save(target,result);print(json.dumps(result))


if __name__=='__main__':main()
