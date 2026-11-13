"""Repartition existing per-pair scores and exact distribution features, no rescoring."""
import copy
from pathlib import Path
import numpy as np
from .protocol import read,save,rows,sha


def feature_blocks(records,features):
    """Feature rows are concatenated in successful pair order by the native scorer."""
    cursor={m:0 for m in ('fid','fvd')};result={}
    for row in records:
        blocks={}
        for metric in cursor:
            detail=row.get(metric,{})
            if detail.get('status')!='features_ready':continue
            count=detail['feature_count'];begin=cursor[metric];end=begin+count
            blocks[metric]=tuple(features[metric+'_'+side][begin:end].copy() for side in ['gt','generated'])
            if any(len(x)!=count for x in blocks[metric]):raise ValueError('Distribution feature row mismatch')
            cursor[metric]=end
        if row['id'] in result:raise ValueError('Duplicate scored pair')
        result[row['id']]=blocks
    for m,n in cursor.items():
        if any(len(features[m+'_'+side])!=n for side in ['gt','generated']):
            raise ValueError('Feature offsets do not consume whole source shard')
    return result


def repartition(work,source,arms):
    """Only full, already-scored source arms; preserve native pixel/face/audio values."""
    from evaluation.video.paired_metrics.run import summarize,METRICS
    from evaluation.listener_metrics.run import quality_summary
    from evaluation.listener_metrics.core import aggregate_activity
    work=Path(work);source=Path(source);p=read(work/'protocol.json');old=read(source/'protocol.json')
    reports=[]
    for arm in arms:
        scored={};blocks={};audio={};contracts=[];source_files={}
        for rank in range(old['workers']):
            folder=source/'metric_shards'/arm/'responsenet'/f'{rank:02d}'
            records=rows(folder/'pairs.jsonl')
            with np.load(folder/'features.npz') as z:pieces=feature_blocks(records,z)
            for row in records:
                if row['id'] in scored:raise ValueError('Duplicate source score')
                scored[row['id']]=row;blocks[row['id']]=pieces[row['id']]
            for row in rows(folder/'audio/pairs.jsonl'):
                if row['id'] in audio:raise ValueError('Duplicate source audio score')
                audio[row['id']]=row
            contracts.append(read(folder/'contract.json'))
            for rel in ['pairs.jsonl','features.npz','contract.json','audio/pairs.jsonl']:
                source_files[str(folder/rel)]=sha(folder/rel)
        if any(c['code']!=contracts[0]['code'] or c['assets']!=contracts[0]['assets'] for c in contracts):
            raise ValueError('Source metric contracts differ')
        for rank in range(p['workers']):
            dest=work/'metric_shards'/arm/'responsenet'/f'{rank:02d}'
            if (dest/'REUSED.json').exists():continue
            ids=[x['id'] for x in read(work/'shards'/f'{rank:02d}.json')]
            records=[scored[sid] for sid in ids];arows=[audio[sid] for sid in ids]
            # Match source GT window and generated file to the new frozen binding.
            manifest={r['id']:r for r in rows(work/'metric_manifests'/arm/'responsenet'/f'{rank:02d}.jsonl')}
            for row in records:
                bound=manifest[row['id']]
                for key in ['gt','generated','source_start','source_end']:
                    if bound[key]!=row['input'][key]:raise ValueError('Source/new score input differs: '+key)
            values={};counts={}
            for m in ['fid','fvd']:
                selected=[blocks[sid][m] for sid in ids if m in blocks[sid]]
                if not selected:raise ValueError('Empty source distribution features')
                for k,side in enumerate(['gt','generated']):values[m+'_'+side]=np.concatenate([b[k] for b in selected])
                counts[m]=len(selected)
            dest.mkdir(parents=True,exist_ok=True);np.savez(dest/'features.npz',**values)
            # Shard FID/FVD are intentionally deferred; the unchanged global merge
            # computes them from ALL selected feature rows, never shard averages.
            summary=summarize(records,{},set(METRICS)-{'fid','fvd'},contracts[0]['schema'],
                contracts[0]['datasets'],len(records),{})
            for m in ['fid','fvd']:
                summary['metrics'][m]=None
                summary['coverage'][m]=dict(pairs=counts[m],samples=len(values[m+'_gt']))
            summary['status']='features_ready_for_global_merge'
            for name,data in [('pairs.jsonl',records),('audio/pairs.jsonl',arows)]:
                path=dest/name;path.parent.mkdir(parents=True,exist_ok=True)
                path.write_text(''.join(__import__('json').dumps(r,ensure_ascii=False)+'\n' for r in data))
            save(dest/'summary.json',summary)
            contract=copy.deepcopy(contracts[0]);contract.setdefault('execution',{})['score_reuse']=dict(
                source=str(source),pair_ids=ids,values_unchanged=True,features_repartitioned=True,
                distribution_metrics='global merge only',new_manifest_sha256=sha(work/'metric_manifests'/arm/'responsenet'/f'{rank:02d}.jsonl'))
            save(dest/'contract.json',contract)
            valid=[r for r in arows if 'audio' in r]
            save(dest/'audio/summary.json',dict(pairs_with_errors=sum(bool(r['errors']) for r in arows),
                requested_pairs=len(arows),quality=quality_summary(arows,('utmos',)),
                activity={k:aggregate_activity([r['audio']['comparison'][k] for r in valid]) for k in ['acoustic','speech']}))
            save(dest/'REUSED.json',dict(source=str(source),pairs=len(ids),all_pair_values_unchanged=True,
                features_consumed_exactly=True))
            save(work/'progress/metrics'/(arm+f'_{rank:02d}.json'),dict(status='complete',reused=True))
        reports.append(dict(arm=arm,selected_pairs=sum(len(read(work/'shards'/f'{k:02d}.json')) for k in range(p['workers'])),source_files=source_files))
    save(work/'REUSED_SCORES.json',dict(arms=reports,pixel_face_audio_values_unchanged=True,
        fid_fvd='recompute from selected200 complete feature sets at global merge'))
    return reports
