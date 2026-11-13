"""Explicit full-set and supported-subset denominators; one pool per seed."""
from pathlib import Path
import math
import numpy as np
from .common import module


def support(items,field=None):
    def value(item):return item[field] if field else item
    good=[i for i in items if value(i)['status']=='success']
    na=[i for i in items if value(i)['status']=='not_applicable']
    failed=[i for i in items if value(i)['status']=='error']
    if len(good)+len(na)+len(failed)!=len(items):raise ValueError('Unknown metric status')
    sid=lambda i:dict(sample_id=i['sample_id'],sample_seed=i['sample_seed'])
    return good,dict(planned=len(items),eligible=len(good)+len(failed),succeeded=len(good),inapplicable=len(na),failed=len(failed),
        succeeded_ids=[sid(i) for i in good],inapplicable_ids=[sid(i) for i in na],failed_ids=[sid(i) for i in failed])


def finish(value,coverage,**extra):
    if value is not None:
        vals=value.values() if isinstance(value,dict) else [value]
        if not all(math.isfinite(v) for v in vals):raise ValueError('Nonfinite aggregate')
    return dict(full_value=value if coverage['succeeded']==coverage['planned'] else None,
                supported_subset_value=value,coverage=coverage,**extra)


def numerical_distance(raw,means_covariances):
    m1,c1,m2,c2=means_covariances
    scale=max(1.,float(np.trace(np.atleast_2d(c1))+np.trace(np.atleast_2d(c2))+np.sum((m1-m2)**2)))
    tolerance=1e-6*scale
    if not math.isfinite(raw) or raw < -tolerance:raise ValueError('Unstable reference Frechet distance: '+str(raw))
    # Low-rank (small-N) sqrtm can give a tiny negative self-distance. Keep the
    # exact reference result for audit; only nonnegative distance is reported.
    return max(0.,raw),dict(raw_reference_value=raw,negative_roundoff_clamped=raw<0,roundoff_tolerance=tolerance)


def distributions(rows,metric,reference_root,expected_options):
    entries=[]
    for r in rows:
        sides=r['metrics'][metric]
        bad=[sides[s] for s in ('gt','generated') if sides[s]['status']=='error']
        na=[sides[s] for s in ('gt','generated') if sides[s]['status']=='not_applicable']
        status='error' if bad else ('not_applicable' if na else 'success')
        entries.append(dict(sample_id=r['sample_id'],sample_seed=r['sample_seed'],pair=dict(status=status),sides=sides))
    good,coverage=support(entries,'pair');value=None;extra={}
    if len(good)>=(2 if metric=='fvd' else 1):
        if metric=='fvd':
            ref=module('p2_score_fvd',Path(reference_root)/'video/fvd/frechet_video_distance.py')
            data={s:np.stack([np.load(r['sides'][s]['features'],allow_pickle=False) for r in good]) for s in ('gt','generated')}
            if data['gt'].shape!=data['generated'].shape:raise ValueError('FVD feature dimensions differ')
            stats=[]
            for s in ('gt','generated'):
                st=ref.FeatureStats(len(good));st.append(data[s]);stats.extend(st.get_mean_cov())
            raw=float(ref.frechet_distance(*stats,eps=1e-6));value,audit=numerical_distance(raw,stats);extra.update(audit)
            extra['name']='FVD-'+str(expected_options['fvd_frames'])
        else:
            ref=module('p2_score_fd',Path(reference_root)/'video/fd/compute_fd_features.py')
            data={s:np.concatenate([np.load(r['sides'][s]['features'],allow_pickle=False) for r in good]) for s in ('gt','generated')}
            if data['gt'].shape[1]!=data['generated'].shape[1]:raise ValueError('FD feature dimensions differ')
            if min(len(v) for v in data.values())>=2:
                stats=(*ref.calculate_activation_statistics(data['gt']),*ref.calculate_activation_statistics(data['generated']))
                raw=float(ref.calculate_frechet_distance(*stats));value,audit=numerical_distance(raw,stats);extra.update(audit)
            extra.update(name='FD',frame_counts={s:len(v) for s,v in data.items()},unit='all supplied frames pooled, sample covariance')
    return finish(value,coverage,reason=None if value is not None else 'need_at_least_two_supported_samples',**extra)


def scalar_side(rows,metric,side):
    entries=[dict(sample_id=r['sample_id'],sample_seed=r['sample_seed'],score=r['metrics'][metric][side]) for r in rows]
    good,coverage=support(entries,'score');value=None;extra={}
    if good:
        if metric=='utmos':value=float(np.mean([r['score']['value'] for r in good]))
        else:
            tracks=[t for r in good for t in r['score']['tracks'] if t['status']=='success']
            value={k:float(np.mean([t[k] for t in tracks])) for k in ('lse_d','lse_c')}
            extra=dict(unit='equal weight per supported face track',tracks=len(tracks),low_support_tracks=sum(t['low_support'] for t in tracks),
                clip_equal_mean={k:float(np.mean([r['score'][k] for r in good])) for k in ('lse_d','lse_c')})
    return finish(value,coverage,**extra)


def text_score(rows,reference_root):
    entries=[dict(sample_id=r['sample_id'],sample_seed=r['sample_seed'],score=r['metrics']['text'],texts=r['texts']) for r in rows]
    good,coverage=support(entries,'score');value=None
    if good:
        keys=('meteor','rouge_l','bertscore_p','bertscore_r','bertscore_f1')
        value={k:float(np.mean([r['score'][k] for r in good])) for k in keys}
        ref=module('p2_score_distinct',Path(reference_root)/'text/compute_distinct.py')
        for side in ('generated','gt'):
            d1,d2=ref.compute_distinct([r['texts'][side] for r in good]);value[side+'_distinct1']=d1;value[side+'_distinct2']=d2
    return finish(value,coverage,unit='pair mean; Distinct is corpus-level on the stated supported set')


def aggregate(protocol,rows):
    result={};root=protocol['runtime']['paths']['reference_root']
    for seed in protocol['seeds']:
        selected=[r for r in rows if r['sample_seed']==seed]
        if len(selected)!=protocol['sample_count']:raise ValueError('Missing planned row in aggregation')
        scores={}
        for metric in protocol['metrics']:
            try:
                if metric in ('fvd','fd'):scores[metric]=distributions(selected,metric,root,protocol['options'])
                elif metric=='text':scores[metric]=text_score(selected,root)
                else:scores[metric]={side:scalar_side(selected,metric,side) for side in ('gt','generated')}
            except Exception as error:scores[metric]=dict(status='error',error=f'{type(error).__name__}: {error}',full_value=None,supported_subset_value=None)
        result[str(seed)]=scores
    def failed(v):
        if isinstance(v,dict):return v.get('status')=='error' or any(failed(x) for x in v.values())
        if isinstance(v,list):return any(failed(x) for x in v)
        return False
    def average(values):
        if any(v is None for v in values):return dict(mean=None,std=None,reason='at_least_one_seed_has_no_full_set_value')
        if isinstance(values[0],dict):return {k:average([v[k] for v in values]) for k in values[0]}
        return dict(mean=float(np.mean(values)),std=float(np.std(values)))
    means={}
    for metric in protocol['metrics']:
        scores=[result[str(seed)][metric] for seed in protocol['seeds']]
        if metric in ('fvd','fd','text'):means[metric]=average([s.get('full_value') for s in scores])
        else:means[metric]={side:average([s.get(side,{}).get('full_value') for s in scores]) for side in ('gt','generated')}
    return dict(schema='p2-omni-results-v1',status='incomplete' if failed(rows) or failed(result) else 'complete',
        sample_count=protocol['sample_count'],seeds=protocol['seeds'],options=protocol['options'],per_seed=result,rows=rows,
        full_set_across_seeds=means,
        note='full_value is null unless every planned sample is supported; subset values require matched ID sets for comparisons')
