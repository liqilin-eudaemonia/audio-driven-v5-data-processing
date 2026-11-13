"""Paired comparison only on explicitly shared GT and supported sample IDs."""
from pathlib import Path
from .common import *
from .pipeline import check_protocol
from .scoring import aggregate


def compare(left,right,output):
    left,right=Path(left),Path(right);a,b=[read(p/'protocol.json') for p in (left,right)]
    for p in (a,b):check_protocol(p)
    check_runtime(a['runtime'])
    for k in ('metrics','options','seeds','runtime','code','sample_count'):
        if a[k]!=b[k]:raise ValueError('Unmatched comparison protocol: '+k)
    def references(p):
        return {(r['sample_id'],r['sample_seed']):{k:v for k,v in r.items() if k.startswith('gt_') or k in ('feature_schema','text_provenance')}
            for r in p['rows']}
    if references(a)!=references(b):raise ValueError('Comparison GT/IDs differ')
    x,y=[read(p/'results.json') for p in (left,right)]
    for root,protocol,result in ((left,a,x),(right,b,y)):
        if result['protocol_sha256']!=digest(protocol):raise ValueError('Results do not belong to protocol')
        for row in result['rows']:
            path=root/'rows'/job_key(row['sample_id'],row['sample_seed'])
            if read(path/'result.json')!=row:raise ValueError('Report and row artifact differ')
            planned=next(r for r in protocol['rows'] if (r['sample_id'],r['sample_seed'])==(row['sample_id'],row['sample_seed']))
            if not _error(row):
                if not artifact_valid(path,dict(protocol=digest(protocol),row=planned,device=result['device'])):raise ValueError('Missing successful row receipt')
    if x['device']!=y['device']:raise ValueError('Use the same device backend for paired scores')
    by=[{(r['sample_id'],r['sample_seed']):r for r in result['rows']} for result in (x,y)];comparisons={}
    if set(by[0])!=set(by[1]):raise ValueError('Result sample IDs differ')
    for metric in a['metrics']:
        comparisons[metric]={}
        for seed in a['seeds']:
            shared=[]
            for key in sorted(by[0]):
                if key[1]!=seed:continue
                if all(_supported(rows[key]['metrics'][metric]) for rows in by):shared.append(key)
            value=dict(common_sample_ids=[sid for sid,_ in shared],common_count=len(shared),planned=a['sample_count'])
            if shared:
                protocol=dict(a,metrics=[metric],seeds=[seed],sample_count=len(shared))
                for name,rows in zip(('left','right'),by):value[name]=aggregate(protocol,[rows[k] for k in shared])['per_seed'][str(seed)][metric]
            else:value.update(left=None,right=None)
            comparisons[metric][str(seed)]=value
    report=dict(status='compared',left=str(left.resolve()),right=str(right.resolve()),metrics=comparisons,
        note='Recomputed on common supported sample IDs; report common_count/planned alongside every comparison')
    atomic_json(output,report);return report


def _error(value):
    from .pipeline import has_error
    return has_error(value)


def _supported(value):
    if 'status' in value:return value['status']=='success'
    return all(_supported(v) for v in value.values())
