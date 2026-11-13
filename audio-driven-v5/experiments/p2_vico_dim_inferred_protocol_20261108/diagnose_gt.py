"""Bounded GT-only diagnostics of paper/code ambiguity, not protocol optimization."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.cluster import KMeans

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
WORK = ROOT/'experiments/p2_vico_literature_protocol_20261108'


def read(p):
    return json.loads(p.read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def entropy(labels,k):
    counts=np.bincount(labels,minlength=k).astype(np.float64)
    hist=counts/counts.sum()
    return float(-np.sum(hist*np.log2(hist+1e-6)))


def main():
    protocol=read(WORK/'protocol.json')
    original=read(WORK/'RESULTS.json')
    inputs={}
    gt={}
    for r in read(WORK/'EXTRACTION_COMPLETE.json')['records']:
        if r['role']!='gt':continue
        assert sha(r['path'])==r['sha256']
        with np.load(r['path']) as data:gt[r['id']]=data['motion'].copy()
        assert np.isfinite(gt[r['id']]).all()
        inputs[r['path']]=r['sha256']
    ids=protocol['ids']
    scopes={
        'official43_all_frames':[gt[i] for i in ids],
        'existing43_shared_valid_frames':[gt[c['id']][c['listener_frame_indices']] for c in original['coverage']],
    }
    scope_projection=read(WORK/'LOADER_SCOPE_DIAGNOSTIC.json')
    scopes['released_audio_key_and_video_length_projection']=[gt[i][1:] for i in scope_projection['ids_with_repetitions']]
    definitions={
        'released_pooled_flatten':'np.var(concatenate(clips).reshape(-1))',
        'pooled_per_coefficient_mean':'np.var(concatenate(clips),axis=0).mean()',
        'equal_clip_flatten':'mean([np.var(clip.reshape(-1)) for clip in clips])',
        'equal_clip_temporal_per_coefficient_mean':'mean([np.var(clip,axis=0).mean() for clip in clips])',
    }
    diagnostics={}
    for name, clips in scopes.items():
        joined=np.concatenate(clips)
        output={}
        for group,sl in [('exp',slice(6,56)),('pose',slice(0,6))]:
            a=joined[:,sl];xs=[c[:,sl] for c in clips]
            output[group]=dict(released_pooled_flatten=float(np.var(a)),
                pooled_per_coefficient_mean=float(np.var(a,axis=0).mean()),
                equal_clip_flatten=float(np.mean([np.var(x) for x in xs])),
                equal_clip_temporal_per_coefficient_mean=float(np.mean([np.var(x,axis=0).mean() for x in xs])))
            # Between-coefficient mean differences explain part of flattening variance.
            between=float(np.var(np.mean(a.astype(np.float64),axis=0)))
            temporal=float(np.var(a.astype(np.float64),axis=0).mean())
            assert abs(float(np.var(a.astype(np.float64)))-between-temporal)<1e-10
            output[group]['flatten_variance_decomposition_float64']=dict(
                mean_within_coefficient_variance=temporal,between_coefficient_means_variance=between)
        diagnostics[name]=dict(rows=len(clips),frames=len(joined),variance=output)
    # Supplement wording says within-sequence entropy; released code pools frames.
    # Both use the same fixed GT-fitted clusters, so no fitting to paper scores.
    clips=scopes['existing43_shared_valid_frames']
    joined=np.concatenate(clips)
    sid={}
    for group,sl,k in [('exp',slice(6,56),40),('pose',slice(0,6),20)]:
        model=KMeans(n_clusters=k,random_state=0,n_init='auto').fit(joined[:,sl])
        labels=model.predict(joined[:,sl])
        per_clip=[];at=0
        for clip in clips:
            per_clip.append(entropy(labels[at:at+len(clip)],k));at+=len(clip)
        sid[group]=dict(k=k,released_global_histogram=entropy(labels,k),
            same_clusters_equal_clip_entropy=float(np.mean(per_clip)),
            per_clip_entropy=per_clip)
        assert abs(sid[group]['released_global_histogram']-original['gt_metrics']['SID'][group])<1e-10
    result=dict(passed=True, scope='GT-only: no generated feature or metric used for configuration selection',
        hypotheses_declared_before_computation=list(definitions),variance_definitions=definitions,
        scopes=diagnostics,sid_same_clusters_aggregation=sid,
        published_gt=dict(Var=dict(exp=.93,pose=.01),SID=dict(exp=5.03,pose=4.07)),
        selected_by_proximity=False, no_rescaling=True, no_crop_or_checkpoint_sweep=True,
        confirms_exact_author_protocol=False,input_gt_feature_sha256=inputs,
        source_metrics_sha256=sha(ROOT/'compare/Dyadic-Interaction-Modeling/code/mymetrics.py'),
        implementation_sha256=sha(__file__))
    (HERE/'GT_DIAGNOSTICS.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(passed=True,scope_rows={k:v['rows'] for k,v in diagnostics.items()},
        official43_variance={g:{k:v for k,v in a.items() if not isinstance(v,dict)} for g,a in diagnostics['official43_all_frames']['variance'].items()},
        sid={g:{k:v for k,v in row.items() if k!='per_clip_entropy'} for g,row in sid.items()},
        confirms_exact_author_protocol=False),ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
