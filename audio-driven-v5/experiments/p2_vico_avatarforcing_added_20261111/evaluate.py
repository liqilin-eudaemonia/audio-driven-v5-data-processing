"""Add verified AF audio-only videos to the current common ViCo evaluation."""
from pathlib import Path
import sys,json,csv,copy,time
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'compare/deployment'))
from vico_more_common import read,save,sha
AF=ROOT/'experiments/p2_vico43_unified_20261107/avatarforcing'
CURRENT=ROOT/'experiments/p2_vico_full_eval_20261111'
PRIOR=ROOT/'experiments/p2_public_weights_vico_20261110'


def prepare():
    p=read(AF/'protocol.json')
    assert read(AF/'GENERATION_COMPLETE.json')['clips']==43
    for path,h in p['assets'].items():assert sha(path)==h,path
    records=[r for r in read(PRIOR/'EXTRACTION_COMPLETE.json')['records'] if r['role']=='avatarforcing_audio_only']
    tasks={r['id']:r for r in read(PRIOR/'protocol.json')['tasks'] if r['role']=='avatarforcing_audio_only'}
    assert len(records)==len(tasks)==43
    for sample in p['samples']:
        uid=sample['id'];dest=AF/'generated'/uid;r=read(dest/'complete.json')
        assert r['protocol_sha256']==sha(AF/'protocol.json')
        assert r['source_only'] and r['user_visual_zero'] and not r['listener_gt_audio_read'] and not r['listener_future_video_read']
        assert sha(dest/'generated.mp4')==r['native_sha256']
        assert sha(r['score_path'])==r['score_sha256']==tasks[uid]['video_sha256']
        assert sha(sample['image'])==sample['image_sha256']==r['input_image_sha256']
        assert r['speaker_waveform_sha256']==sample['speaker_waveform_sha256']
    extraction=PRIOR/'EXTRACTION_PROTOCOL.json'
    for r in records:
        assert sha(r['path'])==r['sha256'] and r['video_sha256']==tasks[r['id']]['video_sha256']
        assert r['extraction_protocol_sha256']==sha(extraction)
    for name in ['COMPLETE.json']:
        for path,h in read(CURRENT/name)['files'].items():assert sha(path)==h,path
    audit=dict(passed=True,clips=43,assets=p['assets'],generation_protocol_sha256=sha(AF/'protocol.json'),
        feature_protocol_sha256=sha(extraction),current_results_sha256=sha(CURRENT/'COMPLETE.json'),
        inference_reused=True,input_profile=p['input'],records=records)
    save(OUT/'INPUT_AUDIT.json',audit)
    folder=OUT/'image_eval';folder.mkdir(exist_ok=True);(folder/'scoring').mkdir(exist_ok=True)
    target=folder/'scoring/avatarforcing'
    if not target.exists():target.symlink_to(AF/'scoring/avatarforcing',target_is_directory=True)
    source=ROOT/'experiments/p2_vico256_combined8000_20261106'
    save(folder/'protocol.json',dict(method='avatarforcing',label='Avatar Forcing official weights, audio-only adaptation',datasets=read(source/'protocol.json')['datasets']))
    dest=folder/'INPUT_PREFLIGHT.json'
    if not dest.exists():dest.symlink_to(source/'INPUT_PREFLIGHT.json')
    return records


def score():
    import numpy as np
    from vico_literature_metrics import evaluate,parity
    added=prepare();p=read(CURRENT/'unified_emoca/protocol.json')
    records=p['reused_feature_records']+read(CURRENT/'unified_emoca/EXTRACTION_COMPLETE.json')['records']+added
    methods=p['methods']+['avatarforcing_audio_only'];features={}
    for r in records:
        assert sha(r['path'])==r['sha256']
        key=(r['id'],r['role']);assert key not in features
        with np.load(r['path']) as z:features[key]=z['motion'].copy()
    assert len(features)==len(p['ids'])*(len(methods)+2)
    masks={};coverage=[]
    for uid in p['ids']:
        n=len(features[uid,'gt'])
        assert all(features[uid,role].shape==(n,56) for role in ['speaker',*methods])
        mask=np.isfinite(features[uid,'speaker'][:-1]).all(1)
        for role in ['gt',*methods]:mask &= np.isfinite(features[uid,role][1:]).all(1)
        assert mask.sum()>=2
        masks[uid]=mask;coverage.append(dict(id=uid,candidate_frames=len(mask),scored_frames=int(mask.sum()),listener_indices=(np.flatnonzero(mask)+1).tolist()))
    gt=[features[u,'gt'][1:][masks[u]] for u in p['ids']];sp=[features[u,'speaker'][:-1][masks[u]] for u in p['ids']]
    def variance(clips):
        x=np.concatenate(clips);return {g:float(np.var(x[:,sl],axis=0,ddof=0).mean()) for g,sl in [('pose',slice(0,6)),('exp',slice(6,56))]}
    rows={}
    for name in methods:
        pred=[features[u,name][1:][masks[u]] for u in p['ids']]
        r=evaluate(gt,pred,sp);r['released_flatten_variance']=copy.deepcopy(r['metrics']['Var'])
        r['metrics']['Var']=variance(pred);r['metrics']['Var_GT']=variance(gt)
        r['aggregation']='DIM functions; per-coefficient temporal population variance averaged over coefficients'
        rows[name]=r
    check=parity(gt,[features[u,'avatarforcing_audio_only'][1:][masks[u]] for u in p['ids']],sp)
    save(OUT/'MOTION_RESULTS.json',dict(pairs=43,methods=rows,coverage=coverage,shared_scored_frames=sum(x['scored_frames'] for x in coverage),metric_parity=check))
    cols=[(m,g) for m in ['FD','P_FD','MSE','SID','Var','rPCC'] for g in ['exp','pose']]
    with (OUT/'motion_metrics.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['method','clips','frames',*[m+'_'+g for m,g in cols]])
        for name,r in rows.items():w.writerow([name,r['clips'],r['frames'],*[r['metrics'][m][g] for m,g in cols]])
    save(OUT/'SCORE_COMPLETE.json',dict(passed=True,methods=len(methods),records=len(records),source_sha256=sha(__file__),motion_sha256=sha(OUT/'MOTION_RESULTS.json')))
    print('Common EMOCA scoring complete',flush=True)

if __name__=='__main__':
    if sys.argv[1]=='prepare':prepare()
    else:score()
