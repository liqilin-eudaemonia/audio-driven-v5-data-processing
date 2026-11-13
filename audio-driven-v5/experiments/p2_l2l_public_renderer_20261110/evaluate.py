"""Use the sealed common EMOCA readout for public L2L+GPAvatar videos."""
import argparse
import copy
import csv
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
WORK=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'compare/deployment')]
import p2_l2l_gpavatar as gp
REFERENCE=ROOT/'experiments/p2_vico_literature_protocol_20261108'
PRIOR=ROOT/'experiments/p2_public_weights_vico_20261110'
OUT=WORK/'emoca_eval'
NEW=['l2l_gpavatar_'+n for n in gp.NAMES]
METHODS=['joint','joint_sft2752','dystream','avatarforcing_audio_only',*NEW]


def prepare():
    p=gp.verify();complete=gp.read(WORK/'RENDER_COMPLETE.json')
    assert complete['passed'] and complete['outputs']==172
    assert complete['protocol_sha256']==gp.sha(WORK/'protocol.json')
    prior=gp.read(PRIOR/'protocol.json');old=gp.read(PRIOR/'COMPLETE.json')
    for f,h in old['files'].items():assert gp.sha(f)==h,f
    ids=prior['ids'];byid={s['id']:s for s in p['samples']}
    assert set(ids)==set(byid) and len(ids)==43
    tasks=[]
    for name in gp.NAMES:
        for uid in ids:
            receipt=gp.read(WORK/'generated'/name/uid/'complete.json')
            assert receipt['frames']==byid[uid]['frames'] and not receipt['future_listener_input']
            assert receipt['protocol_sha256']==gp.sha(WORK/'protocol.json')
            video=Path(receipt['score_path'])
            for f,h in receipt['files'].items():assert gp.sha(f)==h,f
            tasks.append(dict(id=uid,role='l2l_gpavatar_'+name,video=str(video),video_sha256=gp.sha(video),expected_frames=receipt['frames']))
    reused=prior['reused_feature_records']+gp.read(PRIOR/'EXTRACTION_COMPLETE.json')['records']
    source=[Path(__file__),WORK/'image_metrics.py',ROOT/'compare/deployment/vico_literature_emoca.py',
        ROOT/'compare/deployment/vico_literature_metrics.py',ROOT/'scripts/p2_vico_official_metrics.py']
    protocol=dict(schema='public-l2l-gpavatar-emoca-v1',ids=ids,pairs=43,tasks=tasks,methods=METHODS,
        reused_feature_records=reused,source_files={str(f):gp.sha(f) for f in source},
        generator_protocol_sha256=gp.sha(WORK/'protocol.json'),prior_seal_sha256=gp.sha(PRIOR/'COMPLETE.json'),
        sampling='one previously generated seed137 public L2L output, all4 person weights; no GT selection',
        readout='All8 methods share EMOCA-v2 RGB readout, one-frame speaker offset, finite-frame intersection',
        variance='pooled per-coefficient temporal population variance, then coefficient mean',
        strict_published_table_reproduction=False)
    OUT.mkdir(exist_ok=True)
    destination=OUT/'protocol.json'
    if destination.exists():assert gp.read(destination)==protocol
    else:gp.save(destination,protocol)
    for name in ['assets','LICENSE_CONFIRMATION.json']:
        dest=OUT/name
        if not dest.exists():dest.symlink_to(REFERENCE/name,target_is_directory=name=='assets')
    # Original image scorer takes method-specific directory manifests.
    original=gp.read(ROOT/'experiments/p2_vico256_combined8000_20261106/protocol.json')
    preflight=ROOT/'experiments/p2_vico256_combined8000_20261106/INPUT_PREFLIGHT.json'
    for name in gp.NAMES:
        folder=WORK/'image_eval'/name;folder.mkdir(parents=True,exist_ok=True)
        (folder/'scoring').mkdir(exist_ok=True)
        method='l2l_gpavatar_'+name
        dest=folder/'scoring'/method
        if not dest.exists():dest.symlink_to(WORK/'scoring'/method,target_is_directory=True)
        manifest=dict(method=method,label='L2L public '+name+' + GPAvatar public renderer',
            datasets=original['datasets'],generator_protocol_sha256=gp.sha(WORK/'protocol.json'))
        if (folder/'protocol.json').exists():assert gp.read(folder/'protocol.json')==manifest
        else:gp.save(folder/'protocol.json',manifest)
        if not (folder/'INPUT_PREFLIGHT.json').exists():(folder/'INPUT_PREFLIGHT.json').symlink_to(preflight)
    print('Prepared172 EMOCA tasks and4 official image-metric manifests',flush=True)


def extract(smoke=False):
    import vico_literature_emoca as ex
    import vico_more_common as common
    p=gp.read(OUT/'protocol.json')
    for f,h in p['source_files'].items():assert gp.sha(f)==h
    ex.WORK=OUT
    native=common.gpu_guard
    def bounded(free_gib):
        device=native(free_gib)
        import torch
        torch.cuda.set_per_process_memory_fraction(.055,0)
        device['allocator_fraction']=.055
        return device
    common.gpu_guard=bounded;ex.run(smoke)


def score():
    import numpy as np
    from vico_literature_metrics import evaluate,parity
    p=gp.read(OUT/'protocol.json');ex=gp.read(OUT/'EXTRACTION_COMPLETE.json')
    assert ex['success'] and ex['tasks']==172
    features={};records=p['reused_feature_records']+ex['records']
    for r in records:
        assert gp.sha(r['path'])==r['sha256']
        key=r['id'],r['role'];assert key not in features
        with np.load(r['path']) as z:features[key]=z['motion'].copy()
        assert features[key].shape==(r['frames'],56)
    assert len(features)==430
    masks={};coverage=[]
    for uid in p['ids']:
        n=len(features[uid,'gt']);assert all(features[uid,role].shape==(n,56) for role in ['speaker',*METHODS])
        mask=np.isfinite(features[uid,'speaker'][:-1]).all(1)
        for role in ['gt',*METHODS]:mask &= np.isfinite(features[uid,role][1:]).all(1)
        assert mask.sum()>=2,uid
        masks[uid]=mask
        coverage.append(dict(id=uid,candidate_frames=len(mask),scored_frames=int(mask.sum()),
            listener_indices=(np.flatnonzero(mask)+1).tolist(),speaker_indices=np.flatnonzero(mask).tolist()))
    gt=[features[u,'gt'][1:][masks[u]] for u in p['ids']]
    sp=[features[u,'speaker'][:-1][masks[u]] for u in p['ids']]
    def variance(clips):
        a=np.concatenate(clips)
        return {g:float(np.var(a[:,sl],axis=0,ddof=0).mean()) for g,sl in [('pose',slice(0,6)),('exp',slice(6,56))]}
    gtvar=variance(gt);rows={};errors=[]
    for method in METHODS:
        pred=[features[u,method][1:][masks[u]] for u in p['ids']]
        r=evaluate(gt,pred,sp);r['released_flatten_variance']=copy.deepcopy(r['metrics']['Var'])
        r['metrics']['Var']=variance(pred);r['metrics']['Var_GT']=gtvar
        r['aggregation']='DIM public functions, with explicitly selected per-coefficient mean Var'
        for uid,g,a,clip in zip(p['ids'],gt,pred,r['per_clip']):
            clip.update(id=uid,scored_frames=len(g))
            for group,sl in [('pose',slice(0,6)),('exp',slice(6,56))]:
                errors.append(abs(float(np.square(g[:,sl].astype(np.float64)-a[:,sl]).mean())-clip['MSE'][group]))
        rows[method]=r
    check=parity(gt,[features[u,NEW[0]][1:][masks[u]] for u in p['ids']],sp)
    assert max(errors)<1e-5 and check['passed']
    result=dict(pairs=43,methods=rows,coverage=coverage,gt_variance=gtvar,
        shared_scored_frames=sum(x['scored_frames'] for x in coverage),
        public_renderer='GPAvatar ICLR2024',l2l_person_selection='all4, no GT selection',
        strict_published_table_reproduction=False,protocol_sha256=gp.sha(OUT/'protocol.json'))
    gp.save(OUT/'RESULTS.json',result)
    columns=[(m,g) for m in ['FD','P_FD','MSE','SID','Var','rPCC'] for g in ['exp','pose']]
    with (OUT/'metrics.csv').open('w') as f:
        writer=csv.writer(f);writer.writerow(['method','clips','frames',*[m+'_'+g for m,g in columns]])
        for name,row in rows.items():writer.writerow([name,row['clips'],row['frames'],*[row['metrics'][m][g] for m,g in columns]])
    lines=['# L2L＋GPAvatar：统一EMOCA实测','',
        'L2L四套公开人物权重＋GPAvatar公开渲染器，与原生动作成绩和原作者未发布的写实渲染流程分开标注。',
        f"43条，全部8方法共用{result['shared_scored_frames']}个有效评分帧。",'',
        '| 方法 | FD Exp ↓ | FD Pose ↓ | P-FD Exp ↓ | P-FD Pose ↓ | MSE Exp ↓ | MSE Pose ↓ |','|---|---:|---:|---:|---:|---:|---:|']
    for name,row in rows.items():
        lines.append('| '+name+' | '+' | '.join(f'{row["metrics"][m][g]:.6f}' for m in ['FD','P_FD','MSE'] for g in ['exp','pose'])+' |')
    lines += ['','其他SID、Var、rPCC及逐条覆盖见[CSV](metrics.csv)／[JSON](RESULTS.json)。',
        '本地共同读出比较；训练数据、原生输入和渲染器不同，不称原论文同协议复现。L2L额外使用speaker动作；单帧抠图和GPAvatar背景处理另见接入说明。','']
    (OUT/'RESULTS.md').write_text('\n'.join(lines))
    gp.save(OUT/'FINAL_AUDIT.json',dict(passed=True,arrays=430,methods=8,pairs=43,
        shared_mask=True,independent_mse_scalars=len(errors),max_abs=max(errors),metric_parity=check,
        generation_uses_only_first_listener_frame=True))
    gp.save(OUT/'COMPLETE.json',dict(passed=True,files={str(OUT/n):gp.sha(OUT/n) for n in ['RESULTS.json','RESULTS.md','metrics.csv','FINAL_AUDIT.json','protocol.json']}))
    print('EMOCA results complete',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','smoke','extract','score'])
    args=parser.parse_args()
    if args.stage=='prepare':prepare()
    elif args.stage in ['smoke','extract']:extract(args.stage=='smoke')
    else:score()
