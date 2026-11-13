"""Freeze literature evidence and real media before new EMOCA scores exist."""
import argparse
import csv
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
WORK=ROOT/'experiments/p2_vico_literature_protocol_20261108'
DIM=ROOT/'compare/Dyadic-Interaction-Modeling'
EMOCA=ROOT/'compare/deployment/vendor/emoca_v2'
META=Path('/public/0_DATA/2_Avatar/ViCo/dataset/metadata.csv')
OLD=ROOT/'experiments/p2_vico43_combined8000_sft2752_20261107'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n');tmp.replace(path)


def commit(path):return subprocess.check_output(['git','rev-parse','HEAD'],cwd=path,text=True).strip()


def evidence():
    definitions=[
        ('test scope','ViCo; INFP says same test set as DIM','DIM loader keys by audio, keeps5..1024 frames after shortest stream alignment','test43 UUIDs retained independently; no OOD; all source frames retained until declared score mask','Published effective IDs and intervals unavailable; released path may overwrite3 duplicated audio keys','DIM p9-11; INFP p7; dataset/data_loader.py; vico_preprocessing.py'),
        ('extractor','EMOCA expression50 + pose6','vico_preprocessing.py names EMOCA_v2_lr_mse_20','Exact named EMOCA checkpoint required; DECA cache never used as EMOCA','DIM did not publish full extractor config/crop/checkpoint hash','DIM p9-10 and supplement p23; code/vico_preprocessing.py'),
        ('ordering and scale','Raw EMOCA coefficients, Exp/Pose groups','pose6 followed by exp50; no display scaling','pose6+exp50, no z-score or rescaling','Need checkpoint and output verification before scoring','code/vico_preprocessing.py; code/mymetrics.py'),
        ('FD / P-FD / MSE','Fréchet; concatenated speaker/listener Fréchet; MSE','Per-clip computation and equal clip mean','Unmodified author functions and verified structured wrapper','RGB re-extraction differs from native model coefficients in original DIM baseline rows','DIM supplement A; code/mymetrics.py'),
        ('SID','Entropy of cluster-ID histogram within each sequence','GT-fitted KMeans40 expression /20 pose; dataset pooled histogram; log2','Match released code; record library versions and GT anchor','Per-sequence paper wording differs from global released computation; main-table exact aggregation unconfirmed','DIM supplement A; code/metrics/eval_utils.py'),
        ('Var','Time-series variance','np.var on all frames and coefficients flattened together','Use released pooled flattened formula; also report GT statistic','Not existing local mean temporal variance; do not compare old values','DIM supplement A; code/mymetrics.py'),
        ('rPCC / speaker','Residual correlation with real speaker motion','Pooled flattened correlation; default dataset substitutes all-one speaker','Real speaker extraction; constant inputs produce N/A','Default all-one route cannot yield finite paper rPCC; paper execution path unconfirmed','DIM supplement A; dataset/data_loader.py; code/mymetrics.py'),
        ('frame offset','Frame-by-frame synchrony','Tester compares listener t1..T-1 with speaker t0..T-2','Primary released-tester offset explicitly recorded; no GT-fitted lag','Published temporal convention not independently confirmed','code/x_engine_pt.py evaluate_test_epoch'),
        ('sampling','Main table sample policy not fully specified','10 candidates, per-clip minimum GT full56 Fréchet','Reuse one fixed seed137 video per method; never GT-select','Cannot claim same sampling budget; no retrospective best-of10','code/x_engine_pt.py evaluate_test_epoch; main paper leaves budget unspecified'),
        ('readout / conditioning','DIM baselines retrained on EMOCA; INFP/Avatar native transfers','Native coefficients for DIM scoring','Our output video re-extracted with EMOCA; speaker motion used only by scorer','Different readout/conditioning and training are disclosed; no claim of identical author implementation','DIM p9-11; INFP Section5.2.2; AvatarForcing Table4'),
    ]
    rows=[dict(item=a,paper=b,released_code=c,our_execution=d,remaining_difference=e,evidence=f) for a,b,c,d,e,f in definitions]
    save(WORK/'ALIGNMENT.json',dict(anchor='DIM Table1 -> INFP Table3 -> Avatar Forcing Table4',rows=rows,
        status='protocol frozen; exact published-table equivalence not established',
        literature_rows_may_be_cited=True,automatic_cross_source_ranking_allowed=False))
    lines=['# ViCo文献协议：原论文与本次执行对照','',
        '目标：以DIM Table1、INFP Table3、Avatar Forcing Table4为引用依据。官方公开代码不是已被证明的发表时完整执行记录。','',
        '| 项目 | 原论文 | 发布代码 | 本次执行 | 剩余差异 | 依据 |','|---|---|---|---|---|---|']
    for row in rows:lines.append('| '+' | '.join(str(row[k]).replace('|','/') for k in ['item','paper','released_code','our_execution','remaining_difference','evidence'])+' |')
    (WORK/'ALIGNMENT.md').write_text('\n'.join(lines)+'\n')


def literature():
    # Values transcribed from the exact local primary paper tables, never from
    # our old DECA measurements. Unreported cells remain null.
    keys=['FD','P_FD','MSE','SID','Var','rPCC']
    def row(method,source,values,origin,notes=''):
        metrics={k:{'exp':None,'pose':None} for k in keys}
        for k,pair in values.items():metrics[k]=dict(zip(['exp','pose'],pair))
        return dict(method=method,reported_in=source,original_result_source=origin,metrics=metrics,notes=notes)
    rows=[
        row('ELP reimplementation','DIM Table1',dict(FD=[47.17,.08],P_FD=[47.48,.08],MSE=[.98,.02],SID=[1.76,1.66],Var=[1.49,.02],rPCC=[.09,.01]),'DIM Table1','DIM author reimplementation; not original ELP table'),
        row('RLHG','DIM Table1 / INFP Table3 / AvatarForcing Table4',dict(FD=[39.02,.07],P_FD=[40.18,.07],MSE=[.86,.01],SID=[3.62,3.17],Var=[1.52,.02],rPCC=[.08,.02]),'DIM Table1'),
        row('L2L','DIM Table1 / INFP Table3 / AvatarForcing Table4',dict(FD=[33.93,.06],P_FD=[35.88,.06],MSE=[.93,.01],SID=[2.77,2.66],Var=[.83,.02],rPCC=[.06,.08]),'DIM Table1','ViCo retraining; not four original person-specific public checkpoints'),
        row('L2L + CANDOR','DIM Table1',dict(FD=[31.03,.06],P_FD=[33.02,.06],MSE=[.87,.01],SID=[3.53,3.05],Var=[.83,.02],rPCC=[.06,.09]),'DIM Table1'),
        row('DIM + CANDOR','DIM Table1 / INFP Table3 / AvatarForcing Table4',dict(FD=[23.88,.06],P_FD=[24.39,.06],MSE=[.70,.01],SID=[3.71,2.35],Var=[1.53,.02],rPCC=[.06,.03]),'DIM Table1'),
        row('INFP original','INFP Table3',dict(FD=[18.63,.07],MSE=[.51,.01],SID=[4.78,3.92],Var=[2.83,.18]),'INFP Table3','Not AvatarForcing FLOAT-space INFP*'),
        row('INFP* / AvatarForcing reimplementation','AvatarForcing Table4',dict(FD=[17.52,.07],SID=[2.19,3.20],Var=[2.10,.03],rPCC=[.01,.07]),'AvatarForcing Table4','FLOAT/SPECTRE motion-space reimplementation; not original INFP'),
        row('Avatar Forcing','AvatarForcing Table4',dict(FD=[16.64,.05],SID=[3.12,3.00],Var=[2.80,.03],rPCC=[.01,.01]),'AvatarForcing Table4','Native paper conditioning differs from our self-zero adaptation'),
        row('GT / DIM','DIM Table1',dict(SID=[5.03,4.07],Var=[.93,.01]),'DIM Table1','Rounded published anchor; not a scaling or extractor-tuning target'),
    ]
    save(WORK/'LITERATURE_ROWS.json',dict(rows=rows,provenance='Primary PDFs and explicit table footnotes',
        cell_provenance='P-FD/MSE fields absent from downstream tables remain attributed to DIM only',
        existing_local_adapted_results_used=False))
    with (WORK/'literature_rows.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['method','result_source','metric','group','reported_value','notes'])
        for r in rows:
            for metric,groups in r['metrics'].items():
                for g,v in groups.items():w.writerow([r['method'],r['original_result_source'],metric,g,'' if v is None else v,r['notes']])


def prepare():
    WORK.mkdir(exist_ok=True)
    if (WORK/'COMPLETE.json').exists():
        for path,h in read(WORK/'COMPLETE.json')['files'].items():assert sha(path)==h,path
        print('Sealed protocol and results verified; no files rewritten');return
    meta=list(csv.DictReader(META.open()));test=[r for r in meta if r['data_split']=='test']
    assert len(test)==43
    old=read(ROOT/'experiments/p2_vico_motion8000_20261106/protocol.json')
    byid={r['id']:r for r in old['samples']};assert set(byid)=={r['uuid'] for r in test}
    lookup={r['audio']:r for r in meta};counts=Counter(r['audio'] for r in test)
    tasks=[];manifest=[]
    for r in test:
        uid=r['uuid'];sample=byid[uid];frames=sample['streams']['gt']['frames']
        streams={k:Path(sample['streams'][k]['path']) for k in ['speaker','gt']}
        streams['joint']=Path(sample['streams']['prediction']['path'])
        streams['joint_sft2752']=OLD/'scoring/combined8000_sft2752'/(uid+'.listener.mp4')
        for role,path in streams.items():
            assert path.exists(),path
            tasks.append(dict(id=uid,role=role,video=str(path),video_sha256=sha(path),expected_frames=frames))
        mapped=lookup[r['audio']]
        manifest.append(dict(id=uid,audio_key=r['audio'],attitude=r['attitude'],frames=frames,
            source_split='test',contains_ood=False,audio_key_multiplicity=counts[r['audio']],
            released_mapping_listener=mapped['listener'],released_mapping_speaker=mapped['speaker'],
            released_mapping_changes_listener=mapped['listener']!=r['listener'],
            exceeds_loader_frame_limit_before_feature_audio_alignment=frames>1024,
            actual_author_retention='unpublished',author_audio_emoca_min_length=None))
    evidence();literature()
    files=[DIM/'code'/p for p in ['mymetrics.py','metrics/eval_utils.py','vico_preprocessing.py','dataset/data_loader.py','x_engine_pt.py','test_s2s_pretrain.py']]
    files += [ROOT/'writing'/p for p in ['17_DIM_ECCV2024.pdf','35_INFP_CVPR2025.pdf','43_Avatar_Forcing_CVPR2026.pdf']]
    files += [EMOCA/p for p in ['gdl_apps/EMOCA/demos/download_assets.sh','gdl/models/DECA.py','gdl/models/DecaEncoder.py','gdl/models/ResNet.py','gdl/datasets/FaceVideoDataModule.py']]
    protocol=dict(schema='vico-literature-emoca-alignment-v1',created_at=time.time(),
        target='DIM Table1, INFP Table3, AvatarForcing Table4',
        dim_commit=commit(DIM),emoca_commit=commit(EMOCA),source_files={str(p):sha(p) for p in files},
        metadata_sha256=sha(META),ids=[r['uuid'] for r in test],tasks=tasks,extractor='EMOCA_v2_lr_mse_20',
        coefficient_order='pose6 then exp50; raw; no normalization',
        selected_methods=['joint','joint_sft2752'],selection='fixed before new EMOCA test scores exist',
        sampling='one previously sealed seed137 output; no GT selection',
        score_alignment='listener frames1:T and speaker frames0:T-1, following released tester offset',
        alternate_scopes='Released loader projection is diagnostic only; unknown author effective lengths remain unknown',
        includes_ood=False,pairs=43,readout='EMOCA from RGB for ours and real speaker/GT',
        resource=dict(node='node11',gpu=7,cpu_threads=2),
        original_complete_sha256=sha(OLD/'COMPLETE.json'),strict_published_protocol_reproduction=False)
    dest=WORK/'protocol.json'
    if dest.exists():
        prior=read(dest);protocol['created_at']=prior['created_at'];assert protocol==prior,'Frozen alignment changed'
    else:save(dest,protocol)
    save(WORK/'TEST_MANIFEST.json',dict(pairs=manifest,exact43=True,ood=0,
        duplicate_audio_keys={k:v for k,v in counts.items() if v>1},
        long_clips=[r['id'] for r in manifest if r['exceeds_loader_frame_limit_before_feature_audio_alignment']],
        author_effective_test_match_confirmed=False))
    with (WORK/'test_manifest.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(manifest[0]));w.writeheader();w.writerows(manifest)
    print('Frozen43 IDs,172 extraction tasks, primary literature rows and10 alignment dimensions')


if __name__=='__main__':prepare()
