"""Re-score completed public-checkpoint videos with the sealed DIM/EMOCA path.

Prior generation runs and EMOCA results remain read-only. This run uses a new
common finite-frame intersection for every row, including our reference models.
"""
import argparse
import copy
import csv
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'compare/deployment'))
from vico_literature_protocol import read, save, sha

WORK = ROOT / 'experiments/p2_public_weights_vico_20261110'
REFERENCE = ROOT / 'experiments/p2_vico_literature_protocol_20261108'
DY = ROOT / 'experiments/p2_compare_vico43_20261106/dystream'
AV = ROOT / 'experiments/p2_vico43_unified_20261107/avatarforcing'
METHODS = ['joint', 'joint_sft2752', 'dystream', 'avatarforcing_audio_only']
LABELS = dict(joint='Ours / joint8000', joint_sft2752='Ours / joint8000 + SFT2752',
              dystream='DyStream / public weights / listener adaptation',
              avatarforcing_audio_only='Avatar Forcing / public weights / zero speaker visual')


def prepare():
    WORK.mkdir(parents=True, exist_ok=True)
    reference = read(REFERENCE / 'protocol.json')
    ids = reference['ids']
    assert len(set(ids)) == len(ids) == 43
    tasks, source_receipts, protocols = [], {}, {}
    by_id = {s['id']:s for s in read(DY / 'protocol.json')['samples']}
    for role, folder in [('dystream', DY), ('avatarforcing_audio_only', AV)]:
        protocol = read(folder / 'protocol.json')
        assert set(s['id'] for s in protocol['samples']) == set(ids)
        assert protocol['seed'] == 137 if role == 'dystream' else protocol['parameters']['seed'] == 137
        assets = protocol['weights'] if role == 'dystream' else protocol['assets']
        for path, checksum in assets.items():
            assert sha(path) == checksum, path
        protocols[str(folder / 'protocol.json')] = sha(folder / 'protocol.json')
        for uid in ids:
            receipt = folder / 'generated' / uid / 'complete.json'
            r = read(receipt)
            assert r['id'] == uid and r['source_only']
            assert r['protocol_sha256'] == sha(folder / 'protocol.json')
            assert r['score_frames'] == by_id[uid]['frames']
            video = (folder / 'scoring/dystream' / (uid + '.listener.mp4')
                     if role == 'dystream' else Path(r['score_path']))
            assert sha(video) == r['score_sha256']
            assert sha(folder / 'generated' / uid / 'generated.mp4') == r['native_sha256']
            assert sha(by_id[uid]['image']) == by_id[uid]['image_sha256']
            source_receipts[str(receipt)] = sha(receipt)
            tasks.append(dict(id=uid, role=role, video=str(video),
                              video_sha256=sha(video), expected_frames=r['score_frames']))
    frozen_files = [Path(__file__), ROOT / 'compare/deployment/vico_literature_emoca.py',
                    ROOT / 'compare/deployment/vico_literature_metrics.py']
    reused = read(REFERENCE / 'EXTRACTION_COMPLETE.json')['records']
    assert len(reused) == 172
    for r in reused:
        assert sha(r['path']) == r['sha256']
    protocol = dict(schema='public-checkpoint-vico43-emoca-v1', ids=ids, pairs=43,
        tasks=tasks, source_files={str(p):sha(p) for p in frozen_files},
        source_generation_protocols=protocols, source_receipts=source_receipts,
        reference_protocol_sha256=sha(REFERENCE / 'protocol.json'),
        reference_results_sha256=sha(REFERENCE / 'RESULTS.json'),
        reference_complete_sha256=sha(REFERENCE / 'COMPLETE.json'),
        reused_feature_records=reused, methods=METHODS, labels=LABELS,
        seed=137, candidates_per_clip=1, gt_selection=False,
        extractor='EMOCA_v2_lr_mse_20; original FAN2D crop, threshold .9, scale 1.25',
        temporal_alignment='listener 1:T vs speaker 0:T-1',
        frame_mask='intersection of all four models, GT and real speaker; no interpolation',
        variance='mean_j Var_t(x[t,j]), pooled scored frames, ddof=0; native flatten also retained',
        public_weight_training='No training updates; no new 500h training or caption conditioning',
        conditions=dict(dystream='listener reference + real speaker audio; listener audio zero',
            avatarforcing_audio_only='same reference and audio; listener audio zero; speaker visual latent zero'),
        strict_published_table_reproduction=False,
        resource=dict(node='node11', physical_gpu=7, allocator_fraction=.055, minimum_free_mib=32768))
    dest = WORK / 'protocol.json'
    if dest.exists():
        assert read(dest) == protocol, 'Frozen protocol changed; use a new directory'
    else:
        save(dest, protocol)
    for name in ['assets', 'LICENSE_CONFIRMATION.json']:
        target = WORK / name
        if not target.exists():
            target.symlink_to(REFERENCE / name, target_is_directory=name == 'assets')
        assert target.resolve() == (REFERENCE / name).resolve()
    save(WORK / 'INPUT_AUDIT.json', dict(passed=True, clips_per_method=43,
         public_video_outputs_verified=86, reference_feature_streams_verified=172,
         weights_verified=True, no_training_updates=True, captions_not_model_inputs=True))
    print('Verified 86 public-weight outputs and 172 reusable reference feature streams', flush=True)


def verify_sources():
    p = read(WORK / 'protocol.json')
    for path, checksum in {**p['source_files'], **p['source_generation_protocols'], **p['source_receipts']}.items():
        assert sha(path) == checksum, path
    assert sha(REFERENCE / 'COMPLETE.json') == p['reference_complete_sha256']
    assert sha(REFERENCE / 'RESULTS.json') == p['reference_results_sha256']
    return p


def extract(smoke=False):
    verify_sources()
    import vico_literature_emoca as extractor
    import vico_more_common as common
    extractor.WORK = WORK
    original = common.gpu_guard

    def bounded_guard(free_gib):
        device = original(free_gib)
        import torch
        torch.cuda.set_per_process_memory_fraction(.055, 0)
        device['allocator_fraction'] = .055
        return device

    common.gpu_guard = bounded_guard
    extractor.run(smoke=smoke)


def pooled_variance(clips):
    import numpy as np
    a = np.concatenate(clips)
    return {name:float(np.var(a[:,sl], axis=0, ddof=0).mean())
            for name, sl in [('pose', slice(0,6)), ('exp', slice(6,56))]}


def score():
    import numpy as np
    from vico_literature_metrics import evaluate, parity
    p = verify_sources()
    ex = read(WORK / 'EXTRACTION_COMPLETE.json')
    assert ex['success'] and ex['tasks'] == 86
    for name in ['ENCODER_PARITY.json', 'MEDIA_PARITY.json']:
        assert read(WORK / name)['passed']
    assert read(WORK / 'ENCODER.json')['checkpoint_sha256'] == read(REFERENCE / 'ENCODER.json')['checkpoint_sha256']
    features, records = {}, p['reused_feature_records'] + ex['records']
    for r in records:
        assert sha(r['path']) == r['sha256']
        key = r['id'], r['role']
        assert key not in features
        with np.load(r['path']) as z:
            features[key] = z['motion'].copy()
            assert features[key].shape == (r['frames'], 56)
            assert np.array_equal(z['valid'], np.isfinite(features[key]).all(1))
    masks, coverage = {}, []
    for uid in p['ids']:
        size = len(features[uid,'gt'])
        assert all(features[uid,role].shape == (size,56) for role in ['speaker', *METHODS])
        mask = np.isfinite(features[uid,'speaker'][:-1]).all(1)
        for role in ['gt', *METHODS]:
            mask &= np.isfinite(features[uid,role][1:]).all(1)
        assert mask.sum() >= 2, uid
        masks[uid] = mask
        coverage.append(dict(id=uid, candidate_frames=len(mask), scored_frames=int(mask.sum()),
            listener_indices=(np.flatnonzero(mask)+1).tolist(), speaker_indices=np.flatnonzero(mask).tolist()))
    gt = [features[u,'gt'][1:][masks[u]] for u in p['ids']]
    sp = [features[u,'speaker'][:-1][masks[u]] for u in p['ids']]
    results, independent_errors = {}, []
    gt_var = pooled_variance(gt)
    for method in METHODS:
        pred = [features[u,method][1:][masks[u]] for u in p['ids']]
        row = evaluate(gt,pred,sp)
        row['released_flatten_variance'] = copy.deepcopy(row['metrics']['Var'])
        row['metrics']['Var'] = pooled_variance(pred)
        row['metrics']['Var_GT'] = gt_var
        row['aggregation'] = p['variance'] + '; other metrics use pinned released DIM functions'
        row['label'] = LABELS[method]
        row['weights'] = 'our frozen trained checkpoint' if method.startswith('joint') else 'official public pretrained; no retraining'
        for uid, g, a, r in zip(p['ids'],gt,pred,row['per_clip']):
            r.update(id=uid,scored_frames=len(g))
            for group, sl in [('pose',slice(0,6)),('exp',slice(6,56))]:
                independent_errors.append(abs(float(np.square(g[:,sl].astype(np.float64)-a[:,sl]).mean())-r['MSE'][group]))
        results[method] = row
    audit = parity(gt,[features[u,'dystream'][1:][masks[u]] for u in p['ids']],sp)
    save(WORK / 'METRIC_PARITY.json', audit)
    selfcheck = evaluate(gt,gt,sp)
    assert all(abs(v) < 1e-5 for m in ['FD','P_FD','MSE','rPCC'] for v in selfcheck['metrics'][m].values() if v is not None)
    result = dict(schema='public-weights-vico43-results-v1', methods=results, pairs=43,
        coverage=coverage, shared_scored_frames=sum(c['scored_frames'] for c in coverage),
        candidate_frames=sum(c['candidate_frames'] for c in coverage),
        gt_variance=gt_var, protocol_sha256=sha(WORK / 'protocol.json'),
        sampling=p['seed'], candidates_per_clip=1, gt_based_selection=False,
        strict_published_table_reproduction=False,
        note='Common RGB-to-EMOCA readout; native model conditions and pretraining differ; quoted paper rows are separate')
    save(WORK / 'RESULTS.json', result)
    columns = [(m,g) for m in ['FD','P_FD','MSE','SID','Var','rPCC'] for g in ['exp','pose']]
    with (WORK / 'metrics.csv').open('w') as f:
        writer = csv.writer(f)
        writer.writerow(['method','weights','clips','frames',*[m+'_'+g for m,g in columns]])
        for method,row in results.items():
            writer.writerow([row['label'],row['weights'],row['clips'],row['frames'],*[row['metrics'][m][g] for m,g in columns]])
    lines = ['# 公开权重：ViCo test43 统一 EMOCA 实测', '',
        'DyStream 和 Avatar Forcing 使用官方公开预训练权重，没有用本轮500h重训。两者均采用listener音轨全零的适配；本表Avatar Forcing的speaker视觉条件为零。真实speaker视频条件的新版另见独立流程。', '',
        f"固定43条，每条seed137一次采样；四个模型共用{result['shared_scored_frames']}个有效评分帧，候选{result['candidate_frames']}帧。", '',
        '| 方法 | FD Exp ↓ | FD Pose ↓ | P-FD Exp ↓ | P-FD Pose ↓ | MSE Exp ↓ | MSE Pose ↓ |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for row in results.values():
        vals = [row['metrics'][m][g] for m in ['FD','P_FD','MSE'] for g in ['exp','pose']]
        lines.append('| '+row['label']+' | '+' | '.join(f'{v:.6f}' for v in vals)+' |')
    lines += ['', '| 方法 | SID Exp | SID Pose | Var Exp | Var Pose | rPCC Exp ↓ | rPCC Pose ↓ |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for row in results.values():
        vals = [row['metrics'][m][g] for m in ['SID','Var','rPCC'] for g in ['exp','pose']]
        lines.append('| '+row['label']+' | '+' | '.join('N/A' if v is None else f'{v:.6f}' for v in vals)+' |')
    lines += ['',f'GT逐系数方差：Exp {gt_var["exp"]:.6f}，Pose {gt_var["pose"]:.6f}。', '',
        'FD/P-FD/MSE逐片段等权；SID沿用DIM的GT拟合K40/20；rPCC沿用其拉平相关差；Var采用用户确认的逐系数跨帧方差再平均，原发布代码拉平方差留在JSON。',
        '这些是本地共同读出实验，不是原论文表格复现；训练集、输入条件和renderer不同。引用文献数值应另列来源，不据此声称严格同协议领先。', '',
        '[全部指标CSV](metrics.csv) · [原始JSON与逐条覆盖](RESULTS.json) · [验证](FINAL_AUDIT.json) · [入口](README.md)', '']
    (WORK / 'RESULTS.md').write_text('\n'.join(lines))
    assert max(independent_errors) < 1e-5
    save(WORK / 'FINAL_AUDIT.json',dict(passed=True,at=time.time(),pairs=43,
        public_feature_arrays=86,reused_feature_arrays=172,all_methods_same_mask=True,
        independent_float64_mse_scalars=len(independent_errors),max_abs=max(independent_errors),
        released_metric_parity=audit['passed'],gt_selfcheck=selfcheck['metrics'],
        old_reference_results_unchanged=True,results_sha256=sha(WORK / 'RESULTS.json')))
    save(WORK / 'COMPLETE.json',dict(passed=True,files={str(WORK/n):sha(WORK/n) for n in
        ['protocol.json','RESULTS.json','RESULTS.md','metrics.csv','FINAL_AUDIT.json','METRIC_PARITY.json']}))
    save(WORK / 'status.json',dict(stage='complete',pairs=43,methods=4,at=time.time()))
    print('Complete:', WORK / 'RESULTS.md', flush=True)


def run():
    WORK.mkdir(parents=True, exist_ok=True)
    with (WORK / 'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        python = ROOT / 'compare/.envs/vico_more/bin/python'
        for action,marker in [('smoke','MEDIA_PARITY.json'),('extract','EXTRACTION_COMPLETE.json'),('score','COMPLETE.json')]:
            if (WORK / marker).exists():
                continue
            env = dict(os.environ, CUDA_VISIBLE_DEVICES='' if action == 'score' else '7',
                OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1')
            with (WORK / (action+'.log')).open('ab') as log:
                proc = subprocess.Popen([str(python),str(Path(__file__)),action],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                save(WORK / 'ACTIVE.json',dict(stage=action,pid=proc.pid,at=time.time()))
                code = proc.wait()
            save(WORK / (action+'_exit.json'),dict(returncode=code,at=time.time()))
            if code or not (WORK / marker).exists():
                save(WORK / 'status.json',dict(stage='failed',action=action,returncode=code))
                raise RuntimeError(action+' failed; see '+str(WORK/(action+'.log')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action',choices=['prepare','smoke','extract','score','run'])
    args = parser.parse_args()
    if args.action == 'prepare':prepare()
    elif args.action in ['smoke','extract']:extract(smoke=args.action == 'smoke')
    elif args.action == 'score':score()
    else:run()
