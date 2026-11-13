"""Predeclare statistics for the official SPECTRE frontend; do not fit target cells."""
import argparse
from pathlib import Path
import sys

WORK = Path(__file__).resolve().parent
ROOT = WORK.parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'compare/deployment')]
from p2_avatarforcing_table4_generate import read, save, sha


def prepare():
    p = dict(schema='af-official-spectre-metrics-reconstruction-v1', implementation_sha256=sha(__file__),
        shared_kernels_sha256=sha(ROOT / 'scripts/p2_avatarforcing_table4_score.py'),
        primary='retain earlier AF reconstruction statistics for comparability, not an official scoring entry',
        metrics=dict(FD='standard Frechet; per-clip equal mean',
                     rPCC='absolute difference of global flattened speaker/GT and speaker/pred correlations',
                     SID='K15exp/K9pose; testGT-only fitted centers; entropy bits on pooled prediction histogram',
                     Var='pooled temporal variance per coefficient, mean across coefficients'),
        predeclared_diagnostics=['global FD', 'paper unsquared mean FD term',
            'equal-clip entropy with the same diagnostic quantizer', 'equal-clip temporal per-feature Var',
            'coefficientwise and flattened rPCC per clip', 'global flattened Var', 'DIM cited implementation SID K40/20'],
        known_gaps=['AF did not publish aggregation settings or fitted K-means centers',
            'L2L cited metric fits clusters to training motion; testGT quantizer is an explicitly unverified proxy',
            'AF D.5 states feature-mean temporal Var; clip/global pooling is not specified',
            'No diagnostic is selected based on closeness to target Table4 cells'],
        alignment_claim=False, frozen_before_full_native_scoring=True)
    path = WORK / 'OFFICIAL_METRICS_PROTOCOL.json'
    if path.exists():
        assert read(path) == p
    else:
        save(path, p)
    return p


def score():
    import numpy as np
    from p2_avatarforcing_table4_score import metrics, coefficient_correlations
    from vico_literature_metrics import load_author
    p = prepare()
    complete = read(WORK / 'OFFICIAL_EXTRACTION_COMPLETE.json')
    assert complete['passed'] and complete['streams'] == 129
    generation = read(WORK / 'NATIVE_GENERATION_PROTOCOL.json')
    ids = [s['id'] for s in generation['samples']]
    features, audits = {}, {}
    for r in complete['records']:
        assert sha(r['path']) == r['sha256'] and sha(r['video']) == r['video_sha256']
        with np.load(r['path']) as data:
            motion = data['motion'].copy()
            assert float(data['fps']) == 25
            assert np.array_equal(data['frame_indices'], np.arange(r['frames']))
        assert motion.shape == (r['frames'], 56) and np.isfinite(motion).all()
        features[r['id'], r['role']] = motion
        audits[r['id'], r['role']] = r['face_audit']
    gt = [features[u, 'gt'] for u in ids]
    pred = [features[u, 'avatarforcing'] for u in ids]
    speaker = [features[u, 'speaker'] for u in ids]
    for g, a, x in zip(gt, pred, speaker):
        assert g.shape == a.shape == x.shape
    result = metrics(gt, pred, speaker)
    diagnostic = {}
    author = load_author()
    for group, sl in [('exp', slice(6, 56)), ('pose', slice(0, 6))]:
        per_clip = []
        for uid, g, a, x in zip(ids, gt, pred, speaker):
            cg, ca = coefficient_correlations(g[:, sl], x[:, sl]), coefficient_correlations(a[:, sl], x[:, sl])
            valid = np.isfinite(cg) & np.isfinite(ca)
            r1, r2 = np.corrcoef(g[:, sl].reshape(-1), x[:, sl].reshape(-1))[0, 1], np.corrcoef(a[:, sl].reshape(-1), x[:, sl].reshape(-1))[0, 1]
            per_clip.append(dict(id=uid,
                rPCC_per_coefficient=float(np.mean(np.abs(cg[valid]-ca[valid]))) if valid.any() else None,
                valid_coefficients=int(valid.sum()),
                rPCC_flattened=float(abs(r1-r2)) if np.isfinite(r1) and np.isfinite(r2) else None,
                temporal_feature_mean_Var=float(np.var(a[:, sl], axis=0).mean())))
        pooled = np.concatenate(pred)[:, sl]
        diagnostic[group] = dict(per_clip=per_clip,
            SID_DIM_implementation_K40_20=float(author.calcuate_sid(gt, pred, type=group)),
            rPCC_equal_clip_coefficient_mean=float(np.mean([r['rPCC_per_coefficient'] for r in per_clip if r['rPCC_per_coefficient'] is not None])),
            rPCC_equal_clip_flattened=float(np.mean([r['rPCC_flattened'] for r in per_clip if r['rPCC_flattened'] is not None])),
            Var_global_flattened=float(np.var(pooled)),
            Var_equal_clip_temporal_feature_mean=float(np.mean([r['temporal_feature_mean_Var'] for r in per_clip])))
    comparison = []
    for name, values in result['primary'].items():
        for group, value in values.items():
            target = generation['target'][name][group]
            comparison.append(dict(metric=name, group=group, measured=value, published=target,
                delta=value-target if value is not None else None,
                same_at_2dp=value is not None and f'{value:.2f}' == f'{target:.2f}'))
    matches = sum(c['same_at_2dp'] for c in comparison)
    result.update(schema='af-official-spectre-original-view-results-v1', pairs=43, frames=sum(map(len, gt)),
        comparison=comparison, matched_cells_at_published_precision=matches, total_cells=8,
        numerical_table_match=matches == 8, strict_private_protocol_verified=False,
        paper_diagnostics=diagnostic, known_gaps=p['known_gaps'],
        metrics_protocol_sha256=sha(WORK / 'OFFICIAL_METRICS_PROTOCOL.json'),
        generation_protocol_sha256=sha(WORK / 'NATIVE_GENERATION_PROTOCOL.json'),
        extraction_protocol_sha256=sha(WORK / 'OFFICIAL_SPECTRE_PROTOCOL.json'),
        coverage={role:dict(clips=43, frames=sum(len(features[u, role]) for u in ids),
            missing_landmark_frames=sum(len(audits[u, role]['missing_landmark_frames']) for u in ids),
            multi_face_frames=sum(audits[u, role]['multi_face_frames'] for u in ids))
            for role in ['gt', 'speaker', 'avatarforcing']})
    save(WORK / 'OFFICIAL_RESULTS.json', result)
    lines = ['# Avatar Forcing原始视野／原版环境／SPECTRE官方跟踪结果', '',
        f'43条、{result["frames"]}帧；主重建口径8项中{matches}项与论文两位小数相同。', '',
        '| 指标 | 分组 | 实测 | Table4 | 差值 |', '|---|---|---:|---:|---:|']
    for c in comparison:
        measured = 'N/A' if c['measured'] is None else f"{c['measured']:.6f}"
        delta = 'N/A' if c['delta'] is None else f"{c['delta']:+.6f}"
        lines.append(f"| {c['metric']} | {c['group']} | {measured} | {c['published']:.2f} | {delta} |")
    lines += ['', '这仍是公开材料重建，不是AF官方评分脚本。原始ViCo视野、原版Torch2.0.1、官方默认初始化seed20、真实speaker视觉和官方导出和SPECTRE官方RetinaFace／FAN2／JPEG提取路径已接入；静音自身音频、完整43条／完整区间及统计汇总仍有明确未披露项。', '',
              '主统计沿用上一轮以便比较；论文文字对应的时序方差／相关汇总诊断和其他预先声明的变体全部保存在JSON，没有择优换主分数。SID量化器暂以GT拟合，不能冒称已获得论文的训练集聚类中心。', '',
              '本轮同时改变了视野、运行环境、初始化随机流和导出路径，不能把新旧差值单独归因于其中一项。目标是否完成仍需全部结果和协议独立验收。', '']
    (WORK / 'OFFICIAL_RESULTS.md').write_text('\n'.join(lines))
    save(WORK / 'OFFICIAL_EXECUTION_COMPLETE.json', dict(passed_execution=True,
        numerical_table_match=matches == 8, matched_cells=matches,
        strict_private_protocol_verified=False, results_sha256=sha(WORK / 'OFFICIAL_RESULTS.json')))
    print('Native scoring complete:', matches, '/8 cells at2dp', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'score'])
    args = parser.parse_args()
    prepare() if args.action == 'prepare' else score()
