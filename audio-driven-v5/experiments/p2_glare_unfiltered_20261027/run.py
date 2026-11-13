"""Durable no-quality-gate reaction processing, full paired scoring and publication."""
import concurrent.futures
import copy
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'experiments/p2_guided_full_eval_20261027')]
from generate_cluster import remote, PY
from p2_glare_eval.common import read, sha, digest, atomic_json, artifact_valid
from p2_glare_eval.media import aligned
from p2_glare_unfiltered.rules import policy, code_identity, process

SOURCE = ROOT / 'experiments/p2_glare_detector_fix_20261027'
MP = read(ROOT / 'configs/p2_omni_runtime.lock.json')['paths']['metrics_python']
ARMS = ['E0', 'E1', 'E0F', 'E1F']
SEEDS = [137, 251]
NODES = [0, 1]  # Both verified idle; host 2 currently closes SSH during handshake.
WORKERS = 8 * len(NODES)
ENV = {**os.environ, 'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '4'}


def state(phase, **kw):
    atomic_json(OUT / 'status.json', dict(phase=phase, time=time.time(), **kw))


def prepare():
    source = read(SOURCE / 'protocol.json')
    p = dict(schema='p2-glare-rgb-no-quality-gate-v1', policy=policy(), code=code_identity(),
             source_protocol_sha256=sha(SOURCE / 'protocol.json'),
             videos=source['videos'], pairs=source['pairs'], seeds=SEEDS,
             scoring_device='cuda:0', worker_count=WORKERS, node_ranks=NODES)
    atomic_json(OUT / 'protocol.json', p, immutable=True)
    (OUT / 'logs').mkdir(exist_ok=True)


def one_video(item):
    return dict(sha256=item['sha256'], **process(item['video'], SOURCE / 'detections' / item['sha256'],
                OUT / 'detections' / item['sha256'], item['sha256']))


def process_videos():
    p = read(OUT / 'protocol.json')
    assert p['code'] == code_identity() and p['policy'] == policy()
    assert p['source_protocol_sha256'] == sha(SOURCE / 'protocol.json')
    quality = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=24) as pool:
        for result in pool.map(one_video, p['videos']):
            quality.append(result)
            if len(quality) % 24 == 0:
                state('processing_reaction_rules', completed=len(quality), planned=len(p['videos']))
    atomic_json(OUT / 'video_quality.json', quality)
    assert len(quality) == 816
    assert sum(q.get('strict_scores_identical', False) for q in quality) == 643
    state('preparing_pairs', completed=816, unchanged_strict_valid=643)
    jobs = []
    coverage = {}
    by_hash = {q['sha256']: q for q in quality}
    for kind, conditions in p['pairs'].items():
        coverage[kind] = {}
        for condition, pairs in conditions.items():
            records = []
            counts = dict(planned_pairs=48, processed_pairs=0, excluded_pairs=0,
                          strict_valid_pairs=0, frames_per_side=0,
                          gt_fer_no_face_frames=0, generated_fer_no_face_frames=0,
                          gt_mediapipe_no_face_frames=0, generated_mediapipe_no_face_frames=0,
                          gt_fer_multiple_face_frames=0, generated_fer_multiple_face_frames=0)
            for row in pairs:
                record = dict(row)
                flags = []
                geometry = []
                for side in ['gt', 'generated']:
                    h = row['files'][row[side + '_video']]
                    folder = OUT / 'detections' / h
                    marker = read(folder / 'complete.json')
                    assert artifact_valid(folder, marker['contract'])
                    geometry.append(marker['geometry'])
                    record[side + '_csv'] = str(folder / 'scores.csv')
                    record[side + '_csv_sha256'] = sha(folder / 'scores.csv')
                    q = by_hash[h]
                    flags.append(q['strict_valid'])
                    for metric in ['fer_no_face_frames', 'mediapipe_no_face_frames', 'fer_multiple_face_frames']:
                        counts[side + '_' + metric] += q[metric]
                aligned(*geometry)
                record['geometry'] = geometry[0]
                records.append(record)
                counts['strict_valid_pairs'] += int(all(flags))
                counts['frames_per_side'] += geometry[0]['frame_count']
            assert len(records) == 48 and len({(r['sample_id'], r['sample_seed']) for r in records}) == 48
            counts['processed_pairs'] = len(records)
            coverage[kind][condition] = counts
            atomic_json(OUT / 'pairs' / kind / condition / 'records.json', records)
            for seed in SEEDS:
                selected = [r for r in records if r['sample_seed'] == seed]
                assert len(selected) == 24
                jobs.append(dict(key=f'{kind}-{condition}-{seed}', dataset=kind,
                                 condition=condition, seed=seed, records=selected,
                                 protocol_sha256=digest(p)))
    atomic_json(OUT / 'coverage.json', coverage)
    atomic_json(OUT / 'score_jobs.json', jobs, immutable=True)


def score(job):
    from p2_glare_eval.scoring import score_seed
    destination = OUT / 'scores' / (job['key'] + '.json')
    if destination.exists():
        assert read(destination)['job'] == job
        return
    for row in job['records']:
        for side in ['gt', 'generated']:
            assert sha(row[side + '_csv']) == row[side + '_csv_sha256']
            assert sha(row[side + '_video']) == row['files'][row[side + '_video']]
    values = score_seed(job['records'], 'cuda:0', 32)
    atomic_json(destination, dict(job=job, device='cuda:0', result=values))


def worker(rank):
    p = read(OUT / 'protocol.json')
    assert p['code'] == code_identity() and p['policy'] == policy()
    results = []
    for job in read(OUT / 'score_jobs.json')[rank::WORKERS]:
        score(job)
        results.append(job['key'])
        atomic_json(OUT / 'workers' / f'{rank:02d}.json', dict(finished=False, results=results))
    atomic_json(OUT / 'workers' / f'{rank:02d}.json', dict(finished=True, results=results))


def node(rank):
    children = []
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'], text=True)
    if any(int(line.split(',')[1]) > 1000 for line in gpu.strip().splitlines()):
        raise RuntimeError('GPU now occupied; do not displace another task')
    for local in range(8):
        i = rank * 8 + local
        with (OUT / 'logs' / f'score-worker-{i:02d}.log').open('a') as log:
            children.append(subprocess.Popen([MP, str(__file__), 'worker', str(i)],
                            env={**ENV, 'CUDA_VISIBLE_DEVICES': str(local)},
                            stdout=log, stderr=subprocess.STDOUT))
    codes = [child.wait() for child in children]
    atomic_json(OUT / f'node-{rank}.json', dict(finished=True, codes=codes))
    if any(codes):
        raise RuntimeError(f'Scoring worker errors: {codes}')


def publish():
    from p2_glare_eval.scoring import aggregate_seeds
    coverage = read(OUT / 'coverage.json')
    columns = ['FVD-32', 'UTMOSv2', 'LSE-D', 'LSE-C', 'R-F1', 'R-tIoU', 'R-ATD', 'R-FID']
    p = read(OUT / 'protocol.json')
    for kind in ['train', 'test']:
        base = ROOT / f'experiments/p2_guided_{kind}24_fourway_eval_20261027'
        result = copy.deepcopy(read(base / 'comparison_rgb_v2.json'))
        result.update(detector_version='rgb_no_quality_gate_v1', detector_policy=policy(),
                      quality_gates_applied=False, glare_pair_coverage=coverage[kind],
                      unfiltered_protocol=str(OUT / 'protocol.json'),
                      unfiltered_protocol_sha256=digest(p))
        for scope in [result, result['within_fixed'], result['within_guided']]:
            scope['coverage']['glare'] = 24
            scope.setdefault('details', {})['glare'] = dict(samples_per_seed=24, planned_samples=24, conditions={})
        for condition in coverage[kind]:
            arm, variant = condition.split('-')
            scores = {}
            for seed in SEEDS:
                v = read(OUT / 'scores' / f'{kind}-{condition}-{seed}.json')
                assert v['job']['protocol_sha256'] == digest(p)
                scores[str(seed)] = v['result']
            ids = [row['sample_id'] for row in v['job']['records']]
            details = dict(sample_ids=ids, per_seed=scores, aggregation=aggregate_seeds(scores))
            result['values'][condition].update(details['aggregation']['mean'])
            result['within_' + variant]['values'][arm].update(details['aggregation']['mean'])
            result['details']['glare']['conditions'][condition] = details
            result['within_' + variant]['details']['glare']['conditions'][arm] = details
        result['scope'] = 'All 24 planned samples and both seeds for GLARE, without detection quality rejection; AV metric scopes unchanged'
        result.pop('rgb_failure_analysis', None)
        atomic_json(base / 'comparison_unfiltered_v1.json', result)
        lines = ['# GLARE去额外门槛版（RGB）', '',
                 f'数据集：{kind}。八条件各24条×双seed=48配对；GLARE全部24条/seed进入评分。',
                 '取消全帧完整检测、唯一FER人脸及双检测器框IoU准入门槛。RGB修复、事件规则和指标公式保留。',
                 '多脸采用FER返回的第一张脸；漏检按原规则的零信号、detected mask和姿态插值处理。此为去门槛诊断版，不等同于人工真值或官方标准口径。',
                 'FVD/UTMOS/SyncNet完全沿用同范围严格版结果。R-ATD无匹配等数学上不适用项仍为N/A。', '']
        for title, section in [('八条件配对', result), ('固定prompt内部四组', result['within_fixed']), ('指导prompt内部四组', result['within_guided'])]:
            lines += ['## ' + title, '', str(section['coverage']), '',
                      '| 条件 | ' + ' | '.join(columns) + ' |', '|---|' + '---:|' * len(columns)]
            for condition, values in section['values'].items():
                lines.append('| ' + condition + ' | ' + ' | '.join('N/A' if values.get(c) is None else f'{values[c]:.4f}' for c in columns) + ' |')
        lines += ['', '## 检测诊断（均不用于排除样本）', '',
                  '| 条件 | 完成配对 | 原严格通过配对 | GT FER漏检帧 | 生成FER漏检帧 | GT MP漏检帧 | 生成MP漏检帧 | 每侧总帧 |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|']
        for condition, q in coverage[kind].items():
            fields = ['processed_pairs', 'strict_valid_pairs', 'gt_fer_no_face_frames', 'generated_fer_no_face_frames', 'gt_mediapipe_no_face_frames', 'generated_mediapipe_no_face_frames', 'frames_per_side']
            lines.append('| ' + condition + ' | ' + ' | '.join(str(q[f]) for f in fields) + ' |')
        (base / 'report_unfiltered_v1.md').write_text('\n'.join(lines) + '\n')


def main():
    prepare()
    state('processing_reaction_rules')
    process_videos()
    state('scoring', planned_seed_jobs=32)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        for output in pool.map(lambda r: remote(r, [PY, __file__, 'detach-node', r]), NODES):
            print(output, flush=True)
    while True:
        nodes = [OUT / f'node-{r}.json' for r in NODES]
        for n in nodes:
            if n.exists() and (read(n).get('error') or any(read(n).get('codes', []))):
                raise RuntimeError('Scoring node failed: ' + n.read_text())
        if all(n.exists() for n in nodes):
            break
        state('scoring', completed_seed_jobs=len(list((OUT / 'scores').glob('*.json'))), planned_seed_jobs=32)
        time.sleep(5)
    assert len(list((OUT / 'scores').glob('*.json'))) == 32
    state('publishing')
    publish()
    with (OUT / 'build_html.log').open('a') as log:
        subprocess.run([str(PY), str(ROOT / 'scripts/build_p2_av_review.py')], stdout=log, stderr=subprocess.STDOUT, check=True)
    state('complete', videos=816, pairs=768, seed_jobs=32, samples_per_seed=24)


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'run'
    if cmd in ['launch', 'detach-node']:
        target = ['run'] if cmd == 'launch' else ['node', sys.argv[2]]
        logpath = OUT / ('run.log' if cmd == 'launch' else f'node-{sys.argv[2]}.log')
        with logpath.open('a') as log:
            child = subprocess.Popen([str(PY), str(__file__), *target], env=ENV, stdout=log,
                                     stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                     start_new_session=True, close_fds=True)
        print(json.dumps(dict(pid=child.pid)))
    else:
        try:
            if cmd == 'run': main()
            elif cmd == 'worker': worker(int(sys.argv[2]))
            elif cmd == 'node': node(int(sys.argv[2]))
            elif cmd == 'publish': publish()
            else: raise ValueError(cmd)
        except Exception as error:
            if cmd == 'node': atomic_json(OUT / f'node-{sys.argv[2]}.json', dict(error=str(error)))
            elif cmd == 'run': state('failed', error=str(error))
            traceback.print_exc()
            raise
