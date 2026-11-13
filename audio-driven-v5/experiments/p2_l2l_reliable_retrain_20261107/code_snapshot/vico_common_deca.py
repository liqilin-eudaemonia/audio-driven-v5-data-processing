"""Shared DECA motion evaluation of frozen ViCo videos and native L2L outputs.

Existing generation and evaluation artifacts are read-only inputs. GPU work is
restricted to the newly authorized node11 physical GPU7. No model is generated
or selected using listener ground truth.
"""
import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / 'experiments/common_method_comparison_20261106/vico_deca'
BASE = ROOT / 'experiments/p2_vico_motion8000_20261106'
COMPARE = ROOT / 'experiments/p2_compare_vico43_20261106'
L2L = COMPARE / 'learning2listen'
VENDOR = ROOT / 'compare/deployment/vendor/DECA'
WEIGHT = ROOT / 'compare/deployment/assets/deca_model.tar'
GPU_UUID = 'GPU-4d5ee808-e829-c824-c506-cad80ff133f5'
GROUPS = {'exp': slice(0, 50), 'head_rotation': slice(50, 53), 'jaw_rotation': slice(53, 56)}
VARIANTS = ['conan', 'fallon', 'stephen', 'trevor']


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(p, data):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + f'.tmp.{os.getpid()}')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(p)


def prepare():
    old = read(L2L / 'DECA_PROTOCOL.json')
    for path, expected in old['source'].items():
        assert sha(path) == expected, f'DECA source changed: {path}'
    original = read(BASE / 'protocol.json')
    complete = read(L2L / 'DECA_COMPLETE.json')
    receipts = {(r['id'], r['role']): r for r in complete['records']}
    tasks, reused, native = [], [], []
    for sample in original['samples']:
        uid = sample['id']
        for role in ['speaker', 'gt']:
            path = L2L / 'features' / role / (uid + '.npz')
            receipt = receipts[uid, role]
            assert sha(path) == receipt['feature_sha256']
            assert sha(sample['streams'][role]['path']) == receipt['source_sha256']
            assert receipt['deca_protocol_sha256'] == sha(L2L / 'DECA_PROTOCOL.json')
            reused.append(dict(id=uid, role=role, path=str(path), sha256=sha(path), frames=receipt['frames']))
        for method in ['joint', 'dystream']:
            source = Path(sample['streams']['prediction']['path']) if method == 'joint' else COMPARE / 'dystream/scoring/dystream' / (uid + '.listener.mp4')
            landmarks = BASE / 'features/prediction' / (uid + '.npz') if method == 'joint' else COMPARE / 'dystream/motion/features/prediction' / (uid + '.npz')
            landmark_receipt = read(landmarks.with_suffix('.json'))
            assert sha(source) == landmark_receipt['input_sha256']
            assert sha(landmarks) == landmark_receipt['feature_sha256']
            tasks.append(dict(id=uid, method=method, video=str(source), video_sha256=sha(source),
                              landmarks=str(landmarks), landmarks_sha256=sha(landmarks),
                              frames=sample['streams']['gt']['frames']))
        for variant in VARIANTS:
            path = L2L / 'generated' / variant / (uid + '.npz')
            assert sha(path) == read(path.with_suffix('.json'))['feature_sha256']
            native.append(dict(id=uid, variant=variant, path=str(path), sha256=sha(path)))
    assert len(tasks) == len(reused) == 86 and len(native) == 172
    protocol = dict(schema='vico43-common-deca-v1', samples=original['samples'], extraction_tasks=tasks,
        reused=reused, native_l2l=native, extractor_protocol=str(L2L / 'DECA_PROTOCOL.json'),
        extractor_protocol_sha256=sha(L2L / 'DECA_PROTOCOL.json'),
        assets=old['source'], code_sha256=sha(__file__),
        metric_code_sha256=sha(ROOT / 'scripts/p2_vico_motion_metrics.py'),
        fps=30, frames_per_stream=sum(s['streams']['gt']['frames'] for s in original['samples']),
        space='DECA raw expression50 + global head axis-angle3 + jaw axis-angle3; no translation',
        normalization='none; raw coefficients, rotation radians',
        crop=old['crop'], precision='float32 tensors, no autocast; cuDNN TF32 enabled to match original DECA cache; matmul TF32 disabled; exact GT parity checked',
        parameters=dict(max_lag=60, window=120, stride=30),
        aggregation='equal clips; equal complete windows within each clip; raw values with no display scaling',
        short_clips='RTLCC requires T>61 (42 clips); RWTLCC requires T>=120 (41 clips); no metric padding',
        outputs=dict(joint='video re-extraction', dystream='video re-extraction', learning2listen='native motion coefficients'),
        limitations=['Common coefficients do not make training datasets or conditioning identical.',
                     'Video re-extraction and native coefficient outputs have different readout paths.',
                     'L2L person-specific checkpoints are transferred to ViCo, not retrained ViCo baselines.',
                     'Metrics implement explicit local definitions, not a verified DIM/CustomListener table reproduction.'],
        sampling='reuse frozen seed137 results; no GT-based model or sample selection',
        resource=dict(host='172.29.189.119', gpu=7, uuid=GPU_UUID, authorization='explicit user approval this turn'))
    if (WORK / 'protocol.json').exists():
        assert read(WORK / 'protocol.json') == protocol, 'Frozen protocol changed; use a new output directory'
    else:
        save(WORK / 'protocol.json', protocol)
    return protocol


def crop_frame(pair):
    from skimage.transform import estimate_transform, warp
    im, lm = pair
    left, top = lm.min(0)
    right, bottom = lm.max(0)
    old_size = (right - left + bottom - top) / 2 * 1.1
    center = np.array([right - (right-left)/2, bottom - (bottom-top)/2])
    size = int(old_size * 1.25)
    if size <= 0:
        raise ValueError('Degenerate DECA face crop')
    src = np.array([[center[0]-size/2, center[1]-size/2], [center[0]-size/2, center[1]+size/2], [center[0]+size/2, center[1]-size/2]])
    dst = np.array([[0, 0], [0, 223], [223, 0]])
    transform = estimate_transform('similarity', src, dst)
    result = warp(im / 255., transform.inverse, output_shape=(224, 224)).transpose(2, 0, 1).astype(np.float32)
    return result, transform.params.astype(np.float32)


def load_video(path):
    import cv2
    cap = cv2.VideoCapture(str(path))
    frames = []
    try:
        assert cap.isOpened(), path
        while True:
            ok, image = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()
    return frames


def extractor():
    import torch
    import cv2
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '7'
    actual = subprocess.check_output(['nvidia-smi', '-i', '7', '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip()
    assert actual == GPU_UUID, 'Only authorized node11 GPU7 may run this extractor'
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    # The original DECA extraction retained PyTorch's cuDNN TF32 default.
    # A differential probe reproduced the sealed GT exactly with this setting.
    torch.backends.cudnn.allow_tf32 = True
    sys.path.insert(0, str(VENDOR))
    from decalib.models.encoders import ResnetEncoder
    model = ResnetEncoder(236)
    model.load_state_dict(torch.load(WEIGHT, map_location='cpu', weights_only=False)['E_flame'], strict=True)
    return model.eval().cuda()


def encode(model, cropped, batch_size=128):
    import torch
    output = []
    with torch.inference_mode():
        for start in range(0, len(cropped), batch_size):
            x = torch.from_numpy(np.stack([c[0] for c in cropped[start:start+batch_size]])).cuda()
            output.append(model(x).cpu().numpy())
    return np.concatenate(output)


def worker(shard, shards):
    p = read(WORK / 'protocol.json')
    assert sha(__file__) == p['code_sha256']
    model = extractor()
    # Verify encoder/crop compatibility against a previously sealed GT stream.
    sample = p['samples'][shard]
    uid = sample['id']
    frames = load_video(sample['streams']['gt']['path'])
    lm = np.load(BASE / 'features/gt' / (uid + '.npz'))['landmarks']
    indices = np.linspace(0, len(frames)-1, 8).astype(int)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        cropped = list(pool.map(crop_frame, zip(frames, lm)))
    coeff = encode(model, cropped)
    old = np.load(L2L / 'features/gt' / (uid + '.npz'))
    delta = float(np.abs(coeff - old['coeff']).max())
    transform_delta = float(np.abs(np.stack([c[1] for c in cropped]) - old['transform_params']).max())
    assert delta < 1e-3 and transform_delta == 0, (delta, transform_delta)
    save(WORK / f'parity_{shard:02d}.json', dict(passed=True, id=uid, checked_frames=len(frames), coefficient_max_abs=delta, transform_max_abs=transform_delta))
    for task in p['extraction_tasks'][shard::shards]:
        start = time.time()
        dest = WORK / 'features' / task['method'] / (task['id'] + '.npz')
        receipt_path = dest.with_suffix('.json')
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        with dest.with_suffix('.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if receipt_path.exists():
                receipt = read(receipt_path)
                assert sha(dest) == receipt['feature_sha256'] and receipt['protocol_sha256'] == sha(WORK / 'protocol.json')
                continue
            assert sha(task['video']) == task['video_sha256']
            assert sha(task['landmarks']) == task['landmarks_sha256']
            frames = load_video(task['video'])
            landmark_file = np.load(task['landmarks'])
            lm = landmark_file['landmarks']
            assert len(frames) == len(lm) == task['frames'] and np.isfinite(lm).all()
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
                cropped = list(pool.map(crop_frame, zip(frames, lm)))
            coeff = encode(model, cropped)
            motion = np.concatenate([coeff[:, 150:200], coeff[:, 200:206]], axis=1)
            assert motion.shape == (task['frames'], 56) and np.isfinite(motion).all()
            np.savez_compressed(dest, motion=motion, coeff=coeff, transform_params=np.stack([c[1] for c in cropped]))
            save(receipt_path, dict(id=task['id'], method=task['method'], frames=task['frames'],
                video_sha256=task['video_sha256'], landmark_sha256=task['landmarks_sha256'],
                feature_sha256=sha(dest), protocol_sha256=sha(WORK / 'protocol.json'),
                missing_face_frames=int((~landmark_file['detected']).sum()), seconds=time.time()-start))
            print('extracted', shard, task['method'], task['id'], task['frames'], round(time.time()-start, 2), flush=True)


def clip_metrics(s, g, pred):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from p2_vico_motion_metrics import frechet, tlcc
    if s.shape != g.shape or pred.shape != g.shape or g.shape[1] != 56:
        raise ValueError(f'Unaligned DECA streams: {s.shape}, {g.shape}, {pred.shape}')
    assert all(np.isfinite(x).all() for x in [s, g, pred])
    metrics = {k: {} for k in ['L1', 'L2', 'Frechet', 'Delta_Frechet', 'SND', 'V_D', 'V_D_GT', 'RTLCC', 'RWTLCC']}
    for group, sl in GROUPS.items():
        a, b, c = s[:, sl], g[:, sl], pred[:, sl]
        metrics['L1'][group] = float(np.abs(b-c).mean())
        metrics['L2'][group] = float(np.linalg.norm(b-c, axis=1).mean())
        f, d = frechet(b, c), frechet(np.diff(b, axis=0), np.diff(c, axis=0))
        metrics['Frechet'][group], metrics['Delta_Frechet'][group], metrics['SND'][group] = f, d, f+d
        metrics['V_D'][group] = float(np.var(c, axis=0).mean())
        metrics['V_D_GT'][group] = float(np.var(b, axis=0).mean())
        metrics['RTLCC'][group] = float(np.abs(tlcc(a, b)[0] - tlcc(a, c)[0]).mean()) if len(a) > 61 else None
        windows = [float(np.abs(tlcc(a[t:t+120], b[t:t+120])[0] - tlcc(a[t:t+120], c[t:t+120])[0]).mean()) for t in range(0, len(a)-119, 30)]
        metrics['RWTLCC'][group] = float(np.mean(windows)) if windows else None
    return metrics


def score():
    p = read(WORK / 'protocol.json')
    assert p['metric_code_sha256'] == sha(ROOT / 'scripts/p2_vico_motion_metrics.py')
    methods = [('joint', None), ('dystream', None)] + [('learning2listen', v) for v in VARIANTS]
    results = {}
    for method, variant in methods:
        key = method if variant is None else method + '/' + variant
        rows = []
        for sample in p['samples']:
            uid = sample['id']
            s, g = [np.load(L2L / 'features' / role / (uid + '.npz'))['motion'].astype(float) for role in ['speaker', 'gt']]
            source = WORK / 'features' / method / (uid + '.npz') if variant is None else L2L / 'generated' / variant / (uid + '.npz')
            pred = np.load(source)['motion'].astype(float)
            rows.append(dict(id=uid, frames=len(g), metrics=clip_metrics(s, g, pred), prediction_sha256=sha(source)))
        avg = {k: {group: float(np.mean([r['metrics'][k][group] for r in rows if r['metrics'][k][group] is not None])) for group in GROUPS} for k in rows[0]['metrics']}
        coverage = {k: {group: sum(r['metrics'][k][group] is not None for r in rows) for group in GROUPS} for k in avg}
        results[key] = dict(method=method, variant=variant, clips=43, frames=p['frames_per_stream'], metrics=avg, coverage=coverage, per_clip=rows,
            readout='video re-extraction' if variant is None else 'native motion coefficients',
            training='original person-specific pretrained checkpoint transferred to ViCo' if variant else 'see original model training provenance; no retraining in this evaluation')
        save(WORK / 'score_status.json', dict(method=key, completed_methods=len(results), total_methods=6))
        print('scored', key, flush=True)
    save(WORK / 'RESULTS.json', dict(schema='vico43-common-deca-motion-v1', protocol_sha256=sha(WORK / 'protocol.json'),
        space=p['space'], models=results, published_baseline_results=False, strict_DIM_reproduction=False,
        limitations=p['limitations'], photorealistic_metrics_for_l2l=None))


def audit():
    p, r = read(WORK / 'protocol.json'), read(WORK / 'RESULTS.json')
    assert sha(__file__) == p['code_sha256']
    for path, expected in p['assets'].items():
        assert sha(path) == expected
    for item in p['reused'] + p['native_l2l']:
        assert sha(item['path']) == item['sha256']
    for task in p['extraction_tasks']:
        path = WORK / 'features' / task['method'] / (task['id'] + '.npz')
        receipt = read(path.with_suffix('.json'))
        assert sha(path) == receipt['feature_sha256']
        assert sha(task['video']) == receipt['video_sha256'] == task['video_sha256']
        assert sha(task['landmarks']) == task['landmarks_sha256']
        assert receipt['protocol_sha256'] == sha(WORK / 'protocol.json')
    previous = read(L2L / 'RESULTS.json')['models']
    regression = 0.
    for variant in VARIANTS:
        current = r['models']['learning2listen/' + variant]
        for oldrow, newrow in zip(previous[variant]['per_clip'], current['per_clip']):
            assert oldrow['id'] == newrow['id']
            for metric, groups in oldrow['metrics'].items():
                for group, value in groups.items():
                    actual = newrow['metrics'][metric][group]
                    if value is None:
                        assert actual is None
                    else:
                        regression = max(regression, abs(actual-value))
        assert current['coverage'] == previous[variant]['coverage']
    assert regression < 1e-12
    for model in r['models'].values():
        assert len(model['per_clip']) == 43
        assert sum(x['frames'] for x in model['per_clip']) == 16288
        for metric, groups in model['coverage'].items():
            assert all(n == (42 if metric == 'RTLCC' else 41 if metric == 'RWTLCC' else 43) for n in groups.values())
    # GT self-comparison and short-window behavior are meaningful scorer checks.
    sample = min(p['samples'], key=lambda s: s['streams']['gt']['frames'])
    uid = sample['id']
    s, g = [np.load(L2L / 'features' / role / (uid + '.npz'))['motion'].astype(float) for role in ['speaker', 'gt']]
    null = clip_metrics(s, g, g)
    assert all(value in [None, 0.] for k, groups in null.items() if k not in ['V_D', 'V_D_GT'] for value in groups.values())
    parity = [read(f) for f in sorted(WORK.glob('parity_*.json'))]
    assert parity and all(x['passed'] for x in parity)
    save(WORK / 'FINAL_AUDIT.json', dict(passed=True, methods=6, clips_per_method=43, frames_per_method=16288,
        new_extractions=86, reused_speaker_gt=86, preserved_l2l_outputs=172,
        l2l_per_clip_regression_max_abs=regression, encoder_parity=parity, gt_self_comparison=True,
        coverage=dict(default=43, RTLCC=42, RWTLCC=41), result_sha256=sha(WORK / 'RESULTS.json')))


def run(shards):
    prepare()
    logs, processes = [], []
    start = time.time()
    try:
        for shard in range(shards):
            handle = (WORK / f'extract_{shard:02d}.log').open('a')
            logs.append(handle)
            processes.append(subprocess.Popen([sys.executable, __file__, 'worker', '--shard', str(shard), '--shards', str(shards)],
                stdout=handle, stderr=subprocess.STDOUT, env=dict(os.environ, OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='2')))
        codes = [process.wait() for process in processes]
        assert not any(codes), codes
    finally:
        for handle in logs:
            handle.close()
    score()
    audit()
    save(WORK / 'COMPLETE.json', dict(status='complete', seconds=time.time()-start, audit_passed=True,
         result_sha256=sha(WORK / 'RESULTS.json')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['prepare', 'worker', 'score', 'audit', 'run'])
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=4)
    args = parser.parse_args()
    if args.command == 'worker':
        worker(args.shard, args.shards)
    elif args.command == 'run':
        run(args.shards)
    else:
        globals()[args.command]()
