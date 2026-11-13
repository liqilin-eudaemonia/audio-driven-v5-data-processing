#!/usr/bin/env python3
"""CPU-only, bounded GLARE nodding pilot on the frozen 50-clip review cache."""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import argparse
import hashlib
import importlib.util
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data_previews/listen_benchmark_review/prompt_versions_20261029/data.json'
OUT = ROOT / 'experiments/listen_glare_nodding50_20261029'
VENDOR = ROOT / 'related/visual_reaction_annotation'
FACE = VENDOR / 'nod_headshake/face_detection/detect_face_multithread.py'
MODEL = FACE.parent / 'face_landmarker_v2_with_blendshapes.task'
NOD = VENDOR / 'reaction_detector/nodding/run_nodding_intensity_mt_v2.py'
FIELDS = ('smooth_window', 'baseline_window', 'th_on', 'th_off', 'min_event_len',
          'min_turns_per_event', 'yaw_penalty_weight', 'min_pitch_peak_to_valley_deg',
          'min_peak_pitch_vel_deg', 'min_cycles_per_event')
PROFILES = {name: dict(zip(FIELDS, values)) for name, values in {
    'default': (9, 45, .58, .42, 10, 2, .50, 3., .20, 1),
    'balanced': (9, 40, .50, .35, 7, 1, .40, 2., .12, 0),
    'recall': (7, 35, .45, .30, 6, 1, .35, 1.8, .10, 0),
}.items()}

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)

def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

def intervals(scores, detected, fps, duration):
    """Half-open runs of the vendor's already gated score; no extra score threshold."""
    result, start = [], None
    for i in range(len(scores) + 1):
        positive = i < len(scores) and scores[i] > 0 and i / fps < duration
        if positive and start is None:
            start = i
        if not positive and start is not None:
            result.append({'start': start / fps, 'end': min(i / fps, duration),
                           'start_frame': start, 'end_frame_exclusive': i,
                           'peak_score': float(max(scores[start:i])),
                           'face_detection_fraction': float(sum(detected[start:i]) / (i-start)),
                           'touches_clip_boundary': start == 0 or i / fps >= duration})
            start = None
    return result

def main():
    global SOURCE, OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-media-index', type=Path, default=SOURCE, help='JSON samples with id/video/duration/crop; paths relative to index')
    parser.add_argument('--output', type=Path, default=OUT)
    parser.add_argument('--profile', choices=('all', 'recall'), default='all')
    parser.add_argument('--sample-ids', type=Path, help='Optional JSON with ids array')
    parser.add_argument('--workers', type=int, choices=(1, 2), default=2)
    parser.add_argument('--limit', type=int, default=50)
    parser.add_argument('--reviewed-only', action='store_true', help='Only the 15 clips previously reviewed by the user')
    args = parser.parse_args()
    SOURCE, OUT = args.input_media_index.resolve(), args.output.resolve()
    effective_profiles = PROFILES if args.profile == 'all' else {'recall': PROFILES['recall']}
    import cv2
    cv2.setNumThreads(1)
    face, nod = module('glare_pilot_face', FACE), module('glare_pilot_nod', NOD)
    samples = json.loads(SOURCE.read_text())['samples']
    if args.sample_ids:
        ids=json.loads(args.sample_ids.read_text())['ids']
        by_id={s['id']:s for s in samples}
        samples=[by_id[sid] for sid in ids]
    if args.reviewed_only:
        feedback = ROOT / 'experiments/listen_visual_input50_20261029/native_fps_decision/feedback/human_review_20261029T103511Z.json'
        records = json.loads(feedback.read_text())['records']
        ids = list(dict.fromkeys(r['sample_id'] for r in records))
        by_id = {s['id']: s for s in samples}
        samples = [by_id[sid] for sid in ids]
    if args.limit < 0:raise ValueError('--limit must be nonnegative; 0 means all')
    if args.limit:samples = samples[:args.limit]
    if len({s['id'] for s in samples}) != len(samples):raise ValueError('Duplicate sample IDs')
    protocol = {'schema': 'glare-nodding-pilot-v1', 'profiles': effective_profiles,
                'source_sha256': sha(SOURCE), 'face_code_sha256': sha(FACE),
                'nod_code_sha256': sha(NOD), 'model_sha256': sha(MODEL),
                'runner_sha256': sha(Path(__file__)), 'workers': args.workers,
                'human_ground_truth': False, 'training_admitted': False,
                'interval_rule': 'positive vendor gated score runs; frame/fps; end clipped to source duration',
                'interpretation': 'candidate intervals, not calibrated probabilities or counts of individual nods'}
    fingerprint = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    save(OUT / 'protocol.json', protocol)
    save(OUT / 'samples.json', {'ids': [s['id'] for s in samples], 'reviewed_only': args.reviewed_only})

    def process(s):
        started = time.monotonic()
        sid = s['id']
        target = OUT / 'results' / (sid + '.json')
        video = (SOURCE.parent / s['video']).resolve(strict=True)
        video_hash = sha(video)
        if target.exists():
            old = json.loads(target.read_text())
            if old.get('fingerprint') == fingerprint and old.get('video_sha256') == video_hash and old['status'] == 'ok':
                return old
        result = {'id': sid, 'fingerprint': fingerprint, 'video_sha256': video_hash,
                  'video': str(video), 'duration': s['duration'], 'status': 'error'}
        try:
            meta = json.loads((video.parent / 'media.json').read_text())
            assert meta['sample_id'] == sid and meta['video_transform'] == s['crop']
            assert abs(meta['duration_s'] - s['duration']) < 1e-6
            cap = cv2.VideoCapture(str(video))
            fps, count = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            shape = [int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))]
            cap.release()
            assert fps == 25 and shape == [512, 512] and count == meta['frames']
            pkl = OUT / 'landmarks' / (sid + '.pkl')
            tmp = pkl.with_suffix('.tmp.pkl')
            extracted = face.process_video(video, tmp, str(MODEL))
            assert extracted['frames'] == count, (extracted, count)
            tmp.replace(pkl)
            detected, pitch, yaw = nod.load_pose_from_pkl(pkl)
            assert len(detected) == count
            if not detected.any():
                raise ValueError('No face detected; cannot interpret as no nodding')
            profiles = {}
            for name, parameters in effective_profiles.items():
                scores = nod.compute_nodding_intensity(detected, pitch, yaw, argparse.Namespace(**parameters))
                csv = OUT / 'scores' / name / (sid + '.csv')
                csv.parent.mkdir(parents=True, exist_ok=True)
                nod.write_intensity_csv(csv, scores)
                profiles[name] = {'events': intervals(scores, detected, fps, s['duration']),
                                  'scores': [float(x) for x in scores]}
            result.update(status='ok', fps=fps, frames=count, detected_frames=int(detected.sum()),
                          face_detection_fraction=float(detected.mean()), profiles=profiles,
                          pose={'detected': detected.astype(int).tolist(), 'pitch': pitch.tolist(), 'yaw': yaw.tolist()},
                          media_metadata_sha256=sha(video.parent/'media.json'))
        except Exception as exc:
            result['error'] = repr(exc)
        result['elapsed_s'] = time.monotonic() - started
        save(target, result)
        return result

    started, results = time.monotonic(), []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(process, s) for s in samples]):
            r = future.result()
            results.append(r)
            print(json.dumps({'completed': len(results), 'total': len(samples), 'id': r['id'], 'status': r['status'],
                              'error': r.get('error'), 'events': {k: len(v['events']) for k,v in r.get('profiles',{}).items()}}), flush=True)
    ok = [r for r in results if r['status'] == 'ok']
    summary = {'samples': len(samples), 'ok': len(ok), 'errors': [r['id'] for r in results if r['status'] != 'ok'],
               'wall_s': time.monotonic()-started,
               'profiles': {k: {'clips_with_candidates': sum(bool(r['profiles'][k]['events']) for r in ok),
                                 'candidate_intervals': sum(len(r['profiles'][k]['events']) for r in ok),
                                 'candidate_seconds': sum(e['end']-e['start'] for r in ok for e in r['profiles'][k]['events'])}
                            for k in effective_profiles}}
    save(OUT / 'summary.json', summary)
    print(json.dumps(summary), flush=True)
    if summary['errors']:
        raise SystemExit(1)

if __name__ == '__main__':
    main()
