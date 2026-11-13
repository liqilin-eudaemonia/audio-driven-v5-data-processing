#!/usr/bin/env python3
"""CPU-only 25fps GLARE nod + reaction detector and gate on the reviewed bounded crop.

Uses the exact final bounded crop (quality-bounded-crop-v1-20261030):
- MediaPipe 25fps pose + blendshapes (face_landmarker_v2_with_blendshapes.task)
- Nodding: recall profile (same as production listen_j_corrected_nod.py)
- Head shaking: vendor detector + headshake_direction_v1 gate
- Frowning & Surprised: vendor detector (FER + MediaPipe blendshapes) + expression_gate_v3
  with exact blendshape upper-bound precheck so clips whose blendshapes cannot pass
  expression_gate_v3 skip FER without any loss of recall.
"""
from __future__ import annotations

import os
os.environ.update(
    CUDA_VISIBLE_DEVICES='',
    OMP_NUM_THREADS='1',
    OPENBLAS_NUM_THREADS='1',
    MKL_NUM_THREADS='1',
    TF_NUM_INTRAOP_THREADS='2',
    TF_NUM_INTEROP_THREADS='1',
    TF_CPP_MIN_LOG_LEVEL='2',
)
import argparse
import contextlib
import hashlib
import json
from pathlib import Path
import pickle
import select
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]

FER_PYTHON_DEFAULT = (
    ROOT / 'experiments/listen_j_reaction_vocal_omni_20261103/reaction_runtime_env/bin/python'
)
OBSERVATION = {
    'head_shaking': 'head shaking',
    'frowning': 'brow lowering',
    'surprised': 'brows raised',
}


def _overlap(a, b):
    x, y, w, h = a
    x0, y0, x1, y1 = b
    inter = max(0, min(x + w, x1) - max(x, x0)) * max(0, min(y + h, y1) - max(y, y0))
    return inter / max(1, w * h + (x1 - x0) * (y1 - y0) - inter)


def run_fer_worker(use_gpu: bool = False):
    """Persistent FER worker subprocess communicating over stdin/stdout JSON lines."""
    import cv2
    import tensorflow as tf
    import torch
    from fer.fer import FER

    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    if use_gpu and torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.025)
    with contextlib.redirect_stdout(sys.stderr):
        detector = FER(mtcnn=True, use_tflite=True)

    for line in sys.stdin:
        try:
            req = json.loads(line)
            video = req['video']
            bboxes = req['bboxes']
            cap = cv2.VideoCapture(str(video))
            if not cap.isOpened():
                raise RuntimeError(f'Cannot open video {video}')
            sad, angry, surprise = [], [], []
            detected_count = 0
            idx = 0
            try:
                while True:
                    good, frame = cap.read()
                    if not good:
                        break
                    faces = detector.detect_emotions(frame)
                    box = bboxes[idx] if idx < len(bboxes) else None
                    pick = (
                        max(
                            range(len(faces)),
                            key=lambda j: _overlap(faces[j]['box'], box)
                            if box
                            else faces[j]['box'][2] * faces[j]['box'][3],
                        )
                        if faces
                        else None
                    )
                    if pick is not None:
                        detected_count += 1
                        emo = faces[pick]['emotions']
                        sad.append(float(emo['sad']))
                        angry.append(float(emo['angry']))
                        surprise.append(float(emo['surprise']))
                    else:
                        sad.append(0.0)
                        angry.append(0.0)
                        surprise.append(0.0)
                    idx += 1
            finally:
                cap.release()
            reply = {
                'ok': True,
                'frames': idx,
                'detected_count': detected_count,
                'sad': sad,
                'angry': angry,
                'surprise': surprise,
            }
        except Exception as exc:
            reply = {'ok': False, 'error': repr(exc)}
        print(json.dumps(reply), flush=True)


def can_possibly_pass_expression_gate(signal, params):
    """Exact necessary condition for any interval in the clip to pass expression_gate_v3."""
    import numpy as np
    from check_glare_expression_gate import smooth

    w = params['smooth_frames']
    bd = smooth(signal['brow_down'], w)
    bi = smooth(signal['brow_inner'], w)
    bo = smooth(signal['brow_outer'], w)
    # Frowning requires min_event_len=6 -> median needs at least 3 frames >= threshold.
    frown_possible = bool(
        np.sum(bd >= params['frown_min_brow_down']) >= 3
        and (
            float(np.max(bd) - np.min(bd)) >= params['frown_min_lift']
            or np.sum(bd >= params['frown_sustained_min_brow']) >= 3
        )
    )
    # Surprise requires min_event_len=5 -> median needs at least 3 frames >= threshold.
    surp_inner = bool(
        np.sum(bi >= params['surprise_min_brow']) >= 3
        and float(np.max(bi) - np.min(bi)) >= params['surprise_min_lift']
    )
    surp_outer = bool(
        np.sum(bo >= params['surprise_min_brow']) >= 3
        and float(np.max(bo) - np.min(bo)) >= params['surprise_min_lift']
    )
    return frown_possible, (surp_inner or surp_outer)


def _vendor_defaults(mod):
    old = sys.argv
    try:
        sys.argv = [mod.__file__]
        args = mod.parse_args()
    finally:
        sys.argv = old
    return {
        k: v
        for k, v in vars(args).items()
        if k.startswith(('min_', 'th_', 'smooth_', 'baseline_', 'pitch_penalty'))
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--row', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--resources', type=Path)
    p.add_argument('--fer-python', type=Path, default=FER_PYTHON_DEFAULT)
    p.add_argument('--fer-gpu', default='')
    p.add_argument('--serve', action='store_true')
    p.add_argument('--fer-worker', action='store_true')
    args = p.parse_args()

    if args.fer_worker:
        if args.fer_gpu != '':
            os.environ['CUDA_VISIBLE_DEVICES'] = str(args.fer_gpu)
        run_fer_worker(use_gpu=(args.fer_gpu != ''))
        return

    import cv2
    import numpy as np
    cv2.setNumThreads(1)
    from listen_benchmark.j_annotation import read, write
    from listen_j_annotation_nod import render_video
    from listen_visual_native8_nod import decode_slot
    from run_glare_nodding_pilot import module, FACE, NOD, VENDOR, MODEL, PROFILES, intervals
    from check_glare_headshake_direction import PARAMS as SHAKE_GATE_PARAMS, assess as assess_shake
    from check_glare_expression_gate import (
        PARAMS as EXPR_GATE_PARAMS,
        blendshapes as load_gate_blendshapes,
        assess_frown,
        assess_surprise,
    )

    with contextlib.redirect_stdout(sys.stderr):
        face = module('corrected_reaction_face', FACE)
        nod = module('corrected_reaction_nod', NOD)
        shake_mod = module(
            'corrected_reaction_shake',
            VENDOR / 'reaction_detector/head_shaking/run_head_shaking_intensity_mt.py',
        )
        frown_mod = module(
            'corrected_reaction_frown',
            VENDOR / 'reaction_detector/frowning/run_frowning_intensity_mt.py',
        )
        surprise_mod = module(
            'corrected_reaction_surprise',
            VENDOR / 'reaction_detector/surprised/run_surprised_intensity_mt.py',
        )

    shake_cfg = _vendor_defaults(shake_mod)
    frown_cfg = _vendor_defaults(frown_mod)
    surprise_cfg = _vendor_defaults(surprise_mod)
    model_sha = hashlib.sha256(MODEL.read_bytes()).hexdigest()

    fer_proc = {'process': None}

    def query_fer(video_path: Path, bboxes: list):
        proc = fer_proc['process']
        if proc is None or proc.poll() is not None:
            env = dict(
                os.environ,
                CUDA_VISIBLE_DEVICES=str(args.fer_gpu) if args.fer_gpu != '' else '',
                OMP_NUM_THREADS='2',
                OPENBLAS_NUM_THREADS='1',
                MKL_NUM_THREADS='1',
                TF_NUM_INTRAOP_THREADS='2',
                TF_NUM_INTEROP_THREADS='1',
                TF_CPP_MIN_LOG_LEVEL='2',
            )
            cmd = [str(args.fer_python), str(Path(__file__).resolve()), '--fer-worker']
            if args.fer_gpu != '':
                cmd.extend(['--fer-gpu', str(args.fer_gpu)])
            proc = subprocess.Popen(
                cmd,
                env=env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=sys.stderr,
                text=True,
                bufsize=1,
            )
            fer_proc['process'] = proc
        proc.stdin.write(json.dumps({'video': str(video_path), 'bboxes': bboxes}) + '\n')
        proc.stdin.flush()
        if not select.select([proc.stdout], [], [], 300)[0]:
            proc.kill()
            proc.wait()
            fer_proc['process'] = None
            raise TimeoutError('FER worker timed out')
        line = proc.stdout.readline()
        if not line:
            fer_proc['process'] = None
            raise RuntimeError('FER worker exited unexpectedly')
        reply = json.loads(line)
        if not reply.get('ok'):
            raise RuntimeError('FER worker error: ' + str(reply.get('error')))
        return reply

    def execute(row_path, output, resources, decoded_dir=None):
        row = read(row_path)
        if row['source']['crop_proposal']['revision'] != 'quality-bounded-crop-v1-20261030':
            raise ValueError('Expected reviewed corrected crop (quality-bounded-crop-v1-20261030)')
        duration = float(row['source']['duration_s'])
        resources_root = Path(resources)
        selected_cfg = resources_root / 'selected.json'
        slots = read(selected_cfg)['decode_slots'] if selected_cfg.exists() else 48

        with tempfile.TemporaryDirectory(prefix='listen-j-react-', dir='/dev/shm') as tmp:
            directory = Path(tmp)
            t0 = time.monotonic()
            with decode_slot(resources_root / 'resources', slots):
                t_acq = time.monotonic()
                video, meta = render_video(
                    row,
                    directory,
                    fast_crop=True,
                    decoded_dir=Path(decoded_dir) if decoded_dir else None,
                )
            t_ren = time.monotonic()
            pkl = directory / 'pose.pkl'
            with contextlib.redirect_stdout(sys.stderr):
                extracted = face.process_video(video, pkl, str(MODEL))
            t_face = time.monotonic()
            if extracted['frames'] != meta['frames']:
                raise ValueError('Landmark frame count mismatch')
            detected, pitch, yaw = nod.load_pose_from_pkl(pkl)
            base_result = {
                'schema': 'listen-j-corrected-reaction-nod-v1',
                'sample_id': row['sample_id'],
                'duration_s': duration,
                'fps': 25,
                'frames': meta['frames'],
                'source': row['source'],
                'media': meta,
                'profile': 'recall',
                'parameters': {
                    'nodding': PROFILES['recall'],
                    'head_shaking': shake_cfg,
                    'frowning': frown_cfg,
                    'surprised': surprise_cfg,
                    'headshake_direction_gate': SHAKE_GATE_PARAMS,
                    'expression_gate_v3': EXPR_GATE_PARAMS,
                },
                'model_sha256': model_sha,
                'pose': {
                    'detected': detected.astype(int).tolist(),
                    'pitch': pitch.tolist(),
                    'yaw': yaw.tolist(),
                },
                'input_execution': {'fast_frame_crop': True},
                'training_admitted': False,
            }
            if not detected.any():
                out_obj = dict(
                    base_result,
                    status='unavailable',
                    reason='no_detected_face',
                    scores=None,
                    events=None,
                    reaction_cues=[],
                    completed_at=time.time(),
                )
                if Path(output).exists():
                    raise ValueError('Refusing to replace an existing reaction measurement')
                write(output, out_obj)
                return

            # 1. Nodding (recall)
            nod_scores = nod.compute_nodding_intensity(
                detected, pitch, yaw, argparse.Namespace(**PROFILES['recall'])
            )
            nod_raw_events = intervals(nod_scores, detected, 25, duration)
            nod_events = [
                dict(
                    e,
                    event_id=f'nod-{i + 1:04d}',
                    type='nodding',
                    modality='visual',
                    description='Head nodding',
                    source='GLARE_nodding_v2_recall',
                    status='candidate',
                )
                for i, e in enumerate(nod_raw_events)
            ]

            # 2. Head shaking + headshake_direction_v1 gate
            shake_scores = shake_mod.compute_head_shaking_intensity(
                detected, pitch, yaw, argparse.Namespace(**shake_cfg)
            )
            shake_raw_events = intervals(shake_scores, detected, 25, duration)
            smoothed_pose = {
                'pitch': shake_mod.moving_average(
                    shake_mod.fill_pose_gaps(np.asarray(pitch), detected),
                    SHAKE_GATE_PARAMS['smooth_frames'],
                ),
                'yaw': shake_mod.moving_average(
                    shake_mod.fill_pose_gaps(np.asarray(yaw), detected),
                    SHAKE_GATE_PARAMS['smooth_frames'],
                ),
            }
            shake_decisions = []
            shake_kept = []
            for idx, ev in enumerate(shake_raw_events):
                a, b = ev['start_frame'], ev['end_frame_exclusive']
                dec = assess_shake(
                    smoothed_pose['yaw'][a:b],
                    smoothed_pose['pitch'][a:b],
                    detected[a:b],
                    SHAKE_GATE_PARAMS,
                )
                dec.update(event=ev, original_event_id=f'glare-head_shaking-{idx + 1:03d}')
                shake_decisions.append(dec)
                if dec['retain']:
                    shake_kept.append(ev)

            # 3. Frowning & Surprised + expression_gate_v3
            signal = load_gate_blendshapes(pkl)
            frown_possible, surprise_possible = can_possibly_pass_expression_gate(
                signal, EXPR_GATE_PARAMS
            )
            frown_decisions, surprise_decisions = [], []
            frown_kept, surprise_kept = [], []
            fer_status = 'skipped_by_blendshape_bound'
            t_fer_start = time.monotonic()

            if frown_possible or surprise_possible:
                with pkl.open('rb') as handle:
                    poses = pickle.load(handle)
                bboxes = [
                    item.get('bbox_px') if isinstance(item, dict) else None for item in poses
                ]
                fer_reply = query_fer(video, bboxes)
                if fer_reply['frames'] != meta['frames']:
                    raise ValueError('FER frame count mismatch')
                if fer_reply['detected_count'] == 0:
                    fer_status = 'no_face'
                else:
                    fer_status = 'ok'
                    sad = {
                        i: float(np.clip(x, 0.0, 1.0)) for i, x in enumerate(fer_reply['sad'])
                    }
                    angry = {
                        i: float(np.clip(x, 0.0, 1.0)) for i, x in enumerate(fer_reply['angry'])
                    }
                    surprise_arr = {
                        i: float(np.clip(x, 0.0, 1.0))
                        for i, x in enumerate(fer_reply['surprise'])
                    }

                    if frown_possible:
                        mask_f, mouth_f, brow_f, au04_f = frown_mod.load_blendshape_signals(pkl)
                        frown_scores = frown_mod.compute_frowning_intensity(
                            sad, angry, mask_f, mouth_f, brow_f, au04_f, **frown_cfg
                        )
                        frown_raw_events = intervals(frown_scores, detected, 25, duration)
                        for ev in frown_raw_events:
                            a, b = ev['start_frame'], ev['end_frame_exclusive']
                            dec = assess_frown(
                                signal['brow_down'], frown_scores, a, b, EXPR_GATE_PARAMS
                            )
                            dec = {
                                **dec,
                                'type': 'frowning',
                                'start': ev['start'],
                                'end': ev['end'],
                                'peak_score': ev.get('peak_score'),
                            }
                            frown_decisions.append(dec)
                            if dec['retain']:
                                frown_kept.append(ev)

                    if surprise_possible:
                        mask_s, jaw_s, eye_s, brow_s = surprise_mod.load_blendshape_signals(
                            pkl
                        )
                        surprise_scores = surprise_mod.compute_surprised_intensity(
                            surprise_arr, mask_s, jaw_s, eye_s, brow_s, **surprise_cfg
                        )
                        surprise_raw_events = intervals(surprise_scores, detected, 25, duration)
                        for ev in surprise_raw_events:
                            a, b = ev['start_frame'], ev['end_frame_exclusive']
                            dec = assess_surprise(
                                signal['brow_inner'],
                                signal['brow_outer'],
                                signal['jaw'],
                                a,
                                b,
                                EXPR_GATE_PARAMS,
                            )
                            dec = {
                                **dec,
                                'type': 'surprised',
                                'start': ev['start'],
                                'end': ev['end'],
                                'peak_score': ev.get('peak_score'),
                            }
                            surprise_decisions.append(dec)
                            if dec['retain']:
                                surprise_kept.append(ev)
            t_fer_end = time.monotonic()

            reaction_cues = []
            for ev in shake_kept:
                reaction_cues.append(
                    {
                        'start': ev['start'],
                        'end': ev['end'],
                        'observation': OBSERVATION['head_shaking'],
                    }
                )
            for ev in frown_kept:
                reaction_cues.append(
                    {
                        'start': ev['start'],
                        'end': ev['end'],
                        'observation': OBSERVATION['frowning'],
                    }
                )
            for ev in surprise_kept:
                reaction_cues.append(
                    {
                        'start': ev['start'],
                        'end': ev['end'],
                        'observation': OBSERVATION['surprised'],
                    }
                )
            reaction_cues.sort(key=lambda item: (item['start'], item['observation']))
            for idx, cue in enumerate(reaction_cues, 1):
                cue['id'] = f'reaction-{idx:04d}'

            out_obj = dict(
                base_result,
                status='ok',
                fer_status=fer_status,
                scores=[float(x) for x in nod_scores],
                events=nod_events,
                reaction_cues=reaction_cues,
                reaction_diagnostics={
                    'head_shaking': shake_decisions,
                    'frowning': frown_decisions,
                    'surprised': surprise_decisions,
                    'frown_possible': frown_possible,
                    'surprise_possible': surprise_possible,
                },
                performance={
                    'decode_lock_wait_s': round(t_acq - t0, 4),
                    'render_s': round(t_ren - t_acq, 4),
                    'face_s': round(t_face - t_ren, 4),
                    'fer_s': round(t_fer_end - t_fer_start, 4),
                },
                completed_at=time.time(),
            )
            if Path(output).exists():
                raise ValueError('Refusing to replace an existing reaction measurement')
            write(output, out_obj)

    try:
        if args.serve:
            for line in sys.stdin:
                try:
                    req = json.loads(line)
                    execute(
                        req['row'],
                        req['output'],
                        req['resources'],
                        req.get('decoded_dir'),
                    )
                    reply = {'ok': True, 'output': req['output']}
                except Exception as exc:
                    reply = {'ok': False, 'error': repr(exc)}
                print(json.dumps(reply), flush=True)
        else:
            if not all((args.row, args.output, args.resources)):
                p.error('row/output/resources are required')
            execute(args.row, args.output, args.resources)
    finally:
        if fer_proc['process'] is not None and fer_proc['process'].poll() is None:
            fer_proc['process'].terminate()


if __name__ == '__main__':
    main()
