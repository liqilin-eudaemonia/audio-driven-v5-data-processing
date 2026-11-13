#!/usr/bin/env python3
"""Persistent CPU detector using the existing production environments/recipes.

The MP process and FER subprocess share the same small decoded BGR mmap. The
original compressed source is never opened by either detector. Existing nod,
shake and physical brow observations remain authoritative on matching sources.
"""
import argparse
import contextlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time
import traceback

os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                  MKL_NUM_THREADS='1', TF_NUM_INTRAOP_THREADS='2', TF_NUM_INTEROP_THREADS='1',
                  TF_CPP_MIN_LOG_LEVEL='2', GLOG_minloglevel='2')
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
FER_PY = ROOT / 'experiments/listen_j_reaction_vocal_omni_20261103/reaction_runtime_env/bin/python'
CLASSES = ['smiling', 'visual_laughter', 'brow_lowering', 'brows_raised', 'head_nodding', 'head_shaking']
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']


def fer_server():
    import cv2
    import numpy as np
    import torch
    import tensorflow as tf
    from fer.fer import FER
    from listen_j_corrected_reaction import _overlap
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(1)
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    with contextlib.redirect_stdout(sys.stderr):
        model = FER(mtcnn=True, use_tflite=True)
    for line in sys.stdin:
        try:
            req = json.loads(line)
            pixels = np.load(req['pixels'], mmap_mode='r')
            boxes = req['bboxes']
            probs = np.zeros((len(pixels), 7), np.float32)
            detected = np.zeros(len(pixels), bool)
            association = np.zeros(len(pixels), np.float32)
            counts = np.zeros(len(pixels), np.int16)
            for i, bgr in enumerate(pixels):
                with contextlib.redirect_stdout(sys.stderr):
                    faces = model.detect_emotions(bgr)
                counts[i] = len(faces)
                if not faces:
                    continue
                box = boxes[i]
                pick = max(range(len(faces)), key=lambda j: _overlap(faces[j]['box'], box)
                           if box else faces[j]['box'][2] * faces[j]['box'][3])
                detected[i] = True
                probs[i] = [faces[pick]['emotions'][k] for k in EMOTIONS]
                association[i] = _overlap(faces[pick]['box'], box) if box else 0.
            np.savez(req['output'], fer=probs, fer_detected=detected, face_iou=association,
                     fer_face_count=counts)
            reply = dict(ok=True, output=req['output'])
        except Exception:
            reply = dict(ok=False, error=traceback.format_exc())
        print(json.dumps(reply), flush=True)


class Detector:
    def __init__(self):
        import cv2
        import mediapipe as mp
        from run_glare_nodding_pilot import module, FACE, MODEL, VENDOR, NOD
        from listen_j_corrected_reaction import _vendor_defaults
        self.mp, self.model_path = mp, MODEL
        cv2.setNumThreads(1)
        self.face = module('shared_face_native', FACE)
        self.mods = {name: module('shared_reaction_' + name, VENDOR / 'reaction_detector' / name /
                                 ('run_nodding_intensity_mt_v2.py' if name == 'nodding' else 'run_' + name + '_intensity_mt.py'))
                     for name in ['nodding', 'head_shaking', 'smiling', 'laughing', 'frowning', 'surprised']}
        self.parameters = {name: _vendor_defaults(m) for name, m in self.mods.items() if name != 'nodding'}
        self.fer = subprocess.Popen([str(FER_PY), str(Path(__file__).resolve()), 'fer-server'],
                                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)

    def poses(self, pixels):
        data = []
        f = self.face
        with f.FaceLandmarker.create_from_options(f.build_options(str(self.model_path))) as model:
            for i, bgr in enumerate(pixels):
                image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB,
                                      data=__import__('numpy').ascontiguousarray(bgr[:, :, ::-1]))
                result = model.detect_for_video(image, int(i * 1000. / 25))
                row = dict(detected=False, landmarks=None, bbox_norm=None, bbox_px=None,
                           blendshapes={}, au04_proxy=None, pitch=None, roll=None, yaw=None)
                if result.face_landmarks:
                    landmarks = result.face_landmarks[0]
                    row['detected'] = True
                    row['landmarks'] = [[v.x, v.y, v.z] for v in landmarks]
                    row['bbox_norm'], row['bbox_px'] = f.landmarks_to_bbox(landmarks, 512, 512)
                    if result.face_blendshapes:
                        row['blendshapes'] = {b.category_name: b.score for b in result.face_blendshapes[0]}
                        row['au04_proxy'] = (row['blendshapes'].get('browDownLeft', 0.) + row['blendshapes'].get('browDownRight', 0.)) / 2
                    if result.facial_transformation_matrixes:
                        row['pitch'], row['roll'], row['yaw'] = f.matrix_to_euler_degrees(result.facial_transformation_matrixes[0])
                data.append(row)
        return data

    def run(self, folder, output):
        import numpy as np
        from glare_shared_decode import production_pixels
        from run_glare_nodding_pilot import PROFILES, intervals
        from check_glare_headshake_direction import PARAMS as SHAKE_GATE, assess
        from check_glare_expression_gate import PARAMS as EXPR_GATE, blendshapes, assess_frown, assess_surprise
        folder = Path(folder)
        row = json.loads((folder / 'row.json').read_text())
        pixels, media = production_pixels(folder)
        poses = self.poses(pixels)
        pkl = folder / 'pose.pkl'
        with pkl.open('wb') as stream:
            pickle.dump(poses, stream)
        request = dict(pixels=str(folder / 'detector_bgr.npy'),
                       bboxes=[v['bbox_px'] for v in poses], output=str(folder / 'fer_arrays.npz'))
        self.fer.stdin.write(json.dumps(request) + '\n')
        self.fer.stdin.flush()
        reply = json.loads(self.fer.stdout.readline())
        assert reply['ok'], reply
        fer = dict(np.load(request['output']))
        detected, pitch, yaw = self.mods['nodding'].load_pose_from_pkl(pkl)
        n = len(pixels)
        scores, valid = np.zeros((n, 6), np.float32), np.zeros((n, 6), bool)
        smile = self.mods['smiling']
        happy = {i: float(x) for i, x in enumerate(fer['fer'][:, EMOTIONS.index('happy')])}
        mask, mouth = smile.load_smile_blendshape_scores(pkl)
        scores[:, 0] = smile.compute_smiling_intensity(happy, mask, mouth, **self.parameters['smiling'])
        laugh = self.mods['laughing']
        mask, mouth, jaw = laugh.load_blendshape_signals(pkl)
        scores[:, 1] = laugh.compute_laughing_intensity(happy, mask, mouth, jaw, **self.parameters['laughing'])
        fer_valid = detected.astype(bool) & fer['fer_detected'] & (fer['face_iou'] >= .1)
        # Conservative whole-clip validity, as in the accepted real-data adapter.
        valid[:, :4] = bool(fer_valid.all())
        valid[:, 4:] = detected[:, None].astype(bool)
        existing = None
        evidence_file = folder / 'production_evidence.json'
        if evidence_file.exists():
            ev = json.loads(evidence_file.read_text())
            if ev.get('status') == 'indexed' and ev.get('detector_status') == 'ok':
                import hashlib
                raw = Path(ev['detector_file']).read_bytes()
                assert hashlib.sha256(raw).hexdigest() == ev['detector_sha256']
                existing = json.loads(raw)
        nod = self.mods['nodding']
        scores[:, 4] = nod.compute_nodding_intensity(detected, pitch, yaw, argparse.Namespace(**PROFILES['recall']))
        shake = self.mods['head_shaking']
        raw_shake = shake.compute_head_shaking_intensity(detected, pitch, yaw, argparse.Namespace(**self.parameters['head_shaking']))
        smooth = {k: shake.moving_average(shake.fill_pose_gaps(v, detected), SHAKE_GATE['smooth_frames'])
                  for k, v in [('pitch', pitch), ('yaw', yaw)]}
        for event in intervals(raw_shake, detected, 25, row['duration_s']):
            a, b = event['start_frame'], event['end_frame_exclusive']
            if assess(smooth['yaw'][a:b], smooth['pitch'][a:b], detected[a:b], SHAKE_GATE)['retain']:
                scores[a:b, 5] = raw_shake[a:b]
        signal = blendshapes(pkl)
        frown = self.mods['frowning']
        m, mouth, brow, au04 = frown.load_blendshape_signals(pkl)
        raw_frown = frown.compute_frowning_intensity(
            {i:float(v) for i,v in enumerate(fer['fer'][:,4])},
            {i:float(v) for i,v in enumerate(fer['fer'][:,0])}, m, mouth, brow, au04,
            **self.parameters['frowning'])
        surprise = self.mods['surprised']
        m, jaw, eye, brow = surprise.load_blendshape_signals(pkl)
        raw_surprise = surprise.compute_surprised_intensity(
            {i:float(v) for i,v in enumerate(fer['fer'][:,5])}, m, jaw, eye, brow,
            **self.parameters['surprised'])
        for col, channel, gate in [(2, raw_frown, 'frown'), (3, raw_surprise, 'surprise')]:
            for event in intervals(channel, detected, 25, row['duration_s']):
                a, b = event['start_frame'], event['end_frame_exclusive']
                decision = (assess_frown(signal['brow_down'], channel, a, b, EXPR_GATE) if gate == 'frown' else
                            assess_surprise(signal['brow_inner'], signal['brow_outer'], signal['jaw'], a, b, EXPR_GATE))
                if decision['retain']:
                    scores[a:b, col] = channel[a:b]
        origin = 'measured_production_configuration'
        if existing is not None:
            assert existing['frames'] == n
            # Preserve accepted existing observations exactly; don't relabel them
            # from newly measured MP/FER outputs. Missing two expressions are supplemental.
            scores[:, 4] = existing['scores']
            old_face = np.asarray(existing['pose']['detected'], bool)
            valid[:, 4:] = old_face[:, None]
            old_shake = shake.compute_head_shaking_intensity(
                np.asarray(existing['pose']['detected']),
                np.asarray(existing['pose']['pitch'], dtype=float),
                np.asarray(existing['pose']['yaw'], dtype=float),
                argparse.Namespace(**existing['parameters']['head_shaking']))
            scores[:, [2, 3, 5]] = 0.
            for cue in existing['reaction_cues']:
                col = {'brow lowering':2, 'brows raised':3, 'head shaking':5}[cue['observation']]
                a = max(0, round(cue['start'] * 25))
                b = min(n, round(cue['end'] * 25))
                scores[a:b, col] = old_shake[a:b] if col == 5 else 1.
            if existing.get('fer_status') != 'ok':
                valid[:, 2:4] = False
            origin = 'existing_production_nod_shake_brow; supplemental_smile_visual_laughter'
        blend_names = sorted({k for p in poses for k in p['blendshapes']})
        blend = np.asarray([[p['blendshapes'].get(k, 0.) for k in blend_names] for p in poses], np.float32)
        np.savez_compressed(output, reaction=scores, reaction_valid=valid,
            classes=np.asarray(CLASSES), face_detected=detected.astype(bool),
            pose=np.asarray([[p[k] if p[k] is not None else np.nan for k in ('pitch','yaw','roll')] for p in poses], np.float32),
            blendshape_names=np.asarray(blend_names), blendshapes=blend, **fer)
        return dict(frames=n, classes=CLASSES, source=origin,
                    known_per_class=valid.sum(0).tolist(), fer_detected=int(fer['fer_detected'].sum()),
                    detector_video_sha256=media['video_sha256'],
                    original_source_opens=0, physical_brow_names_preserved=True,
                    independent_channels=True, human_verified=False)

    def close(self):
        if self.fer.poll() is None:
            self.fer.terminate()
            self.fer.wait()


def server():
    with contextlib.redirect_stdout(sys.stderr):
        detector = Detector()
    try:
        for line in sys.stdin:
            try:
                req = json.loads(line)
                with contextlib.redirect_stdout(sys.stderr):
                    result = detector.run(req['folder'], req['output'])
                reply = dict(ok=True, output=req['output'], result=result)
            except Exception:
                reply = dict(ok=False, error=traceback.format_exc())
            print(json.dumps(reply), flush=True)
    finally:
        detector.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['server', 'fer-server'])
    args = p.parse_args()
    fer_server() if args.action == 'fer-server' else server()
