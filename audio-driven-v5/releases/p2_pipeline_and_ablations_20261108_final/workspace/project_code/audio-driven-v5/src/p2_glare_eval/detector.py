"""Independent MediaPipe + FER adapter and the pinned public GLARE rules."""
import argparse
import csv
import importlib.metadata
import importlib.util
import math
from pathlib import Path
import pickle
import shutil

from .common import ROOT, atomic_json, sha, read, complete, artifact_valid, code_identity
from .upstream import DETECTOR, REACTIONS, module, reaction_module, metrics, verify_sources
from .media import probe

EMOTIONS = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
BLENDSHAPES = ('mouthSmileLeft', 'mouthSmileRight', 'jawOpen', 'eyeWideLeft', 'eyeWideRight',
              'browInnerUp', 'mouthFrownLeft', 'mouthFrownRight', 'browDownLeft', 'browDownRight')


def runtime_identity():
    packages = ('fer', 'facenet-pytorch', 'mediapipe', 'tensorflow-cpu', 'numpy', 'torch',
                'opencv-python', 'opencv-contrib-python', 'scipy', 'keras')
    versions = {name: importlib.metadata.version(name) for name in packages}
    fer = Path(importlib.util.find_spec('fer').origin).parent
    facenet = Path(importlib.util.find_spec('facenet_pytorch').origin).parent
    files = [fer / 'data/emotion_model.hdf5', fer / 'fer.py',
             *[facenet / 'data' / (name + '.pt') for name in ('pnet', 'rnet', 'onet')]]
    return dict(packages=versions, assets={str(p): sha(p) for p in files},
                execution='cpu', fer_api='detect_emotions(BGR), mtcnn=True',
                tensorflow_onednn=False)


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def valid_face(frame):
    import numpy as np
    if frame.get('detected') is not True:
        return False
    landmarks = np.asarray(frame.get('landmarks'), dtype=float)
    if landmarks.ndim != 2 or landmarks.shape[0] < 468 or landmarks.shape[1] != 3 or not np.isfinite(landmarks).all():
        return False
    bbox = frame.get('bbox_px')
    if not isinstance(bbox, list) or len(bbox) != 4 or not all(finite(x) for x in bbox) or bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
        return False
    if not all(finite(frame.get(k)) and abs(frame[k]) <= 180.0001 for k in ('pitch', 'yaw', 'roll')):
        return False
    values = frame.get('blendshapes', {})
    if not all(finite(values.get(k)) and 0 <= values[k] <= 1 for k in BLENDSHAPES):
        return False
    return finite(frame.get('au04_proxy')) and 0 <= frame['au04_proxy'] <= 1


def valid_emotion(faces):
    if len(faces) != 1:
        return False
    value = faces[0]
    box = value.get('box')
    if box is None or len(box) != 4 or not all(math.isfinite(float(x)) for x in box) or box[2] <= 0 or box[3] <= 0:
        return False
    scores = value.get('emotions', {})
    return all(finite(scores.get(k)) and 0 <= scores[k] <= 1 for k in EMOTIONS)


def face_iou(mp_box, fer_box):
    a, b, c, d = mp_box
    x, y, w, h = fer_box
    overlap = max(0, min(c, x + w) - max(a, x)) * max(0, min(d, y + h) - max(b, y))
    union = (c - a) * (d - b) + w * h - overlap
    return overlap / union if union > 0 else 0.


def read_channel(path, name, count):
    values = []
    with Path(path).open() as stream:
        rows = csv.DictReader(stream)
        if rows.fieldnames != ['frame_index', name + '_intensity']:
            raise ValueError(f'Wrong reaction channel schema: {path}')
        for i, row in enumerate(rows):
            if row['frame_index'] != str(i):
                raise ValueError(f'Noncontiguous reaction frame: {path}')
            score = float(row[name + '_intensity'])
            if not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError(f'Invalid reaction score: {path}')
            values.append(score)
    if len(values) != count:
        raise ValueError(f'Reaction channel lost frames: {path}')
    return values


def run_rules(video, folder, policy, count):
    video, folder = Path(video), Path(folder)
    for name in REACTIONS:
        detector = reaction_module(name)
        item = detector.SampleItem(video, folder / 'fer.csv', folder / 'face.pkl')
        destination = folder / 'reactions' / name / 'output_eval'
        result = detector.process_one(1, 1, item, argparse.Namespace(**policy['reaction_parameters'][name]),
            destination, folder / 'unused_overlays', video.parent, False)
        if not result.ok:
            raise ValueError(result.message)
        read_channel(destination / (video.stem + '.csv'), name, count)
    combiner = module('_p2_glare_combine', DETECTOR / 'reaction_detector/combine_reactions.py')
    result = combiner.combine_one_file(Path(video.stem + '.csv'), folder / 'reactions', folder / 'combined',
                                      False, True, dataset_name='eval')
    if not result.ok:
        raise ValueError(result.message)
    shutil.copyfile(folder / 'combined' / (video.stem + '.csv'), folder / 'scores.csv')
    annotation = metrics()['common'].read_annotation(folder / 'scores.csv')
    if annotation.frame_count != count:
        raise ValueError('Combined reaction CSV lost frames')


def detect(video, folder, policy, expected_runtime=None):
    import os
    os.environ['CUDA_VISIBLE_DEVICES']=''
    os.environ['TF_ENABLE_ONEDNN_OPTS']='0'
    import cv2
    import numpy as np
    import torch
    import tensorflow as tf
    from fer import FER
    torch.set_num_threads(2)
    tf.config.set_visible_devices([], 'GPU')
    video, folder = Path(video).resolve(), Path(folder).resolve()
    runtime = runtime_identity()
    if expected_runtime is not None and runtime != expected_runtime:
        raise ValueError('Detector environment or pretrained weights changed')
    if policy != read(ROOT / 'configs/p2_glare_detector_v1.json'):
        raise ValueError('Unknown detector policy; create and validate a separately versioned protocol')
    sources = verify_sources()
    ident = dict(video_sha256=sha(video), policy=policy, runtime=runtime, upstream=sources, code=code_identity())
    if artifact_valid(folder, ident):
        return read(folder / 'complete.json')
    folder.mkdir(parents=True, exist_ok=True)
    geometry = probe(video, policy['fps'])
    try:
        face = module('_p2_glare_face', DETECTOR / 'nod_headshake/face_detection/detect_face_multithread.py')
        task = DETECTOR / 'nod_headshake/face_detection/face_landmarker_v2_with_blendshapes.task'
        face.process_video(video, folder / 'face.pkl', str(task))
        # Only the pickle just generated by the pinned extractor is deserialized.
        with (folder / 'face.pkl').open('rb') as stream:
            faces = pickle.load(stream)
        if len(faces) != geometry['frame_count']:
            raise ValueError('Face extractor lost frames')
        emotion = FER(mtcnn=True)
        capture = cv2.VideoCapture(str(video)); raw = []; bad = []; face_count = emotion_count = 0
        try:
            with (folder / 'fer.csv').open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=['frame_index', *EMOTIONS]); writer.writeheader()
                for i, face_frame in enumerate(faces):
                    ok, bgr = capture.read()
                    if not ok:
                        raise ValueError('FER decoder lost frames')
                    found = emotion.detect_emotions(bgr)
                    serial = [dict(box=[float(x) for x in f['box']], emotions={k:float(v) for k,v in f['emotions'].items()}) for f in found]
                    raw.append(dict(frame_index=i, faces=serial))
                    fv, ev = valid_face(face_frame), valid_emotion(serial)
                    face_count += int(fv); emotion_count += int(ev)
                    same = fv and ev and face_iou(face_frame['bbox_px'], serial[0]['box']) >= policy['min_face_iou']
                    if not same:
                        bad.append(dict(frame_index=i, mediapipe_valid=fv, fer_valid=ev,
                                        association_valid=bool(same), fer_faces=len(serial)))
                    writer.writerow(dict(frame_index=i, **(serial[0]['emotions'] if ev else {k:'' for k in EMOTIONS})))
                if capture.read()[0]:
                    raise ValueError('FER decoder has extra frames')
        finally:
            capture.release()
        atomic_json(folder / 'fer_raw.json', raw)
        quality = dict(frames=len(faces), mediapipe_valid_frames=face_count, fer_valid_frames=emotion_count,
                       jointly_valid_frames=len(faces)-len(bad), invalid_frames=bad,
                       required_frame_coverage=policy['required_frame_coverage'])
        atomic_json(folder / 'quality.json', quality)
        if bad:
            raise ValueError(f'Incomplete independent detection: {len(bad)}/{len(faces)} frames; see quality.json')
        run_rules(video, folder, policy, geometry['frame_count'])
        complete(folder, ident, [folder / n for n in ('scores.csv','face.pkl','fer.csv','fer_raw.json','quality.json')],
                 geometry=geometry, human_verified=False, detector_based=True)
        return read(folder / 'complete.json')
    except Exception as error:
        atomic_json(folder / 'failed.json', dict(error_type=type(error).__name__, error=str(error), identity=ident))
        raise
