"""SPECTRE demo frontend at pinned submodules, separate from the FAN4 proxy."""
import argparse
import collections
import hashlib
import importlib
import importlib.util
from pathlib import Path
import sys
import tempfile
import time
import types

WORK = Path(__file__).resolve().parent
ROOT = WORK.parents[1]
AUDIT = WORK / 'official_tracker_audit'
VENDOR = AUDIT / 'vendor'
SPECTRE = ROOT / 'compare/deployment/vendor/spectre'
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'compare/deployment')]
from p2_avatarforcing_table4_generate import read, save, sha, guard
from p2_spectre_coefficients import Encoder, decode25, source_crop_function


def load_frontend():
    import torch
    source_manifest = read(AUDIT / 'SOURCE_MANIFEST.json')
    weights_manifest = read(AUDIT / 'WEIGHT_MANIFEST.json')
    for record in source_manifest + weights_manifest:
        path = ROOT.parents[1] / record['path']
        assert sha(path) == record['sha256'], path
    package = types.ModuleType('ibug')
    package.__path__ = [str(VENDOR / name / 'ibug') for name in ['face_alignment', 'face_detection']]
    sys.modules['ibug'] = package
    tracker_package = types.ModuleType('official_spectre_tracker')
    tracker_package.__path__ = [str(VENDOR / 'visual_speech/tracker')]
    sys.modules['official_spectre_tracker'] = tracker_package
    module = importlib.import_module('official_spectre_tracker.face_tracker')
    tracker = module.FaceTracker(device='cuda:0')
    get_landmarks = importlib.import_module('official_spectre_tracker.utils').get_landmarks
    counts = {}
    for name, net, path in [
        ('retinaface_resnet50', tracker.face_detector.net,
         VENDOR / 'face_detection/ibug/face_detection/retina_face/weights/Resnet50_Final.pth'),
        ('2dfan2', tracker.landmark_detector.net,
         VENDOR / 'face_alignment/ibug/face_alignment/fan/weights/2dfan2.pth')]:
        state = torch.load(path, map_location='cpu', weights_only=False)
        if 'state_dict' in state:
            state = state['state_dict']
        state = {k.removeprefix('module.'): v for k, v in state.items()}
        actual = net.state_dict()
        missing = []
        for key, value in actual.items():
            if key not in state:
                assert key.endswith('num_batches_tracked'), key
                missing.append(key)
            else:
                assert torch.equal(value.cpu(), state[key]), (name, key)
        assert not set(state)-set(actual), (name, set(state)-set(actual))
        counts[name] = dict(equal_tensors=len(actual)-len(missing),
                            missing_only_unused_batch_counters=missing, sha256=sha(path))
    assert tracker.face_detector.threshold == .8
    assert tracker.landmark_detector.config.num_modules == 2
    save(AUDIT / 'TRACKER_WEIGHTS_AUDIT.json', dict(passed=True, models=counts,
        face_threshold=.8, landmark_num_modules=2,
        source_manifest_sha256=sha(AUDIT / 'SOURCE_MANIFEST.json')))
    return tracker, get_landmarks


def official_crop(frames, tracker, get_landmarks):
    import cv2
    import numpy as np
    from skimage.transform import warp
    info = collections.defaultdict(list)
    encoded = []
    digest = hashlib.sha256()
    for i, frame in enumerate(frames):
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        boxes = tracker.face_detector(bgr, rgb=False)
        landmarks, scores = tracker.landmark_detector(bgr, boxes, rgb=False)
        info['bbox'].append(boxes)
        info['landmarks'].append(landmarks)
        info['landmarks_scores'].append(scores)
        ok, jpeg = cv2.imencode('.jpg', bgr)  # Same default as demo's imwrite.
        assert ok
        if not (AUDIT / 'JPEG_PARITY.json').exists():
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / 'frame.jpg'
                assert cv2.imwrite(str(path), bgr)
                assert path.read_bytes() == jpeg.tobytes()
            save(AUDIT / 'JPEG_PARITY.json', dict(passed=True, cv2=cv2.__version__,
                imencode_equals_demo_imwrite=True, first_jpeg_sha256=hashlib.sha256(jpeg.tobytes()).hexdigest()))
        encoded.append(jpeg)
        digest.update(jpeg.tobytes())
    landmarks = get_landmarks(info)
    missing = [i for i, value in enumerate(landmarks) if value is None]
    spec = importlib.util.spec_from_file_location('official_spectre_data_utils', SPECTRE / 'datasets/data_utils.py')
    utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(utils)
    filled = utils.landmarks_interpolate(landmarks)
    assert filled is not None, 'No confidently tracked face throughout this video'
    crop = source_crop_function()
    images = []
    for jpeg, points in zip(encoded, filled):
        rgb = cv2.cvtColor(cv2.imdecode(jpeg, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        images.append(warp(rgb, crop(rgb, points, scale=1.6).inverse,
                           output_shape=(224, 224)).transpose(2, 0, 1).astype(np.float32))
    audit = dict(missing_landmark_frames=missing,
        multi_face_frames=sum(len(boxes)>1 for boxes in info['bbox']), num_frames=len(frames),
        frontend='Pinned SPECTRE FaceTracker: RetinaFace ResNet50 threshold.8 and FAN2',
        selection='Official get_landmarks: largest bbox perimeter; min score of landmarks17: >=.2',
        failed_landmarks='Official landmarks_interpolate; no silent frame removal',
        jpeg_sequence_sha256=digest.hexdigest(), jpeg='Demo default OpenCV JPEG imwrite/imread, verified byte-equivalent in memory',
        crop='Official SPECTRE crop_face scale1.6 and224 warp')
    return np.stack(images), np.asarray(filled), audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['probe', 'all'])
    args = parser.parse_args()
    import cv2
    import numpy as np
    import torch
    device = guard(.08, 32768)
    cv2.setNumThreads(1)
    model = Encoder('cuda')
    tracker, get_landmarks = load_frontend()
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    p = read(WORK / 'NATIVE_GENERATION_PROTOCOL.json')
    samples = p['samples']
    if args.action == 'probe':
        samples = [s for s in samples if s['id'] in ['ClNUxPsUByg_000210_000221.0', 'Ew2z_sYABE0_000063_000083.0']]
    else:
        assert read(WORK / 'NATIVE_GENERATION_COMPLETE.json')['clips'] == 43
    contract = dict(schema='official-spectre-demo-frontend-v1', encoder=model.info,
        tracker_source_manifest_sha256=sha(AUDIT / 'SOURCE_MANIFEST.json'),
        tracker_weights_manifest_sha256=sha(AUDIT / 'WEIGHT_MANIFEST.json'),
        script_sha256=sha(__file__), generation_protocol_sha256=sha(WORK / 'NATIVE_GENERATION_PROTOCOL.json'),
        temporal='2frame edge replication,50frame chunks including4context frames;46valid frames',
        frontend='Pinned SPECTRE demo RetinaFaceResNet50/FAN2 and largest confident face',
        jpeg='Detect on decoded frames, crop from default JPEG roundtrip, matching SPECTRE demo',
        precision='FP32; benchmarkFalse deterministicTrue TF32False', fps=25,
        feature_order='pose6 then exp50', full_interval=True, empirical_offset=0,
        AF_private_extractor_code_available=False)
    path = WORK / 'OFFICIAL_SPECTRE_PROTOCOL.json'
    if path.exists():
        assert read(path) == contract
    else:
        save(path, contract)
    save(AUDIT / 'DEVICE.json', device)
    tasks = []
    for sample in samples:
        uid = sample['id']
        for role, video in [('speaker', WORK / 'inputs' / uid / 'speaker/cropped25.mkv'),
                            ('gt', WORK / 'inputs' / uid / 'listener/cropped25.mkv'),
                            ('avatarforcing', WORK / 'native_generated' / uid / 'generated.mp4')]:
            tasks.append(dict(id=uid, role=role, video=str(video), video_sha256=sha(video),
                              frames=round(sample['duration']*25)))
    records = []
    for task in tasks:
        folder = WORK / 'official_features' / task['role']
        folder.mkdir(parents=True, exist_ok=True)
        output = folder / (task['id'] + '.npz')
        receipt = output.with_suffix('.json')
        if receipt.exists():
            r = read(receipt)
            assert r['protocol_sha256'] == sha(path) and r['video_sha256'] == task['video_sha256']
            assert sha(output) == r['sha256']
            records.append(r)
            continue
        started = time.time()
        frames, grid = decode25(task['video'], task['frames'])
        images, landmarks, face_audit = official_crop(frames, tracker, get_landmarks)
        if not (AUDIT / 'ENCODER_PARITY.json').exists():
            save(AUDIT / 'ENCODER_PARITY.json', model.parity(images[:8]))
        values = model.chunks(images, batch=46)
        assert values.shape == (task['frames'], 56) and np.isfinite(values).all()
        np.savez_compressed(output, motion=values, landmarks=landmarks, fps=25,
                            frame_indices=np.arange(len(values)))
        r = dict(**task, path=str(output), sha256=sha(output), protocol_sha256=sha(path),
                 grid=grid, face_audit=face_audit, seconds=time.time()-started)
        save(receipt, r)
        records.append(r)
        save(WORK / 'OFFICIAL_SPECTRE_STATUS.json', dict(stage='official_spectre',
            action=args.action, completed=len(list((WORK / 'official_features').glob('*/*.json'))), total=129,
            last_id=task['id'], last_role=task['role'], pid=__import__('os').getpid()))
        print('Official SPECTRE', len(records), '/', len(tasks), task['role'], task['id'],
              round(time.time()-started, 2), 'seconds', flush=True)
        del frames, images, landmarks, values
    save(WORK / ('OFFICIAL_SPECTRE_PROBE.json' if args.action == 'probe' else 'OFFICIAL_EXTRACTION_COMPLETE.json'),
         dict(passed=True, streams=len(records), records=records, protocol_sha256=sha(path)))


if __name__ == '__main__':
    main()
