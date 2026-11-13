"""Leakage-isolated, two-stage L2L training feasibility check on ViCo train only.

This bounded pilot is deliberately not a converged benchmark result. Official
VQ/predictor architectures, loss functions, normalization and optimizers are
reused. It never opens official test/OOD media or selects a person checkpoint.
"""
import argparse
import concurrent.futures
import csv
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import types

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from vico_common_deca import crop_frame, encode, extractor, save, sha, read

WORK = ROOT / 'experiments/common_method_comparison_20261106/l2l_training_pilot'
DATA = Path('/public/0_DATA/2_Avatar/ViCo/dataset')
REPO = ROOT / 'compare/learning2listen/src'
SEED = 137


def source_id(row):
    return row['audio'].rsplit('_', 2)[0]


def prepare():
    rows = list(csv.DictReader((DATA / 'metadata.csv').open()))
    original_train = [r for r in rows if r['data_split'] == 'train']
    test = [r for r in rows if r['data_split'] == 'test']
    ood = [r for r in rows if r['data_split'] == 'ood']
    def interval(r):
        s, a, b = r['audio'].rsplit('_', 2)
        return s, int(a), int(b)
    quarantined, eligible = [], []
    for row in original_train:
        source, start, end = interval(row)
        conflicts = [r['uuid'] for r in test + ood if interval(r)[0] == source and max(start, interval(r)[1]) < min(end, interval(r)[2])]
        if conflicts:
            quarantined.append(dict(row=row, conflicting_held_out_ids=conflicts, reason='source speaker time interval overlaps held-out media'))
        else:
            eligible.append(row)
    sources = sorted({source_id(r) for r in eligible})
    random.Random(SEED).shuffle(sources)
    validation_sources = set(sources[:max(1, round(len(sources) * .15))])
    train = [r for r in eligible if source_id(r) not in validation_sources]
    val = [r for r in eligible if source_id(r) in validation_sources]
    assert len(train) + len(val) + len(quarantined) == 398 and len(test) == 43 and len(ood) == 42
    assert not {source_id(r) for r in train} & {source_id(r) for r in val}
    assert not {r['uuid'] for r in train + val} & {r['uuid'] for r in test + ood}
    assert not {r['audio'] for r in train + val} & {r['audio'] for r in test + ood}
    def subset(items, count):
        selected, seen = [], set()
        for r in sorted(items, key=lambda r: r['uuid']):
            times = r['audio'].rsplit('_', 2)
            if int(times[-1]) - int(times[-2]) < 3 or source_id(r) in seen:
                continue
            selected.append(r)
            seen.add(source_id(r))
            if len(selected) == count:
                break
        assert len(selected) == count
        return selected
    pilot = dict(train=subset(train, 8), validation=subset(val, 4))
    manifest = dict(schema='vico-l2l-source-grouped-split-v1', seed=SEED,
        metadata=str(DATA / 'metadata.csv'), metadata_sha256=sha(DATA / 'metadata.csv'),
        train=train, validation=val, held_out_test_ids=[r['uuid'] for r in test], held_out_ood_ids=[r['uuid'] for r in ood],
        quarantined_training_rows=quarantined,
        validation_source_ids=sorted(validation_sources), pilot=pilot,
        policy='Start from original train398; quarantine source-time overlaps with test/OOD; validation15% of eligible source conversations, seed137; split before windowing; no test/OOD media opened.',
        original_train_test_source_overlap=sorted({source_id(r) for r in original_train} & {source_id(r) for r in test}),
        initialization='from scratch; official default Conan configuration defines architecture only; no person checkpoint selection',
        pilot_scope='8 train conversations +4 validation conversations; first64 frames per pair; 100 VQ steps then100 predictor steps',
        final_benchmark=False)
    if (WORK / 'split.json').exists():
        assert read(WORK / 'split.json') == manifest
    else:
        save(WORK / 'split.json', manifest)
    return manifest


def first_frames(path, count=64):
    import cv2
    cap = cv2.VideoCapture(str(path))
    assert cap.isOpened(), path
    assert abs(cap.get(cv2.CAP_PROP_FPS) - 30) < 1e-6
    frames = []
    try:
        for _ in range(count):
            ok, frame = cap.read()
            assert ok, f'Short pilot video: {path}'
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()
    return np.stack(frames)


def features():
    import torch
    import face_alignment
    import librosa
    from PIL import Image
    m = prepare()
    assert not (ROOT / 'experiments/common_method_comparison_20261106/vico_deca/LAUNCH.json').exists() or (ROOT / 'experiments/common_method_comparison_20261106/vico_deca/COMPLETE.json').exists(), 'Wait for common DECA job before pilot GPU work'
    encoder = extractor()
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device='cuda', flip_input=False)
    for split, items in m['pilot'].items():
        for item in items:
            uid = item['uuid']
            dest = WORK / 'features' / split / (uid + '.npz')
            if dest.with_suffix('.json').exists():
                assert sha(dest) == read(dest.with_suffix('.json'))['feature_sha256']
                continue
            result, paths, missing = {}, {}, {}
            for role in ['speaker', 'listener']:
                path = DATA / 'by_split/train/videos' / (uid + '.' + role + '.mp4')
                # This is the only media root opened by this pilot.
                assert path.parent == DATA / 'by_split/train/videos'
                paths[role] = dict(path=str(path), sha256=sha(path))
                frames = first_frames(path)
                with torch.no_grad():
                    tensor = torch.from_numpy(frames.transpose(0, 3, 1, 2).copy()).float().cuda()
                    faces = fa.face_detector.detect_from_batch(tensor)
                    landmarks = [fa.get_landmarks_from_image(frame, detected_faces=face[:1])[0] if len(face) else None for frame, face in zip(frames, faces)]
                valid = [i for i, lm in enumerate(landmarks) if lm is not None]
                assert valid, f'No face in pilot: {uid}/{role}'
                missing[role] = len(frames) - len(valid)
                previous = landmarks[valid[0]]
                for i, lm in enumerate(landmarks):
                    if lm is not None:
                        previous = lm
                    else:
                        landmarks[i] = previous
                with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
                    cropped = list(pool.map(crop_frame, zip(frames, landmarks)))
                coeff = encode(encoder, cropped)
                result[role] = np.concatenate([coeff[:, 150:200], coeff[:, 200:206]], axis=1)
            # ViCo videos have no audio track; metadata.audio maps to the WAV.
            audio_path=DATA/'audios'/(item['audio']+'.wav')
            paths['audio']=dict(path=str(audio_path),sha256=sha(audio_path))
            audio = subprocess.check_output(['ffmpeg', '-v', 'error', '-threads', '1', '-i', str(audio_path),
                '-t', str(64/30), '-vn', '-ar', '16000', '-ac', '1', '-f', 'f32le', 'pipe:1'])
            waveform = np.frombuffer(audio, dtype='<f4')
            assert len(waveform) >= int(16000 * 63/30), 'Pilot speaker audio missing/short'
            mel = librosa.feature.melspectrogram(y=waveform, sr=16000, hop_length=160, win_length=400, n_fft=512, n_mels=128)
            result['audio'] = np.array(Image.fromarray(mel).resize((256, 128), Image.Resampling.LANCZOS)).T.astype(np.float32)
            assert all(np.isfinite(x).all() for x in result.values())
            dest.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(dest, **result)
            save(dest.with_suffix('.json'), dict(id=uid, split=split, frames=64, files=paths,
                feature_sha256=sha(dest), split_sha256=sha(WORK / 'split.json'), missing_face_frames=missing))
            print('pilot features', split, uid, flush=True)
    save(WORK / 'FEATURES_COMPLETE.json', dict(train_pairs=8, validation_pairs=4, frames_per_pair=64, official_test_media_opened=False))


def train(steps=100):
    import torch
    from torch import nn
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '7'
    # Validate the exact host/GPU identity before using .cuda() in upstream code.
    probe = extractor()
    del probe
    torch.cuda.empty_cache()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.set_num_threads(2)
    sys.path.insert(0, str(REPO))
    from modules.fact_model import setup_model, calc_logit_loss
    from vqgan.vqmodules.gan_models import setup_vq_transformer, calc_vq_loss
    from utils.load_utils import bilateral_filter, mean_std_swap, create_data_vq
    m = read(WORK / 'split.json')
    assert (WORK / 'FEATURES_COMPLETE.json').exists()
    arrays = {}
    for split, items in m['pilot'].items():
        loaded = [np.load(WORK / 'features' / split / (r['uuid'] + '.npz')) for r in items]
        arrays[split] = {key: np.stack([row[key] for row in loaded]).astype(np.float32) for key in ['speaker', 'listener', 'audio']}
        for key in ['speaker', 'listener']:
            arrays[split][key] = bilateral_filter(arrays[split][key])
    stats = {}
    for key in ['speaker', 'listener', 'audio']:
        mean, std = mean_std_swap(arrays['train'][key])
        assert np.isfinite(mean).all() and np.isfinite(std).all() and (std > 0).all()
        stats[key + '_mean'], stats[key + '_std'] = mean, std
        for split in arrays:
            arrays[split][key] = ((arrays[split][key] - mean) / std).astype(np.float32)
    np.savez_compressed(WORK / 'train_only_stats.npz', **stats)
    config = read(REPO / 'configs/vq/delta_v6.json')
    vqconfig = read(REPO / config['l_vqconfig'])
    args = types.SimpleNamespace()
    vq, opt, _ = setup_vq_transformer(args, vqconfig, load_path=None)
    def vq_data(split):
        y = arrays[split]['listener']
        return np.concatenate([y[:, :32], y[:, 32:]], axis=0)
    train_y = torch.from_numpy(vq_data('train')).cuda()
    val_y = torch.from_numpy(vq_data('validation')).cuda()
    history, best = [], float('inf')
    started = time.time()
    for step in range(steps):
        vq.train()
        pred, quant = vq(train_y, None)
        loss = calc_vq_loss(pred, train_y, quant)
        assert torch.isfinite(loss)
        opt.zero_grad()
        loss.backward()
        assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0 for p in vq.parameters())
        opt.step_and_update_lr()
        if step % 10 == 0 or step == steps - 1:
            vq.eval()
            with torch.no_grad():
                vp, vq_loss = vq(val_y, None)
                value = calc_vq_loss(vp, val_y, vq_loss).item()
            assert np.isfinite(value)
            history.append(dict(stage='vq', step=step+1, train_loss=loss.item(), validation_loss=value))
            if value < best:
                best = value
                torch.save(dict(state_dict=vq.state_dict(), step=step+1, validation_loss=value), WORK / 'pilot_vq_best.pth')
            print('pilot VQ', history[-1], flush=True)
    vq.load_state_dict(torch.load(WORK / 'pilot_vq_best.pth', weights_only=False)['state_dict'], strict=True)
    vq.eval()
    for p in vq.parameters():
        p.requires_grad_(False)
    generator, opt, _ = setup_model(config, vqconfig, load_path=None)
    inputs = {}
    for split in arrays:
        inputs[split] = create_data_vq(vq, arrays[split]['speaker'], arrays[split]['listener'], arrays[split]['audio'], 32, patch_size=8)
    best = float('inf')
    for step in range(steps):
        generator.train()
        x, future, _, _ = inputs['train']
        pred = generator(x, 4, -1)
        loss = calc_logit_loss(pred[:, :future.shape[1]], future)
        assert torch.isfinite(loss)
        opt.zero_grad()
        loss.backward()
        assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0 for p in generator.parameters())
        opt.step_and_update_lr()
        if step % 10 == 0 or step == steps - 1:
            generator.eval()
            x, future, _, _ = inputs['validation']
            # mask_index0 fully masks listener history at validation time.
            with torch.no_grad():
                val = generator(x, 4, 0)
                value = calc_logit_loss(val[:, :future.shape[1]], future).item()
            assert np.isfinite(value)
            history.append(dict(stage='predictor', step=step+1, train_loss=loss.item(), validation_loss=value))
            if value < best:
                best = value
                torch.save(dict(state_dict=generator.state_dict(), step=step+1, validation_loss=value), WORK / 'pilot_predictor_best.pth')
            print('pilot predictor', history[-1], flush=True)
    save(WORK / 'RESULTS.json', dict(status='feasibility_passed', final_benchmark=False, converged=False,
        official_test_evaluated=False, initialization='from scratch, official default architecture',
        train_pairs=8, validation_pairs=4, vq_steps=steps, predictor_steps=steps,
        stats_fit_scope='train8 only', validation_listener_history_masked=True,
        train_validation_source_disjoint=True, seconds=time.time()-started, history=history,
        split_sha256=sha(WORK / 'split.json'), stats_sha256=sha(WORK / 'train_only_stats.npz'),
        implementation_sha256=sha(__file__),
        next='Extract all train/validation pairs, train to validation convergence, then run a separately frozen test evaluation. Pilot checkpoints must not be reported as final L2L baseline.'))
    save(WORK / 'COMPLETE.json', dict(status='feasibility_passed', final_benchmark=False, results_sha256=sha(WORK / 'RESULTS.json')))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=['prepare', 'features', 'train', 'run'])
    p.add_argument('--steps', type=int, default=100)
    a = p.parse_args()
    if a.command == 'run':
        features()
        train(a.steps)
    elif a.command == 'train':
        train(a.steps)
    else:
        globals()[a.command]()
