"""Train-only BFM73/audio45 cache shared by RLHG and ListenFormer.

Uses the sealed ViCo extractor unchanged. Old DECA56 caches are never treated
as BFM. Per-clip atomic receipts permit safe restart and independent workers.
"""
import argparse
import ast
import concurrent.futures
import fcntl
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
from vico_more_common import ROOT, WORK, DATA, SPLIT, BFM, read, save, sha, gpu_guard, test_samples

OUT = WORK / 'features'


def prepare():
    split = read(SPLIT)
    tasks = []
    for name in ['train', 'validation']:
        for row in split[name]:
            uid = row['uuid']
            old = read(ROOT / 'experiments/common_method_comparison_20261106/l2l_retrained/features' / name / (uid + '.json'))
            tasks.append(dict(id=uid, split=name, frames=old['frames'], inputs=old['inputs']))
    refs = {r['uuid']: r for r in read(ROOT / 'evaluation/vico/index.json')['clips'] if r['split'] == 'test'}
    for sample in test_samples():
        uid = sample['id']
        # Only speaker/initial-listener are inference inputs; full GT remains
        # external and is not materialized by the test input loader.
        tasks.append(dict(id=uid, split='test', frames=sample['streams']['speaker']['frames'],
            inputs=dict(audio=dict(path=refs[uid]['audio'], sha256=sha(refs[uid]['audio'])),
                speaker=sample['streams']['speaker'], listener=sample['streams']['gt'])))
    source = ROOT / 'scripts/p2_vico_motion_extract.py'
    p = dict(split_sha256=sha(SPLIT), tasks=tasks, audio_order='MFCC13,delta13,delta2_13,RMS3,ZCR3',
        dynamic_order='angle3,exp64,trans3,crop3', extractor=str(source), extractor_sha256=sha(source),
        audio_source=str(WORK / 'vendor/rlhg/preprocess/extract.py'),
        compatibility='librosa keyword-only API: rms second positional was S (ignored with y); zcr second was frame_length=sr',
        audio_normalization='fit train only; exact-zero std -> 1; other dimensions unmodified',
        extractor_protocol=str(BFM / 'protocol.json'), extractor_protocol_sha256=sha(BFM / 'protocol.json'))
    dest = OUT / 'protocol.json'
    if dest.exists():
        assert read(dest) == p
    else:
        save(dest, p)
    return p


def raw_audio(path, frames):
    import soundfile as sf
    import librosa
    import torch
    import torchaudio
    audio, sr = sf.read(path)
    if audio.ndim == 2:
        audio = audio.mean(-1)
    samples = int(sr / 30)
    target = samples * frames
    audio = np.pad(audio, (0, max(0, target-len(audio))))[:target]
    features = []
    for i in range(frames):
        chunk = audio[i*samples:(i+1)*samples]
        mfcc = torchaudio.compliance.kaldi.mfcc(torch.from_numpy(chunk).float()[None],
            sample_frequency=sr, use_energy=True).T
        delta = torchaudio.functional.compute_deltas(mfcc)
        delta2 = torchaudio.functional.compute_deltas(delta)
        # Match actual pre-keyword-only librosa semantics, not the misleading call.
        rms = librosa.feature.rms(y=chunk).reshape(-1)
        zcr = librosa.feature.zero_crossing_rate(y=chunk, frame_length=sr).reshape(-1)
        features.append(np.concatenate([np.stack([mfcc.numpy(), delta.numpy(), delta2.numpy()]).reshape(-1), rms, zcr]))
    result = np.stack(features)
    assert result.shape == (frames, 45), (path, sr, result.shape)
    assert np.isfinite(result).all()
    return result


def audio_one(task):
    import torch
    torch.set_num_threads(1)
    path = OUT / task['split'] / task['id'] / 'audio.npy'
    receipt = path.with_suffix('.json')
    if receipt.exists():
        assert sha(path) == read(receipt)['sha256']
        return read(receipt)
    t = time.time()
    source = task['inputs']['audio']
    assert sha(source['path']) == source['sha256']
    a = raw_audio(source['path'], task['frames'])
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.tmp.{os.getpid()}')
    with temp.open('wb') as f:
        np.save(f, a)
    temp.replace(path)
    r = dict(id=task['id'], split=task['split'], path=str(path), sha256=sha(path),
        source=source, frames=len(a), seconds=time.time()-t)
    save(receipt, r)
    return r


def audio_parity(task):
    """Run unchanged official extraction with an explicit legacy librosa adapter."""
    import librosa
    import torch
    import torchaudio
    import soundfile as sf
    source = WORK / 'vendor/rlhg/preprocess/extract.py'
    node = next(n for n in ast.parse(source.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'extract_audio_features')
    sandbox = OUT / 'audio_parity'
    sandbox.mkdir(parents=True, exist_ok=True)
    legacy = SimpleNamespace(feature=SimpleNamespace(
        rms=lambda y, S: librosa.feature.rms(y=y, S=S),
        zero_crossing_rate=lambda y, frame_length: librosa.feature.zero_crossing_rate(y, frame_length=frame_length)))
    env = dict(os=os, osp=os.path, sf=sf, torch=torch, torchaudio=torchaudio, np=np,
        librosa=legacy, loadmat=lambda _p: dict(coeff=np.zeros((task['frames'], 1))))
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), env)
    env['extract_audio_features'](task['inputs']['audio']['path'], 'not_read', str(sandbox))
    expected = np.load(sandbox / (Path(task['inputs']['audio']['path']).stem + '.npy'))
    actual = raw_audio(task['inputs']['audio']['path'], task['frames'])
    assert np.array_equal(expected, actual)
    save(OUT / 'AUDIO_PARITY.json', dict(passed=True, id=task['id'], exact=True, shape=list(actual.shape),
        max_abs=float(np.abs(actual-expected).max()), official_source_sha256=sha(source),
        compatibility='Legacy positional librosa RMS S and ZCR frame_length semantics retained'))


def audio(workers):
    p = prepare()
    audio_parity(p['tasks'][0])
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        for i, result in enumerate(pool.map(audio_one, p['tasks'])):
            if i % 20 == 0:
                print('audio', i+1, len(p['tasks']), result['id'], flush=True)
    save(OUT / 'AUDIO_COMPLETE.json', dict(success=True, tasks=len(p['tasks'])))


def dynamic(arrays):
    c, t = arrays['coeff'], arrays['transform_params']
    return np.concatenate([c[:, 224:227], c[:, 80:144], c[:, 254:257], t[:, -3:]], 1)


def bfm(shard, shards, smoke):
    p = prepare()
    assert sha(p['extractor']) == p['extractor_sha256']
    device = gpu_guard(18)
    sys.path.insert(0, str(ROOT / 'scripts'))
    from p2_vico_motion_extract import Extractor, load_frames
    ext = Extractor()
    ext.torch.set_num_threads(2)
    save(OUT / f'device_{shard}.json', device)
    jobs = [(task, role) for task in p['tasks'] if task['split'] != 'test' for role in ['speaker', 'listener']]
    jobs = jobs[shard::shards]
    if smoke:
        jobs = jobs[:1]
    parity_done = False
    for index, (task, role) in enumerate(jobs):
        dest = OUT / task['split'] / task['id'] / (role + '.npz')
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.with_suffix('.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            receipt = dest.with_suffix('.json')
            if receipt.exists():
                assert sha(dest) == read(receipt)['sha256']
                continue
            t = time.time()
            source = task['inputs'][role]
            assert sha(source['path']) == source['sha256']
            frames = load_frames(source['path'])
            assert len(frames) == task['frames']
            arrays, timing, faces = ext.extract(frames)
            if not parity_done:
                parity = ext.parity(frames, arrays, faces)
                save(OUT / f'BFM_PARITY_{shard}.json', dict(id=task['id'], role=role, **parity))
                parity_done = True
            arrays['dynamic'] = dynamic(arrays)
            temp = dest.with_name(dest.name + f'.tmp.{os.getpid()}')
            with temp.open('wb') as f:
                np.savez_compressed(f, **arrays)
            temp.replace(dest)
            save(receipt, dict(id=task['id'], split=task['split'], role=role, path=str(dest), sha256=sha(dest),
                source=source, frames=len(frames), timing=timing, seconds=time.time()-t,
                missing_faces=int((~arrays['detected']).sum()), protocol_sha256=sha(OUT / 'protocol.json')))
            print('bfm', shard, index+1, len(jobs), task['id'], role, round(time.time()-t, 2), flush=True)
    if not smoke:
        save(OUT / f'BFM_COMPLETE_{shard}.json', dict(success=True, jobs=len(jobs), shard=shard, shards=shards))


def stats():
    import torch
    torch.set_num_threads(2)
    p = prepare()
    sets = {'audio': [], 'dynamic': []}
    rows = []
    for task in p['tasks']:
        if task['split'] == 'test':
            continue
        base = OUT / task['split'] / task['id']
        for name in ['audio', 'speaker', 'listener']:
            suffix = '.npy' if name == 'audio' else '.npz'
            path = base / (name + suffix)
            r = read(path.with_suffix('.json'))
            assert sha(path) == r['sha256']
            rows.append(dict(split=task['split'], id=task['id'], name=name, sha256=r['sha256']))
            if task['split'] == 'train':
                sets['audio' if name == 'audio' else 'dynamic'].append(
                    np.load(path) if name == 'audio' else np.load(path)['dynamic'])
    arrays, meta = {}, {}
    for key, parts in sets.items():
        a = torch.from_numpy(np.concatenate(parts))
        mean, std = a.mean(0).numpy(), a.std(0, correction=1).numpy()
        zero = std == 0
        arrays[key + '_mean'] = mean
        arrays[key + '_std'] = np.where(zero, 1., std)
        meta[key] = dict(frames=len(a), std_correction=1, zero_std_indices=np.where(zero)[0].tolist(),
            smallest_nonzero_std=float(std[~zero].min()), dimension=a.shape[1])
    np.savez(OUT / 'train_stats.npz', **arrays)
    save(OUT / 'COMPLETE.json', dict(success=True, records=rows, train_only=True,
        statistics=meta, stats_sha256=sha(OUT / 'train_stats.npz'), protocol_sha256=sha(OUT / 'protocol.json')))


def test_inputs():
    """Separate sealed t0 artifacts prevent the generator opening any GT sequence."""
    records=[]
    for sample in test_samples():
        uid=sample['id']
        base=OUT/'test'/uid;base.mkdir(parents=True,exist_ok=True)
        origin=BFM/'features/gt'/(uid+'.npz')
        with np.load(origin) as gt:
            coeff=gt['coeff'][:1].copy();transform=gt['transform_params'][:1].copy()
        path=base/'initial_listener.npz'
        np.savez(path,coeff=coeff,transform_params=transform)
        speaker=BFM/'features/speaker'/(uid+'.npz')
        records.append(dict(id=uid,initial=str(path),initial_sha256=sha(path),
            speaker=str(speaker),speaker_sha256=sha(speaker),listener_frames=1))
    save(OUT/'TEST_INPUTS.json',dict(source_only=True,records=records))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'audio', 'bfm', 'stats','test_inputs'])
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=1)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.stage == 'prepare':
        prepare()
    elif args.stage == 'audio':
        audio(args.workers)
    elif args.stage == 'bfm':
        bfm(args.shard, args.shards, args.smoke)
    elif args.stage=='test_inputs':
        test_inputs()
    else:
        stats()
