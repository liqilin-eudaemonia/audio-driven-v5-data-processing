"""Official DualTalk checkpoint, core demo parity, and self-zero ViCo motion inference."""
import argparse
import ast
import math
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
from vico_more_common import ROOT, WORK, DECA, read, save, sha, gpu_guard, test_samples

OUT = WORK / 'dualtalk'
VENDOR = WORK / 'vendor/dualtalk'
WEIGHT = WORK / 'assets/dualtalk/DualTalk.pth'


def construct():
    import torch
    from transformers import Wav2Vec2Config
    sys.path.insert(0, str(VENDOR))
    import wav2vec
    from DualTalk import DualTalkModel
    # The official full state includes both encoders. Instantiate their exact
    # config locally, then require complete strict coverage; no random weights survive.
    original = wav2vec.Wav2Vec2Model.from_pretrained
    wav2vec.Wav2Vec2Model.from_pretrained = classmethod(lambda cls, *_a, **_k:
        cls(Wav2Vec2Config.from_pretrained(str(WORK / 'assets/wav2vec'), local_files_only=True)))
    try:
        model = DualTalkModel(SimpleNamespace(blendshape_dim=56, feature_dim=256))
    finally:
        wav2vec.Wav2Vec2Model.from_pretrained = original
    state = torch.load(WEIGHT, map_location='cpu', weights_only=False)
    report = model.load_state_dict(state, strict=True)
    assert not report.missing_keys and not report.unexpected_keys
    # Torch2.10's transformers migration renames legacy weight_norm g/v.
    mapped = {k.replace('.weight_g', '.parametrizations.weight.original0').replace(
        '.weight_v', '.parametrizations.weight.original1'): v for k, v in state.items()}
    assert set(mapped) == set(model.state_dict())
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in mapped.items())
    save(OUT / 'WEIGHT_AUDIT.json', dict(strict=True, tensor_exact=True, keys=len(state),
        parameters=sum(p.numel() for p in model.parameters()), checkpoint_sha256=sha(WEIGHT),
        config_sha256=sha(WORK / 'assets/wav2vec/config.json'),
        code={str(VENDOR / x): sha(VENDOR / x) for x in ['DualTalk.py', 'wav2vec.py', 'demo.py']},
        constructor_adaptation='Local identical wav2vec config; complete checkpoint replaces every state tensor',
        weight_norm_migration='legacy g/v -> parametrizations original0/original1; all values tensor-exact'))
    return model


def processor():
    from transformers import Wav2Vec2Processor
    return Wav2Vec2Processor.from_pretrained(str(WORK / 'assets/wav2vec'), local_files_only=True)


def inputs(proc, a1, a2, blendshape):
    import torch
    audio = [torch.from_numpy(np.asarray(proc(a, sampling_rate=16000).input_values[0], dtype=np.float32)).cuda()[None]
             for a in [a1, a2]]
    return (*audio, torch.from_numpy(blendshape.astype(np.float32)).cuda()[None])


def parity(model):
    """Execute the original AST function, avoiding its unrelated import-time FLAME renderer."""
    import torch
    import librosa
    import soundfile as sf
    proc = processor()
    args = SimpleNamespace(result_path=str(OUT / 'official_demo'), save_path=str(WEIGHT.parent),
        device='cuda:0', audio1_path=str(VENDOR / 'demo/xkHwlcDSOjc_sub_video_109_000_speaker2.wav'),
        audio2_path=str(VENDOR / 'demo/xkHwlcDSOjc_sub_video_109_000_speaker1.wav'),
        bs2_path=str(VENDOR / 'demo/xkHwlcDSOjc_sub_video_109_000_speaker1.npz'))
    node = next(n for n in ast.parse((VENDOR / 'demo.py').read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'test')
    env = dict(torch=torch, os=os, np=np, librosa=librosa, time=time,
        Wav2Vec2Processor=SimpleNamespace(from_pretrained=lambda *_a, **_k: proc))
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(VENDOR / 'demo.py'), 'exec'), env)
    env['test'](args, model)
    original = np.load(OUT / 'official_demo' / (Path(args.audio1_path).stem + '.npy'))
    z = np.load(args.bs2_path)
    b = np.concatenate([z['exp'], z['pose'][:, 3:], z['pose'][:, :3]], axis=1)
    a1 = librosa.load(args.audio1_path, sr=16000)[0]
    a2 = librosa.load(args.audio2_path, sr=16000)[0]
    with torch.inference_mode():
        adapted = model(*inputs(proc, a1, a2, b)).cpu().numpy()[0]
    delta = float(np.abs(original - adapted).max())
    assert delta < 1e-6, delta
    save(OUT / 'DEMO_PARITY.json', dict(passed=True, max_abs=delta, frames=len(original),
        official_function='demo.py:test executed unchanged via AST',
        waveform_loading='official librosa16k + Wav2Vec2Processor', source_only=True))


def generate(model):
    import torch
    sys.path.insert(0, str(ROOT / 'scripts'))
    from p2_vico256_combined_eval import source_wave
    from vico_common_deca import clip_metrics
    proc = processor()
    records = []
    items = {s['id']: s for s in read(ROOT / 'experiments/p2_vico256_combined8000_20261106/shards/00.json')}
    old_complete = read(DECA / 'DECA_COMPLETE.json')
    receipts = {(r['id'], r['role']): r for r in old_complete['records']}
    for sample in test_samples():
        uid = sample['id']
        dest = OUT / 'predictions' / (uid + '.npz')
        if dest.with_suffix('.json').exists():
            rec = read(dest.with_suffix('.json'))
            assert sha(dest) == rec['sha256']
            records.append(rec)
            continue
        t = time.time()
        speaker_path = DECA / 'features/speaker' / (uid + '.npz')
        assert sha(speaker_path) == receipts[uid, 'speaker']['feature_sha256']
        speaker = np.load(speaker_path)['motion']
        n = len(speaker)
        wave, audit = source_wave(items[uid], ROOT / 'experiments/p2_vico256_combined8000_20261106/conditions/vico' / uid)
        count = math.ceil(len(wave) / 16000 * 25)
        indices = np.minimum(np.floor(np.arange(count) * 30 / 25 + .5).astype(int), n - 1)
        blend = speaker[indices][:, np.r_[0:50, 53:56, 50:53]]
        with torch.inference_mode():
            pred = model(*inputs(proc, np.zeros_like(wave), wave, blend)).cpu().numpy()[0]
        assert pred.shape == (count, 56) and np.isfinite(pred).all()
        score_idx = np.minimum(np.floor(np.arange(n) * 25 / 30 + .5).astype(int), count - 1)
        motion = pred[score_idx][:, np.r_[0:50, 53:56, 50:53]]
        dest.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dest, native_exp_jaw_neck=pred, motion=motion, source_indices=indices,
                            scoring_indices=score_idx, native_fps=25, scoring_fps=30)
        rec = dict(id=uid, sha256=sha(dest), path=str(dest), native_frames=count, score_frames=n,
            speaker_feature_sha256=sha(speaker_path), input_audit=audit,
            self_audio='all-zero waveform before official processor',
            listener_future_input=False, elapsed_s=time.time()-t)
        save(dest.with_suffix('.json'), rec)
        records.append(rec)
        save(OUT / 'status.json', dict(stage='inference', complete=len(records), total=43))
        print(uid, count, round(time.time()-t, 2), flush=True)
    # Scoring is separate from model inputs and starts after all inference.
    metrics = []
    for rec in records:
        uid = rec['id']
        gt_path = DECA / 'features/gt' / (uid + '.npz')
        assert sha(gt_path) == receipts[uid, 'gt']['feature_sha256']
        gt = np.load(gt_path)['motion']
        speaker = np.load(DECA / 'features/speaker' / (uid + '.npz'))['motion']
        pred = np.load(rec['path'])['motion']
        metrics.append(dict(id=uid, **clip_metrics(speaker, gt, pred)))
    save(OUT / 'METRICS.json', dict(method='DualTalk official checkpoint, self-zero transfer',
        readout='native DECA coefficients; RGB metrics N/A', records=metrics))
    save(OUT / 'COMPLETE.json', dict(success=True, outputs=43, records=records,
        weight_audit_sha256=sha(OUT / 'WEIGHT_AUDIT.json'), demo_parity_sha256=sha(OUT / 'DEMO_PARITY.json'),
        metrics_sha256=sha(OUT / 'METRICS.json')))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['audit', 'run'])
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    model = construct()
    if a.stage == 'audit':
        return
    device = gpu_guard(16)
    save(OUT / 'device.json', device)
    import torch
    torch.manual_seed(137)
    model = model.eval().cuda()
    parity(model)
    generate(model)


if __name__ == '__main__':
    main()
