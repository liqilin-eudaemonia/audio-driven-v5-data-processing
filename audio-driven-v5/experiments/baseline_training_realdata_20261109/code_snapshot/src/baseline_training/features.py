"""Genuine native features from frozen final-crop media; no fake labels."""
import gc
import importlib
import json
import os
import fcntl
from pathlib import Path
import subprocess
import sys

import numpy as np

from .data import ROOT, WORK, SelectedPool, read, save, sha


def backend(device):
    import torch
    torch.set_num_threads(2)
    if device.startswith('cuda'):
        # Short validation work shares a busy GPU without taking its remaining
        # memory. Never stop another process or enlarge this cap automatically.
        torch.cuda.set_per_process_memory_fraction(24 * 1024**3 / torch.cuda.get_device_properties(0).total_memory)
        assert torch.cuda.mem_get_info()[0] > 26 * 1024**3, 'Insufficient headroom for feature validation'
    return torch


def ids_for(work, ids):
    p = read(Path(work) / 'DATA_PROTOCOL.json')
    return ids or p['diagnostic_train'] + p['diagnostic_validation']


def native_glare():
    sys.path.insert(0, str(ROOT / 'related/glare'))
    from float.options.train_options import TrainOptions
    opt = TrainOptions().parse(['--lia_version', 'LIA', '--num_ref_frames', '1', '--num_prev_frames', '0',
                                '--use_alpha_mean', '--use_alpha_std', '--use_speed_mean', '--use_speed_std',
                                '--use_accel_mean', '--use_accel_std'])
    opt.rank = 'cuda:0'
    opt.device = 'cuda:0'
    # The author's filenames are swapped relative to their contents: all212
    # motion_autoencoder tensors actually live in flow_transformer.pth.
    opt.lia_path = str(ROOT / 'compare/AvatarForcing/pretrained_dir/flow_transformer.pth')
    opt.wav2vec_model_path = str(ROOT / 'compare/AvatarForcing/pretrained_dir/wav2vec2-base-960h')
    opt.audio2emotion_path = str(ROOT / 'compare/deployment/assets/wav2vec-english-speech-emotion-recognition')
    opt.qwen2audio_path = '/private/ckpts/Qwen2-Audio-7B-Instruct'
    return opt


def detector(sid, work):
    directory = Path(work) / 'media' / sid
    output = Path(work) / 'reaction' / sid
    args = [str(ROOT / '.venv-p2-glare-detector/bin/python'), '-u', '-c',
            'import sys;sys.path.insert(0,' + repr(str(ROOT / 'src')) + ');'
            'from baseline_training.reactions import extract;from p2_glare_eval.common import read;'
            'extract(' + repr(str(directory / 'listener.mp4')) + ',' + repr(str(output)) + ','
            'read(' + repr(str(ROOT / 'configs/p2_glare_detector_v1.json')) + '))']
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'detector.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with (output / 'run.log').open('w') as log:
            environment = dict(os.environ, PYTHONPATH=str(ROOT / 'src'), CUDA_VISIBLE_DEVICES='')
            subprocess.run(args, stdout=log, stderr=subprocess.STDOUT, check=True, env=environment)


def export_glare(work, ids, device):
    import soundfile as sf
    import csv
    torch = backend(device)
    opt = native_glare()
    from float.models.generator import Generator
    from float.models.intensity_encoder import IntensityEncoder
    from transformers import Wav2Vec2FeatureExtractor
    audio_processor = Wav2Vec2FeatureExtractor.from_pretrained(opt.wav2vec_model_path, local_files_only=True)
    model = Generator(size=512, style_dim=512, motion_dim=20)
    origin = torch.load(opt.lia_path, map_location='cpu', weights_only=True)
    weights = {k.removeprefix('motion_autoencoder.'): v for k, v in origin.items() if k.startswith('motion_autoencoder.')}
    model.load_state_dict(weights, strict=True)
    assert all(torch.equal(model.state_dict()[k], v) for k, v in weights.items())
    model.eval().to(device).requires_grad_(False)
    del origin, weights
    asset = dict(motion_checkpoint=opt.lia_path, motion_sha256=sha(opt.lia_path),
                  encoder_sha256=sha(ROOT / 'related/glare/float/models/encoder.py'),
                  motion_load='all motion_autoencoder tensors, strict and exact equality',
                  native_dim=512, fps=25)
    save(Path(work) / 'GLARE_MOTION_ASSET.json', asset)
    for sid in ids:
        path = Path(work) / 'features/glare' / (sid + '.pt')
        if path.exists():
            continue
        frames = np.load(Path(work) / 'media' / sid / 'listener.npy', mmap_mode='r')
        values = []
        with torch.inference_mode():
            for start in range(0, len(frames), 4):
                image = torch.from_numpy(np.array(frames[start:start+4])).permute(0, 3, 1, 2).float().to(device) / 127.5 - 1
                appearance, _, _ = model.enc(image, input_target=None)
                motion = model.dec.direction(model.enc.fc(appearance))
                values.append(motion.cpu())
        motion = torch.cat(values)
        assert motion.shape == (len(frames), 512) and torch.isfinite(motion).all()
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(motion=motion, direction=model.dec.direction.weight.detach().cpu()), path)
        print('lia', sid, len(frames), flush=True)
    del model
    gc.collect()
    if device.startswith('cuda'):
        torch.cuda.empty_cache()
    # Run the unchanged Qwen2-Audio intensity encoder, not a caption-to-score
    # heuristic. Keep generated text as provenance for every frame vector.
    need_intensity = any('intensity' not in torch.load(Path(work) / 'features/glare' / (sid + '.pt'),
                                                      map_location='cpu', weights_only=True) for sid in ids)
    intensity = IntensityEncoder(opt, cuda=device.startswith('cuda')) if need_intensity else None
    import inspect
    if intensity is not None and 'audios' in inspect.signature(intensity.processor.__call__).parameters:
        class ProcessorCompatibility:
            def __init__(self, original):
                self.original = original

            def __getattr__(self, name):
                return getattr(self.original, name)

            def __call__(self, **kwargs):
                if 'audio' in kwargs:
                    kwargs['audios'] = kwargs.pop('audio')
                return self.original(**kwargs)
        intensity.processor = ProcessorCompatibility(intensity.processor)
    for sid in ids:
        path = Path(work) / 'features/glare' / (sid + '.pt')
        value = torch.load(path, map_location='cpu', weights_only=True)
        audio, sr = sf.read(Path(work) / 'media' / sid / 'speaker.wav', dtype='float32')
        text = intensity.infer_text(audio, sampling_rate=sr) if 'intensity' not in value else value['intensity_text']
        score = (intensity.text_to_frame_vector(text, len(value['motion']), fps=25)
                 if 'intensity' not in value else value['intensity'])
        value['intensity'] = score
        value['audio'] = torch.from_numpy(np.asarray(
            audio_processor(audio, sampling_rate=sr).input_values[0], dtype=np.float32))
        value['intensity_text'] = text
        torch.save(value, path)
        if intensity is not None:
            save(path.with_suffix('.intensity.json'), dict(text=text, model_path=opt.qwen2audio_path,
                  prompt=intensity.prompt, samples=len(audio), rate=sr, frames=len(score)))
        print('intensity', sid, text, flush=True)
    del intensity
    gc.collect()
    if device.startswith('cuda'):
        torch.cuda.empty_cache()
    for sid in ids:
        detector(sid, work)
        from p2_glare_eval.upstream import REACTIONS
        with (Path(work) / 'reaction' / sid / 'scores.csv').open() as f:
            rows = list(csv.DictReader(f))
        scores = torch.tensor([[float(row[name]) for name in REACTIONS] for row in rows])
        path = Path(work) / 'features/glare' / (sid + '.pt')
        value = torch.load(path, map_location='cpu', weights_only=True)
        assert scores.shape == (len(value['motion']), 6)
        value['reaction'] = scores
        value['reaction_valid'] = torch.from_numpy(np.load(Path(work) / 'reaction' / sid / 'training_valid_mask.npy'))
        torch.save(value, path)
        receipt(path, sid, work, method='glare', extra=dict(reaction_columns=list(REACTIONS),
                reaction_csv_sha256=sha(Path(work) / 'reaction' / sid / 'scores.csv'),
                intensity_source='unchanged Qwen2-Audio intensity encoder', motion_asset=asset))


def receipt(path, sid, work, method, extra=None):
    source = read(Path(work) / 'media' / sid / 'receipt.json')
    save(path.with_suffix('.json'), dict(method=method, sample_id=sid, feature_sha256=sha(path),
          media_receipt_sha256=sha(Path(work) / 'media' / sid / 'receipt.json'),
          caption=source['contract']['caption'], caption_sha256=source['contract']['caption_sha256'],
          group_id=source['contract']['group_id'], **(extra or {})))


def export_bfm(work, ids, device):
    import torch
    import soundfile as sf
    backend(device)
    sys.path.insert(0, str(ROOT / 'scripts'))
    from p2_vico_motion_extract import Extractor
    extractor = Extractor()
    sys.path.insert(0, str(ROOT / 'compare/deployment'))
    from vico_more_features import raw_audio, dynamic
    for sid in ids:
        base = Path(work) / 'media' / sid
        dest = Path(work) / 'features/bfm' / (sid + '.pt')
        if dest.with_suffix('.json').exists():
            from .engine import check_features
            check_features(work, 'bfm', sid)
            continue
        media = read(base / 'receipt.json')
        listener_frames = np.load(base / 'listener30.npy')
        speaker_frames = np.load(base / 'speaker30.npy')
        listener_pts, speaker_pts = media['listener_pts30'], media['speaker_pts30']
        audio = raw_audio(str(base / 'speaker_bfm.wav'), len(listener_frames))
        values = {'audio': torch.from_numpy(audio)}
        for role in ['listener', 'speaker']:
            frames = listener_frames if role == 'listener' else speaker_frames
            arrays, timing, _ = extractor.extract(frames)
            values[role] = torch.from_numpy(dynamic(arrays)).float()
            values[role + '_detected'] = torch.from_numpy(arrays['detected'])
            values[role + '_landmarks'] = torch.from_numpy(arrays['landmarks'])
        dest.parent.mkdir(parents=True, exist_ok=True)
        torch.save(values, dest)
        receipt(dest, sid, work, 'bfm', dict(dimensions=dict(audio=45, motion=73),
            audio_extractor=str(ROOT / 'compare/deployment/vico_more_features.py'),
            audio_native_fps=30, visual_fps=30, listener_pts=listener_pts, speaker_pts=speaker_pts,
            final_listener_geometry_preserved=True))
        print('bfm', sid, flush=True)


def export(method, work=WORK, ids=None, device='cuda:0'):
    ids = ids_for(work, ids)
    if method == 'glare':
        return export_glare(work, ids, device)
    if method in ['listenformer', 'vico']:
        return export_bfm(work, ids, device)
    if method in ['dystream', 'avatarforcing']:
        from .streaming import export as stream_export
        return stream_export(method, work, ids, device)
    from .motion56 import export as motion_export
    return motion_export(method, work, ids, device)
