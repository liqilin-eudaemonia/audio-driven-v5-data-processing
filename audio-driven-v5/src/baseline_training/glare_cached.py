"""Sealed full-pair caches -> native FLOAT.forward, without frozen encoders.

Uses the existing first-reference/no-history data contract, NOT arbitrary random
subwindow equivalence. Native losses/flow sampling stay in the upstream forward.
"""
import hashlib
import json
import os
from pathlib import Path
from types import MethodType

import numpy as np
import torch
from torch.utils.data import Dataset


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


class Registry(Dataset):
    """Byte offsets only in RAM. Immutable sealed registry, lazy payload loading."""
    def __init__(self, latest, split='train', verify_payloads=False):
        latest = json.loads(Path(latest).read_text())
        self.root = Path(latest['export'])
        assert sha(self.root / 'SEAL.json') == latest['seal_sha256']
        self.seal = json.loads((self.root / 'SEAL.json').read_text())
        self.summary = json.loads((self.root / 'SUMMARY.json').read_text())
        assert self.summary['cache_registry_complete'] and self.summary['unavailable'] == 0
        assert sha(self.root / 'SUMMARY.json') == self.seal['files']['SUMMARY.json']
        assert split in ('train', 'dev')
        self.path = self.root / (split + '.jsonl')
        assert sha(self.path) == self.seal['files'][self.path.name]
        self.identity = latest['seal_sha256']
        self.verify_payloads = verify_payloads
        self.offsets = []
        with self.path.open('rb') as f:
            while True:
                offset = f.tell()
                if not f.readline(): break
                self.offsets.append(offset)
        assert len(self.offsets) == self.summary['counts'][split]

    def __len__(self): return len(self.offsets)

    def __getitem__(self, index):
        with self.path.open('rb') as f:
            f.seek(self.offsets[index]); row = json.loads(f.readline())
        return load_row(row, self.verify_payloads)


def load_row(row, verify_payloads=False):
    c = row['glare_caches']
    def path(kind):
        d = c[kind]; p = Path(d['payload'])
        assert p.is_file() and p.stat().st_size
        assert sha(d['receipt']) == d['receipt_sha256'], 'Changed receipt: ' + kind
        if verify_payloads: assert sha(p) == d['payload_sha256'], 'Changed payload: ' + kind
        return p
    def tensor(kind): return torch.load(path(kind), map_location='cpu', weights_only=True)
    motion = tensor('lia')['motion'].float()
    audio = tensor('audio')['audio'].float()
    frozen = tensor('frozen_audio')
    stats = tensor('statistics')
    with np.load(path('reaction'), allow_pickle=False) as r:
        reaction = torch.from_numpy(r['reaction'].copy()).float()
        valid = torch.from_numpy(r['reaction_valid'].copy()).bool()
    with np.load(path('prosody'), allow_pickle=False) as p:
        intensity = torch.from_numpy(p['intensity'].copy()).float()
        assert p['intensity_valid'].all(), 'Official policy expects complete zero defaults'
    n = len(motion)
    assert c['view']['reference_frame'] == 0 and c['view']['history_frames'] == 0
    assert c['view']['frames'] == n and c['view']['target_frames'] == n - 1
    assert motion.shape == (n, 512) and audio.shape == (n * 640,)
    assert reaction.shape == valid.shape == (n, 6) and intensity.shape == (n, 1)
    assert frozen['wav2vec'].shape == (n - 1, 9216) and frozen['emotion'].shape == (7,)
    assert c['frozen_audio']['pre_projection']
    assert c['frozen_audio']['waveform_samples'] == (n - 1) * 640
    value = dict(sample_id=row['sample_id'], motion=motion, audio=audio,
                 reaction=reaction, reaction_valid=valid, intensity=intensity,
                 wav2vec=frozen['wav2vec'].float(), emotion=frozen['emotion'].float(), statistics=stats)
    for x in [motion, audio, reaction, intensity, value['wav2vec'], value['emotion'], *stats.values()]:
        assert torch.isfinite(x).all(), 'Nonfinite cache: ' + row['sample_id']
    return value


def configure_window(model, n, device):
    from float.models.FMT import get_sinusoid_encoding_table, enc_dec_mask
    length = n - 1
    model.num_frames_for_clip = model.fmt.num_frames_for_clip = length
    model.num_total_frames = model.fmt.num_total_frames = n
    model.opt.wav2vec_sec = length / 25
    model.fmt.pos_embed.data = get_sinusoid_encoding_table(n, model.opt.dim_h).unsqueeze(0).to(device)
    mask = enc_dec_mask(n, n, expansion=model.opt.attention_window, num_ref_frames=1).to(device)
    count = model.fmt.num_stat_cond + model.fmt.num_null_cond
    expanded = torch.zeros(n + count, n + count, dtype=torch.bool, device=device)
    expanded[:n, :n] = mask
    model.fmt.alignment_mask = expanded


class CachedFLOAT(torch.nn.Module):
    """Only replace frozen feature calls; retain official native forward/losses."""
    def __init__(self, model):
        super().__init__(); self.model = model
        def audio_cached(encoder, audio, seq_len):
            assert seq_len == self._length
            return encoder.audio_projection(self._wav2vec)
        def emotion_cached(encoder, audio): return self._emotion
        model.audio_encoder.inference = MethodType(audio_cached, model.audio_encoder)
        model.emotion_encoder.predict_emotion = MethodType(emotion_cached, model.emotion_encoder)

    def forward(self, value):
        model = self.model; device = next(model.fmt.parameters()).device
        n = len(value['motion']); configure_window(model, n, device)
        self._length = n - 1
        self._wav2vec = value['wav2vec'][None].to(device)
        self._emotion = value['emotion'][None].to(device)
        model.reaction_loss.mask = value['reaction_valid'][1:][None].to(device)
        def reorder(x): return torch.cat([x[1:], x[:1]])[None].to(device)
        # Official floating-point slicing can omit one final waveform sample for
        # some lengths. Cached features deliberately use the documented integer
        # (N-1)*640 window; no audio encoder consumes that slice here.
        audio = torch.cat([value['audio'][640:], torch.zeros(640)])[None].to(device)
        stats = {k: v[None].to(device) for k, v in value['statistics'].items()}
        try:
            return model(reorder(value['motion']), audio, stats,
                         reorder(value['intensity']), reorder(value['reaction']))
        finally:
            self._wav2vec = self._emotion = None


def save_checkpoint(path, value):
    path = Path(path); tmp = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    with tmp.open('wb') as f:
        torch.save(value, f); f.flush(); os.fsync(f.fileno())
    tmp.replace(path)
