"""Speaker-only, resumable Whisper caches; never recode existing AV/Qwen caches."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from functools import lru_cache
from itertools import islice
from pathlib import Path
import fcntl
import hashlib
import json
import math

import numpy as np
import torch

from p2_listen.cache import save_part, load_part
from p2_listen.contracts import ROOT, read, sha, digest, atomic_json, safe_id
from .features import FrozenWhisper, SpeakerFeatures

CACHE_VERSION = 'p2-j-speaker-whisper-v1'


def feature_code_identity():
    return {name: sha(Path(__file__).parent / name) for name in ('cache.py', 'features.py')}


def model_contract(path):
    path = Path(path).resolve()
    files = sorted(path.glob('*.safetensors')) + sorted(path.glob('*.json'))
    if not (path/'config.json').is_file() or not any(p.suffix == '.safetensors' for p in files):
        raise ValueError('Whisper requires a local pinned safetensors snapshot')
    cfg = read(path/'config.json')
    if cfg.get('model_type') != 'whisper' or cfg.get('d_model') != 1280:
        raise ValueError('Expected Whisper-large-v3, last layer width 1280')
    return dict(path=str(path), files={p.name: sha(p) for p in files}, extractor='WhisperModel.encoder',
        layer='last', input_dim=1280, sample_rate=16000, hop_samples=320, window_s=30,
        context_s=2, dtype='float32', timeline='common-clip-origin; no silence removal; disjoint cores')


def observation_contract(item):
    source, speaker = item['source'], item['speaker']
    start, end = source['source_start_s'], source['source_end_s']
    if (source['speaker_id'] == source['listener_id'] or not 0 <= start < end or
            Path(speaker['contract']['source']['member']).stem != source['speaker_id']):
        raise ValueError('Speaker/listener identity or source interval mismatch')
    return dict(sample_id=item['sample_id'], split=item['split'], speaker=speaker,
        speaker_id=source['speaker_id'], listener_id=source['listener_id'],
        source_start_s=start, source_end_s=end, origin_s=0.)


def prepare(c, *, sample_ids=None):
    from p2_j_cache.data import open_manifest
    work = Path(c['whisper_cache']); source = Path(c['j_cache'])
    manifest = open_manifest(source)
    ids = list(manifest['splits']['train'] if sample_ids is None else sample_ids)
    if not ids or len(set(ids)) != len(ids) or not set(ids) <= set(manifest['sample_ids']):
        raise ValueError('Expected unique selected J sample IDs')
    encoder = model_contract(c['whisper']['model_path'])
    if encoder['input_dim'] != c['whisper']['input_dim']:
        raise ValueError('Encoder dimension differs from training configuration')
    protocol = dict(schema=CACHE_VERSION, source=str(source.resolve()), source_manifest_sha256=sha(source/'manifest.json'),
        sample_ids=ids, sample_ids_sha256=digest(ids), samples=len(ids), encoder=encoder,
        code=feature_code_identity(), intermediate_wav=False, intermediate_video=False)
    atomic_json(work/'protocol.json', protocol, immutable=True)
    return dict(samples=len(ids), protocol=str(work/'protocol.json'), training_started=False)


class Reader:
    def __init__(self, work, *, require_complete=False, source=None):
        from p2_j_cache.data import open_manifest
        self.work = Path(work); self.protocol = read(self.work/'protocol.json')
        p = self.protocol
        if p['schema'] != CACHE_VERSION or p['code'] != feature_code_identity():
            raise ValueError('Whisper cache implementation changed; use a new cache directory')
        if len(set(p['sample_ids'])) != p['samples'] or digest(p['sample_ids']) != p['sample_ids_sha256']:
            raise ValueError('Whisper cache membership changed')
        self.ids = set(p['sample_ids'])
        self.source = Path(p['source'])
        if source is not None and Path(source).resolve() != self.source.resolve():
            raise ValueError('Whisper cache belongs to another J dataset')
        if sha(self.source/'manifest.json') != p['source_manifest_sha256']:
            raise ValueError('Whisper J source manifest changed')
        self.manifest = open_manifest(self.source)
        if require_complete:
            complete = read(self.work/'COMPLETE.json')
            if (complete.get('success') is not True or complete.get('samples') != p['samples'] or
                    complete.get('protocol_sha256') != sha(self.work/'protocol.json')):
                raise ValueError('Whisper cache full audit incomplete')

    def item(self, sid):
        from p2_j_cache.data import sample
        if sid not in self.ids:
            raise ValueError('Sample absent from the frozen Whisper cache list')
        return sample(self.source, self.manifest, sid)

    def identity(self, item):
        return dict(schema=CACHE_VERSION, code=self.protocol['code'], encoder=self.protocol['encoder'],
                    observation=observation_contract(item))

    def folder(self, sid):
        return self.work/'cache'/safe_id(sid)

    def load(self, sid, device='cpu', *, missing_ok=False):
        item = self.item(sid); value = load_part(self.folder(sid), self.identity(item), missing_ok=missing_ok)
        if value is None:
            return None
        features = validate_features(value, self.protocol['encoder']['input_dim'])
        expected_duration = item['source']['source_end_s']-item['source']['source_start_s']
        if abs(value['num_samples_16k']/16000-expected_duration) > 2/16000:
            raise ValueError('Cached speaker duration differs from the common clip timeline')
        return features.to(device)


def validate_features(value, input_dim):
    features = SpeakerFeatures(**value['features']).validate()
    n = value['num_samples_16k']; count = math.ceil(n/320)
    if (type(n) is not int or n < 1 or features.hidden.shape != (1, count, input_dim)
            or not features.valid.all() or features.hidden.dtype != torch.float32):
        raise ValueError('Invalid Whisper token geometry')
    starts = torch.arange(count, dtype=torch.int64)*320
    times = torch.stack((starts, (starts+320).clamp(max=n)), -1).double().div(16000).float()[None,None]
    if not torch.equal(features.bounds, times):
        raise ValueError('Whisper time grid shifted, truncated, duplicated, or retimed')
    return features


def speaker_wave(item):
    from p2_j_cache.data import read_speaker
    from scipy.signal import resample_poly
    # Same paired source, exact interval, and leading silence as baseline Qwen.
    wave, rate = read_speaker(item)
    wave = np.asarray(wave, dtype=np.float32)
    if wave.ndim != 2 or wave.shape[1] != 1 or not np.isfinite(wave).all():
        raise ValueError('Expected finite mono speaker waveform')
    wave = wave[:,0]
    if rate != 16000:
        g = math.gcd(rate,16000)
        wave = resample_poly(wave,16000//g,rate//g).astype(np.float32)
    if not len(wave):
        raise ValueError('Empty speaker interval')
    return np.ascontiguousarray(wave)


def _read_batch(reader, ids):
    return [(sid, reader.item(sid)) for sid in ids]


def extract(c, *, rank=0, world=1, device='cpu', batch_size=16, read_workers=2, limit=None):
    if (type(world) is not int or world < 1 or type(rank) is not int or not 0 <= rank < world
            or type(batch_size) is not int or batch_size < 1 or type(read_workers) is not int or read_workers < 1
            or limit is not None and (type(limit) is not int or limit < 1)):
        raise ValueError('Invalid cache worker geometry')
    reader = Reader(c['whisper_cache'], source=c['j_cache'])
    if model_contract(c['whisper']['model_path']) != reader.protocol['encoder']:
        raise ValueError('Whisper weights/config changed since cache preparation')
    ids = reader.protocol['sample_ids'][rank::world]
    if limit is not None:
        ids = ids[:limit]
    encoder = None; complete = reused = 0
    iterator = iter(ids)
    # Keep only one batch of waveforms plus the active encoder batch in memory.
    with ThreadPoolExecutor(max_workers=read_workers) as pool:
        while batch := list(islice(iterator,batch_size)):
            if (reader.work/'STOP').exists():
                raise InterruptedError('Whisper cache STOP; existing results retained')
            pending = []
            for sid, item in _read_batch(reader, batch):
                if reader.load(sid,missing_ok=True) is None:
                    pending.append((sid,item))
                else:
                    reused += 1
            if pending:
                waves = list(pool.map(speaker_wave,[item for _,item in pending]))
                if encoder is None:
                    encoder = FrozenWhisper(c['whisper']['model_path'],device,torch.float32)
                features = encoder.encode_batch(waves,batch_size=batch_size)
                for (sid,item),wave,value in zip(pending,waves,features):
                    payload = dict(features=asdict(value),num_samples_16k=len(wave),
                        speaker_wave_sha256=hashlib.sha256(wave.astype('<f4',copy=False).tobytes()).hexdigest())
                    validate_features(payload,reader.protocol['encoder']['input_dim'])
                    folder = reader.folder(sid);folder.mkdir(parents=True,exist_ok=True)
                    with (folder/'writer.lock').open('a') as lock:
                        fcntl.flock(lock,fcntl.LOCK_EX)
                        save_part(folder,reader.identity(item),payload)
                    reader.load(sid)
                    complete += 1
            atomic_json(reader.work/'progress'/f'rank{rank:03d}.json',
                dict(rank=rank,world=world,new=complete,reused=reused,total=len(ids),finished=False))
    result = dict(rank=rank,world=world,new=complete,reused=reused,total=len(ids),finished=True)
    atomic_json(reader.work/'progress'/f'rank{rank:03d}.json',result)
    return result


def finalize(c):
    reader = Reader(c['whisper_cache'],source=c['j_cache'])
    count = tokens = 0; h = hashlib.sha256()
    for sid in reader.protocol['sample_ids']:
        features = reader.load(sid);tokens += features.hidden.shape[1];count += 1
        h.update((sid+':'+sha(reader.folder(sid)/'complete.json')+'\n').encode())
        if count % 4096 == 0:
            print(json.dumps(dict(stage='whisper_cache_audit',samples=count,total=len(reader.ids))),flush=True)
    result = dict(success=True,samples=count,tokens=tokens,protocol_sha256=sha(reader.work/'protocol.json'),
                  cache_receipts_sha256=h.hexdigest(),audio_source='speaker',intermediate_wav=False)
    atomic_json(reader.work/'COMPLETE.json',result,immutable=True)
    return result
