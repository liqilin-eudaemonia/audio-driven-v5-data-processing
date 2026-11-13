"""Read the final pool, not historical 500h caches or validation/test media."""
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
from functools import lru_cache

import numpy as np

ROOT = Path(os.environ.get('BASELINE_PROJECT_ROOT', Path(__file__).resolve().parents[2]))
POOL = ROOT / 'experiments/listen_j_upright_av500h_train_dev_v3_20261105'
WORK = ROOT / 'experiments/baseline_training_realdata_20261109'
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


class SelectedPool:
    def __init__(self):
        from p2_j_current_qwen_hidden import Reader
        self.reader = Reader(ROOT / 'experiments/p2_j_train_dev500h_qwen_hidden_20261105', verify=False)
        summary = read(POOL / 'selection/SUMMARY.json')
        assert summary['samples'] == 284218 and summary['test_excluded']
        assert sha(POOL / 'selection/sample_list.csv') == summary['files']['sample_list.csv']['sha256']
        self.ids = list(self.reader.protocol['sample_ids'])
        assert len(self.ids) == len(set(self.ids)) == 284218

    def row(self, sid):
        from p2_j_current_qwen_hidden import item_from_row
        row = self.reader.row(sid)
        assert row['split'] in ('train', 'dev') and row['source_split'] == row['split']
        return row, item_from_row(row)


@lru_cache(maxsize=1)
def selected_pool():
    return SelectedPool()


def prepare(work=WORK):
    pool = selected_pool()
    rows = [pool.row(sid)[0] for sid in pool.reader.protocol['canary_ids']]
    # Actual train and diagnostic validation groups are disjoint. Neither is
    # an official benchmark test, and no claim of independent original dev.
    train = []
    for row in sorted([r for r in rows if r['split'] == 'train'], key=lambda r: r['duration_s']):
        if not train or row['group_id'] != train[0]['group_id']:
            train.append(row)
        if len(train) == 2:
            break
    long = max((r for r in rows if r['group_id'] not in {x['group_id'] for x in train}),
               key=lambda r: r['duration_s'])
    train.append(long)
    val = next(r for r in rows if r['group_id'] not in {x['group_id'] for x in train})
    result = dict(schema='baseline-training-current500h-v1', samples=len(pool.ids),
                  hours=read(POOL / 'selection/SUMMARY.json')['hours'],
                  selection=str(POOL / 'selection/selected.jsonl'),
                  selection_sha256=pool.reader.protocol['selection_sha256'],
                  index=str(pool.reader.work / 'inputs.sqlite'),
                  csv_sha256=sha(POOL / 'selection/sample_list.csv'),
                  diagnostic_train=[r['sample_id'] for r in train],
                  diagnostic_validation=[val['sample_id']],
                  diagnostic_groups={r['sample_id']: r['group_id'] for r in train + [val]},
                  diagnostic_durations={r['sample_id']: r['duration_s'] for r in train + [val]},
                  held_out_benchmark_test_opened=False,
                  formal_training=False, validation_policy='group-disjoint diagnostic canaries only',
                  full_pool_reader=True)
    save(Path(work) / 'DATA_PROTOCOL.json', result)
    return result


def export_media(sid, work=WORK):
    from p2_j_cache.data import decode_frames, read_speaker
    from listen_visual_quality.measurement import SourceBytes
    import av
    import soundfile as sf
    from scipy.signal import resample_poly
    pool = selected_pool()
    row, item = pool.row(sid)
    directory = Path(work) / 'media' / sid
    directory.mkdir(parents=True, exist_ok=True)
    contract = dict(sample_id=sid, caption=row['caption'], caption_sha256=row['caption_file_sha256'],
                    row_sha256=hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest(),
                    listener_source=row['source'], group_id=row['group_id'], source_split=row['split'])
    if (directory / 'receipt.json').exists() and (directory / 'speaker30.npy').exists():
        receipt = read(directory / 'receipt.json')
        assert receipt['contract'] == contract
        assert all(sha(directory / name) == expected for name, expected in receipt['files'].items())
        return receipt
    listener, listener_pts = decode_frames(item['source'], fps=25)
    speaker_audio, rate = read_speaker(item)
    speaker_audio = speaker_audio.mean(axis=1)
    native_audio = speaker_audio.copy()
    native_rate = rate
    speaker_audio = resample_poly(speaker_audio, 16000 // math.gcd(rate, 16000), rate // math.gcd(rate, 16000)).astype(np.float32)
    exact_audio_samples = len(speaker_audio)
    # <= one frame of explicitly recorded tail rounding; no repeat of sound.
    samples = len(listener) * 640
    speaker_audio = np.pad(speaker_audio, (0, max(0, samples - len(speaker_audio))))[:samples]
    np.save(directory / 'listener.npy', listener, allow_pickle=False)
    sf.write(directory / 'speaker.wav', speaker_audio, 16000, subtype='FLOAT')
    bfm_audio = resample_poly(native_audio, 44100 // math.gcd(native_rate, 44100),
                             native_rate // math.gcd(native_rate, 44100)).astype(np.float32)
    sf.write(directory / 'speaker_bfm.wav', bfm_audio, 44100, subtype='FLOAT')
    # Speaker video is an extra native baseline input, independent of listener
    # final geometry. Its proposal is recorded, never presented as final GT crop.
    source = item['source']
    clip = read(source['source_result'])['clips'][source['source_clip_index']]
    with SourceBytes(clip['sources']['speaker']) as stream, av.open(stream) as video:
        size = [video.streams.video[0].width, video.streams.video[0].height]
    proposal = clip['crop_proposals']['speaker']
    side = min(round(proposal['size']), *size)
    x = min(max(0, round(proposal['x'])), size[0] - side)
    y = min(max(0, round(proposal['y'])), size[1] - side)
    speaker_source = dict(source, source=clip['sources']['speaker'],
                          video_transform=dict(source_size=size, output_size=[512, 512], resample='bicubic',
                                               crop=dict(x=x, y=y, width=side, height=side)))
    speaker, speaker_pts = decode_frames(speaker_source, fps=25)
    orientation_file = Path(work) / 'SPEAKER_ORIENTATION.json'
    orientations = read(orientation_file) if orientation_file.exists() else {}
    media_key = hashlib.sha256((clip['sources']['speaker']['archive'] + '\0' +
                               clip['sources']['speaker']['member']).encode()).hexdigest()[:24]
    orientation = orientations.get(media_key, {})
    rotation = orientation.get('rotation_ccw_degrees', 0)
    assert rotation in (-90, 0, 90, 180)
    if rotation:
        speaker = np.rot90(speaker, rotation // 90, axes=(1, 2)).copy()
    speaker_source['extra_orientation_review'] = orientation
    assert len(speaker) == len(listener)
    np.save(directory / 'speaker.npy', speaker, allow_pickle=False)
    listener30, listener_pts30 = decode_frames(item['source'], fps=30)
    speaker30, speaker_pts30 = decode_frames(speaker_source, fps=30)
    if rotation:
        speaker30 = np.rot90(speaker30, rotation // 90, axes=(1, 2)).copy()
    assert len(listener30) == len(speaker30)
    np.save(directory / 'listener30.npy', listener30, allow_pickle=False)
    np.save(directory / 'speaker30.npy', speaker30, allow_pickle=False)
    proc = subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-y', '-f', 'rawvideo', '-pixel_format', 'rgb24',
                           '-video_size', '512x512', '-framerate', '25', '-i', 'pipe:0', '-an', '-c:v',
                           'libx264rgb', '-crf', '0', '-preset', 'fast', '-threads', '1', str(directory / 'listener.mp4')],
                          input=listener.tobytes(), check=True)
    receipt = dict(contract=contract, frames=len(listener), fps=25,
                   speaker_source=speaker_source, listener_pts=listener_pts, speaker_pts=speaker_pts,
                   frames30=len(listener30), listener_pts30=listener_pts30, speaker_pts30=speaker_pts30,
                   speaker_original_resampled_samples=exact_audio_samples,
                   speaker_grid_samples=samples, explicit_audio_tail_padding=max(0, samples - exact_audio_samples),
                   original_listener_crop_preserved=True, future_listener_input=False,
                   native_baseline_speaker_video_input=True,
                   files={name: sha(directory / name) for name in ['listener.npy', 'speaker.npy', 'listener30.npy', 'speaker30.npy', 'speaker.wav', 'speaker_bfm.wav', 'listener.mp4']})
    save(directory / 'receipt.json', receipt)
    print('media', sid, len(listener), flush=True)
    return receipt
