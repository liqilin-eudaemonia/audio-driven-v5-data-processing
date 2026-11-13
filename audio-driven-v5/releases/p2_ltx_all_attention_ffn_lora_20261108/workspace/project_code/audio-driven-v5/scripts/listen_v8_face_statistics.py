"""Native-pixel face measurements, independent of audio admission or crop choice."""
import json
import math
import tarfile
from pathlib import Path

import numpy as np
import soundfile as sf
import prepare_listen_300h as base

REVISION = 'native-face-statistics-no-visual-gate-1'
PIXEL_EDGES = [0, 64, 96, 128, 160, 192, 256, 320, 384, 448, 512, 768, 1024, float('inf')]
AREA_EDGES = [0, .01, .02, .04, .06, .08, .10, .15, .20, .30, .50, 1.01]


def load_person(record):
    """Keep AV integrity checks; missing/invalid face metadata is only annotated."""
    with tarfile.open(record['archive'], 'r:') as tar:
        by = {m.name: m for m in tar if m.isfile()}
        refs = {}
        for ext in ('.json', '.mp4', '.wav'):
            name = record['id'] + ext
            m = by[name]
            refs[ext] = dict(archive=record['archive'], member=name, offset=m.offset_data, byte_size=m.size)
        info = json.load(tar.extractfile(by[record['id'] + '.json']))
        with sf.SoundFile(tar.extractfile(by[record['id'] + '.wav'])) as audio:
            audio_samples, sample_rate = len(audio), audio.samplerate
        keypoints = None
        geometry_issue = None
        try:
            with np.load(tar.extractfile(by[record['id'] + '.npz']), allow_pickle=False) as arrays:
                keypoints = arrays['boxes_and_keypoints:keypoints']
        except (KeyError, ValueError, OSError, EOFError) as e:
            geometry_issue = type(e).__name__ + ': ' + str(e)
    streams = json.loads(base.media.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_streams', '-of', 'json', base.media.uri(refs['.mp4'])]))['streams']
    if not streams:
        raise ValueError('source_missing_video_stream: ' + record['id'])
    probe = streams[0]
    a, b = map(int, probe['avg_frame_rate'].split('/'))
    fps = a / b
    duration = float(probe['duration'])
    frames = int(probe.get('nb_frames', round(duration * fps)))
    if abs(duration - audio_samples / sample_rate) > .08:
        raise ValueError('Source AV length disagreement')
    annotation = dict(id=record['id'], transcript=info.get('metadata:transcript', []), vad=info.get('metadata:vad', []))
    if not annotation['transcript']:
        raise ValueError('Missing transcript')
    if keypoints is not None and (keypoints.ndim != 3 or keypoints.shape[1] < 91 or keypoints.shape[2] < 3):
        geometry_issue = 'invalid_keypoint_shape'
        keypoints = None
    if keypoints is not None and abs(len(keypoints) - frames) > 1:
        # Do not align a discrepant timeline by guessing or scaling it.
        geometry_issue = 'keypoint_video_length_disagreement'
        keypoints = None
    angles = [float(probe.get('tags', {}).get('rotate', 0))] + [float(d.get('rotation', 0)) for d in probe.get('side_data_list', [])]
    geometry = dict(id=record['id'], source=refs['.mp4'], duration=min(duration, audio_samples/sample_rate),
        fps=fps, width=int(probe['width']), height=int(probe['height']),
        rotation_degrees=next((v for v in angles if v % 360), 0), geometry_issue=geometry_issue,
        arrays={'boxes_and_keypoints:keypoints': keypoints})
    source = {k: geometry[k] for k in ('id', 'source', 'duration', 'fps', 'width', 'height')}
    source.update(audio_samples=audio_samples, sample_rate=sample_rate)
    return annotation, geometry, source


def quantiles(values):
    return dict(zip(('min', 'p05', 'p10', 'p50', 'p90', 'p95', 'max'),
                    map(float, np.percentile(values, [0, 5, 10, 50, 90, 95, 100]))))


def clip_geometry(geometry, start, end):
    lo, hi = int(start * geometry['fps']), math.ceil(end * geometry['fps'])
    out = dict(revision=REVISION, status='unmeasured', width=geometry['width'], height=geometry['height'],
        fps=geometry['fps'], expected_frames=hi-lo, valid_frames=0, valid_fraction=0.,
        issue=geometry.get('geometry_issue'), rotation_degrees=geometry.get('rotation_degrees', 0),
        definition='Confident facial landmarks 23:91, per-frame 3rd/97th percentiles; native encoded pixels, not crop size.',
        confidence_threshold=.4, minimum_confident_landmarks=45, visual_filter_applied=False)
    points = geometry['arrays'].get('boxes_and_keypoints:keypoints')
    if points is None or hi <= lo:
        return out
    k = points[lo:hi, 23:91]
    confident = (k[:, :, 2] > .4) & np.isfinite(k[:, :, :3]).all(axis=2)
    valid = confident.sum(axis=1) >= 45
    if not valid.any():
        out.update(status='no_valid_face_frames', issue=out['issue'] or 'insufficient_confident_landmarks')
        return out
    xy = np.where(confident[valid, :, None], k[valid, :, :2], np.nan)
    low, high = np.nanpercentile(xy, [3, 97], axis=1)
    inside = (low >= 0).all(axis=1) & (high <= [geometry['width'], geometry['height']]).all(axis=1)
    sizes = (high-low)[inside]
    sizes = sizes[(sizes > 0).all(axis=1)]
    if not len(sizes):
        out.update(status='no_valid_face_frames', issue='degenerate_or_out_of_frame_landmarks')
        return out
    width, height = sizes.T
    short = np.minimum(width, height)
    area = width*height/(geometry['width']*geometry['height'])
    out.update(status='measured', valid_frames=len(sizes), valid_fraction=len(sizes)/max(1, hi-lo),
        face_width_px=quantiles(width), face_height_px=quantiles(height), face_short_side_px=quantiles(short),
        face_area_fraction=quantiles(area), face_width_fraction=quantiles(width/geometry['width']),
        face_height_fraction=quantiles(height/geometry['height']),
        frame_short_side_histogram=np.histogram(short, PIXEL_EDGES)[0].tolist(),
        frame_area_fraction_histogram=np.histogram(area, AREA_EDGES)[0].tolist())
    return out


def bin_index(value, edges):
    return min(len(edges)-2, max(0, int(np.searchsorted(edges, value, side='right')-1)))


def summarize_clips(clips):
    """Small per-pair records make live aggregation independent of QC payload size."""
    pools = {}
    for row in clips:
        key = '/'.join((row['split'], row['admission_pool'], row['audio_final']['pool']))
        group = pools.setdefault(key, dict(clips=0, seconds=0., cached_clips=0, fresh_clips=0, roles={}))
        group['clips'] += 1
        group['seconds'] += row['duration_s']
        group['cached_clips' if row['quality_provenance']['reused'] else 'fresh_clips'] += 1
        for role, face in row['face_geometry'].items():
            stats = group['roles'].setdefault(role, dict(measured_clips=0, unmeasured_clips=0, unmeasured_seconds=0.,
                valid_frames=0, expected_frames=0, median_short_side_counts=[0]*(len(PIXEL_EDGES)-1),
                median_short_side_seconds=[0.]*(len(PIXEL_EDGES)-1), median_area_counts=[0]*(len(AREA_EDGES)-1),
                median_area_seconds=[0.]*(len(AREA_EDGES)-1),
                frame_short_side_counts=[0]*(len(PIXEL_EDGES)-1), frame_area_counts=[0]*(len(AREA_EDGES)-1),
                validity_counts=[0]*5, validity_seconds=[0.]*5,
                p10_short_side_threshold_seconds={str(t):0. for t in PIXEL_EDGES[1:-1]}))
            stats['expected_frames'] += face['expected_frames']
            stats['valid_frames'] += face['valid_frames']
            v = bin_index(face['valid_fraction'], [0, .5, .8, .9, .99, 1.01])
            stats['validity_counts'][v] += 1
            stats['validity_seconds'][v] += row['duration_s']
            if face['status'] != 'measured':
                stats['unmeasured_clips'] += 1
                stats['unmeasured_seconds'] += row['duration_s']
                continue
            stats['measured_clips'] += 1
            for label, metric, edges in [('short_side', 'face_short_side_px', PIXEL_EDGES), ('area', 'face_area_fraction', AREA_EDGES)]:
                i = bin_index(face[metric]['p50'], edges)
                stats['median_'+label+'_counts'][i] += 1
                stats['median_'+label+'_seconds'][i] += row['duration_s']
                histogram = face['frame_short_side_histogram' if label == 'short_side' else 'frame_area_fraction_histogram']
                stats['frame_'+label+'_counts'] = [a+b for a,b in zip(stats['frame_'+label+'_counts'], histogram)]
            for t in PIXEL_EDGES[1:-1]:
                if face['face_short_side_px']['p10'] >= t:
                    stats['p10_short_side_threshold_seconds'][str(t)] += row['duration_s']
    return pools


def merge_counts(target, source):
    for key, value in source.items():
        if isinstance(value, dict):
            merge_counts(target.setdefault(key, {}), value)
        elif isinstance(value, list):
            target[key] = [a+b for a,b in zip(target.get(key, [0]*len(value)), value)]
        else:
            target[key] = target.get(key, 0) + value
    return target
