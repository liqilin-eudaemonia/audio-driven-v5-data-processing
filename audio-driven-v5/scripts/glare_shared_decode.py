"""Bounded same-source decoding for GLARE's two existing frame contracts.

LIA uses nearest PTS. Production detection uses native frames inside its rounded
audio interval, then its original ceil sampling and x264 recipe. Never confuse
these two grids merely because both eventually have 25 frames per second.
"""
import bisect
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np


def media_key(source):
    return json.dumps(dict(media=source['source'], size=source['video_transform']['source_size']),
                      sort_keys=True, separators=(',', ':'))


def group_rows(rows, max_gap=1., max_seconds=16., max_pairs=2):
    groups, current, key, last_end, seconds = [], [], None, None, 0.
    for row in sorted(rows, key=lambda r: (media_key(r['source']), r['source']['source_start_s'], r['sample_id'])):
        s = row['source']
        k = media_key(s)
        if current and (k != key or s['source_start_s'] - last_end > max_gap or
                        seconds + s['duration_s'] > max_seconds or len(current) >= max_pairs):
            groups.append(current)
            current, seconds = [], 0.
        current.append(row)
        key, last_end = k, max(last_end or 0., s['source_end_s']) if k == key else s['source_end_s']
        seconds += s['duration_s']
    if current:
        groups.append(current)
    return groups


def decode_group(rows, directory):
    """One original container open/seek for <=2 adjacent pairs; no raw full video."""
    import av
    from listen_visual_quality.measurement import SourceBytes
    from listen_benchmark.frame_crop import crop_original_frame_roi
    from listen_benchmark.frame_crop import crop_original_frame
    from listen_benchmark.shared_visual_decode import save_native
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    assert rows and len({media_key(r['source']) for r in rows}) == 1
    states, requests = {}, []
    for row in rows:
        sid, s = row['sample_id'], row['source']
        n = math.ceil(s['duration_s'] * 25 - 1e-7)
        rate = s['normalized_audio']['sample_rate']
        origin = round(s['source_start_s'] * rate) / rate
        states[sid] = dict(row=row, n=n, lia=[], pts=[], native=[], times=[], origin=origin,
                           detector_end=origin + s['duration_s'], crop_cache={}, next_detector=0,
                           last_native=None, last_native_t=None)
        requests += [(s['source_start_s'] + i / 25, sid, i) for i in range(n)]
    requests.sort()
    start = min(r['source']['source_start_s'] for r in rows)
    end = max(st['detector_end'] for st in states.values())
    previous = None
    previous_t = None
    index = 0
    decoded = 0
    def cropped(st, frame, production=False):
        key = (frame.pts, production)
        if key not in st['crop_cache']:
            # Production uses full RGB conversion; LIA's frozen exporter uses
            # its ROI path. Do not assume identical chroma/dither behavior.
            fn = crop_original_frame if production else crop_original_frame_roi
            st['crop_cache'][key] = np.asarray(fn(frame, st['row']['source']['video_transform']))
        return st['crop_cache'][key]
    with SourceBytes(rows[0]['source']['source']) as member, av.open(member) as container:
        stream = container.streams.video[0]
        stream.thread_type = 'SLICE'
        stream.codec_context.thread_count = 1
        assert [stream.width, stream.height] == rows[0]['source']['video_transform']['source_size']
        native_fps = float(stream.average_rate)
        container.seek(int(max(0, start - 1) / float(stream.time_base)), stream=stream,
                       backward=True, any_frame=False)
        for frame in container.decode(video=0):
            assert frame.pts is not None
            t = float(frame.pts * frame.time_base)
            decoded += 1
            for st in states.values():
                relative = t - st['origin']
                if 0 <= relative < st['row']['source']['duration_s']:
                    st['last_native'], st['last_native_t'] = frame, relative
                    # Retain only the actual native frames selected by the
                    # production ceil grid, plus the last support frame below.
                    # Its subsequent original bisect/encode output is unchanged.
                    if st['next_detector'] < st['n'] and relative >= st['next_detector'] / 25:
                        st['native'].append(cropped(st, frame, production=True))
                        st['times'].append(relative)
                        while st['next_detector'] < st['n'] and relative >= st['next_detector'] / 25:
                            st['next_detector'] += 1
            while index < len(requests) and t + 1e-8 >= requests[index][0]:
                wanted, sid, _ = requests[index]
                chosen = previous if previous is not None and abs(previous_t - wanted) <= abs(t - wanted) else frame
                pts = float(chosen.pts * chosen.time_base)
                if abs(pts - wanted) > 1 / native_fps + 1e-5:
                    from p2_j_cache.data import SourceVideoGap
                    raise SourceVideoGap('Shared source video gap')
                states[sid]['lia'].append(cropped(states[sid], chosen))
                states[sid]['pts'].append(pts)
                index += 1
            if index == len(requests) and t >= end:
                break
            previous, previous_t = frame, t
        while index < len(requests) and previous is not None:
            wanted, sid, _ = requests[index]
            if abs(previous_t - wanted) > 1 / native_fps + 1e-5:
                break
            states[sid]['lia'].append(cropped(states[sid], previous))
            states[sid]['pts'].append(previous_t)
            index += 1
    assert index == len(requests), 'Incomplete shared source support'
    result = []
    total_bytes = 0
    for sid, st in states.items():
        if st['last_native'] is not None and (not st['times'] or st['last_native_t'] > st['times'][-1]):
            st['native'].append(cropped(st, st['last_native'], production=True))
            st['times'].append(st['last_native_t'])
        assert len(st['lia']) == st['n'] and st['times'] and st['times'][0] < .1
        assert st['row']['duration_s'] - st['times'][-1] < .1
        folder = directory / sid
        folder.mkdir()
        frames = np.stack(st['lia'])
        np.save(folder / 'lia_rgb.npy', frames, allow_pickle=False)
        total_bytes += frames.nbytes
        native_dir = folder / 'native'
        save_native(native_dir, st['row']['source'], st['native'], st['times'],
                    dict(method='shared_original_container', original_opens=1, decoded_frames=decoded))
        total_bytes += sum(x.nbytes for x in st['native'])
        (folder / 'row.json').write_text(json.dumps(st['row'], ensure_ascii=False))
        result.append(dict(sample_id=sid, folder=str(folder), pts=st['pts'], frames=st['n']))
        # Arrays are only resident until their shared-memory files are complete.
        st['crop_cache'].clear()
        st['lia'].clear()
        st['native'].clear()
    assert total_bytes < 2 * 1024**3, 'Shared group exceeds the explicit 2GiB bound'
    return dict(items=result, original_opens=1, decoded_frames=decoded, shared_bytes=total_bytes)


def production_pixels(folder):
    """Original production recipe on shared native frames, with no TAR re-read."""
    import cv2
    import tempfile
    from listen_j_annotation_nod import render_video
    folder = Path(folder)
    row = json.loads((folder / 'row.json').read_text())
    with tempfile.TemporaryDirectory(prefix='production-render-', dir=folder) as tmp:
        video, meta = render_video(row, Path(tmp), fast_crop=True, decoded_dir=folder / 'native')
        cap = cv2.VideoCapture(str(video))
        frames = []
        try:
            while True:
                ok, bgr = cap.read()
                if not ok:
                    break
                frames.append(bgr)
        finally:
            cap.release()
        assert len(frames) == meta['frames']
        array = np.stack(frames)
        np.save(folder / 'detector_bgr.npy', array, allow_pickle=False)
        return array, meta
