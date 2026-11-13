"""Validate actual decoded frames and timestamps, never repair evaluation media."""
import json
import math
from pathlib import Path
import subprocess


def probe(path, expected_fps=25.):
    import cv2
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    command = ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_frames',
        '-show_entries', 'frame=best_effort_timestamp_time,width,height:stream=avg_frame_rate,r_frame_rate,width,height,start_time',
        '-show_streams', '-of', 'json', str(path)]
    data = json.loads(subprocess.check_output(command, text=True))
    streams = data.get('streams', [])
    if len(streams) != 1:
        raise ValueError(f'Expected one selected video stream: {path}')
    stream = streams[0]
    for key in ('avg_frame_rate', 'r_frame_rate'):
        numerator, denominator = map(float, stream[key].split('/'))
        if denominator == 0 or not math.isclose(numerator / denominator, expected_fps, abs_tol=1e-5):
            raise ValueError(f'Expected CFR {expected_fps} FPS ({key}): {path}')
    frames = data.get('frames', [])
    if not frames:
        raise ValueError(f'Empty video: {path}')
    size = (stream['width'], stream['height'])
    for i, frame in enumerate(frames):
        stamp = float(frame.get('best_effort_timestamp_time', 'nan'))
        if not math.isfinite(stamp) or abs(stamp - i / expected_fps) > 1e-4:
            raise ValueError(f'Nonzero origin, VFR, gap or shifted PTS at frame {i}: {path}')
        if (frame['width'], frame['height']) != size:
            raise ValueError(f'Geometry changed within video: {path}')
    capture = cv2.VideoCapture(str(path))
    count = 0
    try:
        if not capture.isOpened():
            raise ValueError(f'Cannot decode video: {path}')
        while True:
            ok, bgr = capture.read()
            if not ok:
                break
            if (bgr.shape[1], bgr.shape[0]) != size:
                raise ValueError(f'Decoder geometry differs: {path}')
            count += 1
    finally:
        capture.release()
    if count != len(frames):
        raise ValueError(f'Decoded frame count differs from timestamps: {path}')
    return dict(frame_count=count, width=size[0], height=size[1], fps=expected_fps,
                start_time=0., duration_s=count / expected_fps)


def aligned(left, right):
    if left != right:
        raise ValueError(f'GT/generated geometry or timeline differs: {left} vs {right}')
