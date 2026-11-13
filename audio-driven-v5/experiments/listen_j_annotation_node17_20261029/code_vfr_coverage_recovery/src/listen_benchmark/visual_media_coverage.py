"""Frozen native8 frame construction with correct variable-duration terminal coverage.

The original visual_native8.py remains the frozen input protocol/cache identity.
Only terminal support validation differs; pixels, sampling and metadata do not.
"""
from .visual_native8 import hashlib, io, math, open_source

VALIDATOR_VERSION = 'terminal-frame-support-v2'


def validate_terminal_support(last_end, duration, source_fps, count):
    if not 0 < count <= 256:
        raise ValueError('Invalid sampled frame count')
    if last_end is None or not all(math.isfinite(x) for x in (last_end,duration,source_fps)) or source_fps <= 0 or duration <= 0:
        raise ValueError('Invalid source timing')
    # A variable-duration frame may legitimately display beyond the clip end.
    # Missing coverage retains the previous one-nominal-frame tolerance.
    if duration-last_end > 1/source_fps+1e-5:
        raise ValueError(f'Source video ends {duration-last_end:.9g}s before clip support')


def prepare_media_covered(row, frame_crop=None):
    """Decode the original listener at 8 fps with the frozen crop/JPEG protocol."""
    from PIL import Image
    from listen_labeling.media import crop_resize
    rate = row['normalized_audio']['sample_rate']
    first, last = round(row['source_start_s'] * rate), round(row['source_end_s'] * rate)
    origin, duration = first / rate, (last - first) / rate
    result = {fps: dict(images=[], audio=None, duration_s=duration, meta={}) for fps in (8,)}
    records = {8: []}
    next_grid = {8: 0}
    last_end, previous = None, None
    with open_source(row['source']) as (container, reader):
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 1
        assert [stream.width, stream.height] == row['video_transform']['source_size']
        source_fps = float(stream.average_rate)
        container.seek(int(origin / stream.time_base), stream=stream, backward=True, any_frame=False)
        for frame in container.decode(stream):
            assert frame.pts is not None
            source_t = float(frame.pts * frame.time_base)
            t = source_t - origin
            if previous is not None and source_t <= previous:
                raise ValueError('Nonmonotonic original PTS')
            previous = source_t
            if t >= duration:
                break
            last_end = t + (float(frame.duration * frame.time_base) if frame.duration else 1 / source_fps)
            chosen = [fps for fps in (8,) if t >= 0 and t + 1e-8 >= next_grid[fps] / fps]
            if not chosen:
                continue
            image = (frame_crop(frame,row['video_transform']) if frame_crop is not None
                     else crop_resize(frame.to_image(), row['video_transform']))
            buf = io.BytesIO()
            image.save(buf, format='JPEG', quality=85)
            data = buf.getvalue()
            image = Image.open(io.BytesIO(data)).convert('RGB')
            for fps in chosen:
                result[fps]['images'].append(image)
                records[fps].append(dict(time_s=t, source_time_s=source_t, source_pts=frame.pts,
                    size=[512, 512], jpeg_sha256=hashlib.sha256(data).hexdigest()))
                next_grid[fps] = math.floor((t + 1e-8) * fps) + 1
    for fps in (8,):
        validate_terminal_support(last_end, duration, source_fps, len(records[fps]))
        result[fps]['meta']['video'] = dict(frames=records[fps], last_frame_s=records[fps][-1]['time_s'],
            decoded_end_s=last_end, source_time_origin_s=origin, requested_fps=fps,
            video_transform=row['video_transform'], source=row['source'], reader=reader,
            source_fps=source_fps)
    return result
