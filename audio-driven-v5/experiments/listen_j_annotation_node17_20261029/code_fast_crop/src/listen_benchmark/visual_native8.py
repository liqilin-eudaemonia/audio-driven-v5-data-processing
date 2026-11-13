"""Confirmed visual input: C observation prompt + native video8.

Extracted from the frozen 2026-10-29 input experiment. No image8/video16 branch.
No inference or file mutation occurs on import.
"""
import hashlib
import io
import math
from listen_benchmark.full_raw import open_source

MODEL = '/private/ckpts/Qwen3-VL-8B-Instruct'
MM_SIZE = {'shortest_edge':65536, 'longest_edge':67108864}

def observation_prompt(duration):
    return f'''Watch this {duration:.9g}-second listener clip. Describe all directly visible movements and facial-expression changes you recognize.
For each occurrence, give its start time, end time (seconds from clip start), and a brief description of what visibly changes.
Describe physical observations only, without interpreting their psychological or communicative meaning.
Represent a continuous action or state as one interval, not separate entries for successive frames. Keep genuinely separate occurrences separate.
Return JSON with an events array. Each event has start, end, modality ("visual"), and behavior (description).'''

def prepare_media(row, frame_crop=None):
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
        assert records[fps] and len(records[fps]) <= 256 and abs(last_end - duration) <= 1 / source_fps + 1e-5
        result[fps]['meta']['video'] = dict(frames=records[fps], last_frame_s=records[fps][-1]['time_s'],
            decoded_end_s=last_end, source_time_origin_s=origin, requested_fps=fps,
            video_transform=row['video_transform'], source=row['source'], reader=reader,
            source_fps=source_fps)
    return result

def native(media):
    import numpy as np
    frames = media['meta']['video']['frames']
    source_fps = media['meta']['video']['source_fps']
    # Official metadata uses integer frame indices and original FPS. A cut need
    # not land on a source frame: record nearest-grid timestamp errors explicitly.
    indices = [round(r['time_s'] * source_fps) for r in frames]
    errors = [i / source_fps - r['time_s'] for i, r in zip(indices, frames)]
    assert indices == sorted(set(indices)) and max(abs(x) for x in errors) <= .5 / source_fps + 1e-6
    metadata = dict(fps=source_fps, total_num_frames=max(indices[-1] + 1, math.ceil(media['duration_s'] * source_fps)),
        duration=media['duration_s'], frames_indices=indices, width=512, height=512,
        video_backend='original-pyav-presampled', do_sample_frames=False)
    evidence = dict(metadata=metadata, timestamp_rounding_errors_s=errors,
        max_abs_timestamp_rounding_error_s=max(abs(x) for x in errors),
        native_temporal_patch_size=2, final_frame_padding_count=len(frames) % 2,
        do_resize=False, do_sample_frames=False, audio_input=False)
    return (np.stack([np.asarray(im) for im in media['images']]), metadata), evidence

def prepare_input(processor, whole, start, end, view):
    from listen_benchmark.windows import slice_media
    media = whole[8] if view == 'full' else slice_media(whole[8], start, end, 8)
    prompt = observation_prompt(end-start)
    parts = [dict(type='text', text=prompt), dict(type='video', video='predecoded-listener')]
    video, evidence = native(media)
    rendered = processor.apply_chat_template([dict(role='user', content=parts)], tokenize=False,
        add_generation_prompt=True, enable_thinking=False)
    entry = dict(prompt=rendered, multi_modal_data={'video':[video]},
        mm_processor_kwargs=dict(do_resize=False, do_sample_frames=False))
    return entry, dict(media=media['meta'], native=evidence, prompt=prompt, rendered_prompt=rendered)
