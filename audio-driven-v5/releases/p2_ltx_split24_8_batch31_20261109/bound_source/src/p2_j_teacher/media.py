"""Final-crop listener targets on the existing 25 fps training timeline."""
import hashlib
import math
from pathlib import Path

import numpy as np
from PIL import Image

from p2_listen.contracts import read, digest


def frame_indices(duration):
    n = math.ceil(duration * 25 - 1e-7)
    count = 2 * math.ceil(duration * 4 - 1e-7)
    # Exact nearest training-grid frame, including earlier-frame tie breaking.
    return [min(range(n), key=lambda j: abs(j / 25 - i / 8)) for i in range(count)]


def selected_frames(item, override, size, reference=False):
    from p2_j_cache.data import decode_frames, SourceVideoGap
    source = item['source']; indices = frame_indices(source['duration_s'])
    policy = None if override is None else override['frames']
    if override is not None and override['source_digest'] != digest(source):
        raise ValueError('Reviewed frame override source mismatch')
    if reference:
        frames, times = decode_frames(source, gap_overrides=policy)
        return np.stack([np.asarray(Image.fromarray(frames[i]).resize(size)) for i in indices]), [times[i] for i in indices]
    # Decode the same source frames as the native 25 fps reader, crop only frames
    # selected by the 8 fps teacher. No compressed review MP4 is used.
    import av
    from listen_visual_quality.measurement import SourceBytes
    from listen_benchmark.frame_crop import crop_original_frame_roi
    desired = [source['source_start_s'] + i / 25 for i in indices]
    result = []; times = []; previous = None; previous_t = None; last_pts = None; rgb = None
    with SourceBytes(source['source']) as member, av.open(member) as container:
        stream = container.streams.video[0]
        stream.thread_type = 'SLICE'; stream.codec_context.thread_count = 1
        if [stream.width, stream.height] != source['video_transform']['source_size']:
            raise ValueError('Source video dimensions changed')
        fps = float(stream.average_rate)
        container.seek(int(max(0, source['source_start_s'] - 1) / float(stream.time_base)),
                       stream=stream, backward=True, any_frame=False)
        for frame in container.decode(video=0):
            if frame.pts is None:
                raise ValueError('Missing source PTS')
            t = float(frame.pts * frame.time_base)
            while len(result) < len(indices) and t + 1e-8 >= desired[len(result)]:
                wanted = desired[len(result)]
                chosen = previous if previous is not None and abs(previous_t-wanted) <= abs(t-wanted) else frame
                actual = float(chosen.pts * chosen.time_base)
                if abs(actual-wanted) > 1/fps + 1e-5:
                    approved = (policy or {}).get(str(indices[len(result)]))
                    record = dict(target_pts_s=wanted, previous_pts_s=previous_t, next_pts_s=t, selected_pts_s=actual)
                    if (approved is None or previous_t is None or set(approved) != set(record)
                            or any(abs(approved[k]-v)>1e-8 for k,v in record.items())):
                        raise SourceVideoGap(item['sample_id'] + ': teacher source frame gap')
                if chosen.pts != last_pts:
                    rgb = np.asarray(crop_original_frame_roi(chosen, source['video_transform']).resize(size))
                    last_pts = chosen.pts
                result.append(rgb); times.append(actual)
            if len(result) == len(indices):
                break
            previous, previous_t = frame, t
        while len(result) < len(indices) and previous is not None and abs(previous_t-desired[len(result)]) <= 1/fps+1e-5:
            if previous.pts != last_pts:
                rgb = np.asarray(crop_original_frame_roi(previous, source['video_transform']).resize(size))
                last_pts = previous.pts
            result.append(rgb); times.append(previous_t)
    if len(result) != len(indices):
        raise ValueError('Incomplete teacher source video: '+item['sample_id'])
    return np.stack(result), times


def listener_wave(item):
    from listen_benchmark.j_omni import read_audio
    from scipy.signal import resample_poly
    wave, rate, identity = read_audio(item['audio_recipe'])
    if identity != item['listener_audio_identity']:
        raise ValueError('Purified listener waveform/recipe changed')
    if item['audio_recipe']['listener_id'] != item['source']['listener_id']:
        raise ValueError('Wrong participant audio')
    mono = wave.mean(axis=1)
    if rate != 16000:
        g = math.gcd(rate, 16000)
        mono = resample_poly(mono, 16000//g, rate//g).astype(np.float32)
    n = round(item['source']['duration_s'] * 16000)
    if not n <= len(mono) <= n+1:
        raise ValueError('Wrong teacher audio length')
    return mono[:n], identity


_processor = None


def initialize_processor(model):
    global _processor
    import torch
    from transformers import AutoProcessor
    torch.set_num_threads(1)
    _processor = AutoProcessor.from_pretrained(model, local_files_only=True)


def prepare(item, override, size, grid, reference=False):
    from listen_benchmark.j_omni import check_file
    from p2_listen.layout import make_layout
    for path, h in item['source_files'].items():
        check_file(path, h)
    source = item['source']
    measurement = read(source['quality_screening']['measurement'])
    if (item['split'] != 'train' or source['video_transform']['output_size'] != [512,512]
            or source['video_transform']['crop'] != measurement['crop']['new']
            or item['crop_provenance']['revision'] != 'quality-bounded-crop-v1-20261030'):
        raise ValueError('Not accepted train/final corrected listener crop')
    frames, pts = selected_frames(item, override, size, reference)
    wave, identity = listener_wave(item)
    vi = dict(_processor.video_processor(videos=[frames], return_tensors='pt', do_sample_frames=False, do_resize=False))
    ai = dict(_processor.feature_extractor(wave, sampling_rate=16000, return_tensors='pt',
                                         return_attention_mask=True, padding='max_length'))
    layout = make_layout(source['duration_s'], grid, 8, 8)
    return dict(sample_id=item['sample_id'], video=vi, audio=ai, layout=layout,
                audit=dict(final_crop=source['video_transform']['crop'], crop_revision=item['crop_provenance']['revision'],
                    listener_audio_identity=identity, teacher_frame_indices_25fps=frame_indices(source['duration_s']),
                    source_pts=pts, teacher_rgb_sha256=hashlib.sha256(frames.tobytes()).hexdigest(),
                    audio_16k_sha256=hashlib.sha256(wave.tobytes()).hexdigest(),
                    duration_s=source['duration_s'], split=item['split'], source_video_used=True))
