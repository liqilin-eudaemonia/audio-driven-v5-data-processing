"""Resolve source references lazily; never feed future listener media as input."""
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re

from listen_benchmark.j_omni import check_file
from listen_benchmark.j_omni_crop import effective_row, REVISION
from listen_visual_quality.io import canonical_hash

SCHEMA = 'j-smallface-response-sft-reference-v1'


def require(condition, message):
    if not condition:
        raise ValueError(message)


@lru_cache(maxsize=8)
def _json_cached(path, size, mtime):
    return json.loads(Path(path).read_bytes())


def checked_json(path, digest):
    check_file(path, digest)
    st = Path(path).stat()
    return _json_cached(str(path), st.st_size, st.st_mtime_ns)


def validated_target_text(target, original):
    """Allow only the recorded, deterministic removal of a leading role label."""
    fmt = target.get('format', 'original_caption_verbatim')
    if fmt == 'original_caption_verbatim':
        expected = original
    elif fmt == 'original_caption_strip_chat_prefix_v1':
        require(target.get('original_text') == original, 'Original target text changed')
        match = re.match(r'^\s*Assistant\s*:\s*', original, re.IGNORECASE)
        require(match is not None, 'Original caption has no leading Assistant prefix')
        require(target.get('removed_prefix') == match.group(), 'Removed prefix differs from original')
        expected = original[match.end():]
        require(bool(expected.strip()), 'Prefix removal left an empty caption')
    else:
        raise ValueError('Unknown target text format')
    require(target['text'] == expected, 'Target text changed beyond allowed normalization')
    return expected


def statistics(row):
    p = row['provenance']
    with open(p['statistics_file'], 'rb') as f:
        f.seek(p['statistics_byte_offset'])
        raw = f.readline()
    require(hashlib.sha256(raw).hexdigest() == p['statistics_line_sha256'], 'Statistics changed')
    r = json.loads(raw)
    require(r['sample_id'] == row['sample_id'] and r['split'] == row['split'], 'Statistics identity mismatch')
    validated_target_text(row['target'], r['caption'])
    return r


def resolve_example(row, crop_config):
    """Validate provenance and return resolved media references; no decoding."""
    require(row['schema'] == SCHEMA, 'Unknown reference schema')
    rec = statistics(row)
    src = rec['source']
    require(src['listener_id'] != src['speaker_id'], 'Same speaker and listener')
    cap = checked_json(row['target']['source_file'], row['target']['source_sha256'])
    require(cap['sample_id'] == row['sample_id'], 'Caption identity mismatch')
    target_text = validated_target_text(row['target'], cap['caption'])
    original = cap.get('original_source', cap['source'])
    require(src == cap['source'], 'Statistics source differs from caption input source')
    audio = row['inputs']['speaker_audio']
    frame = row['inputs']['listener_first_frame']
    require(audio['participant_id'] == src['speaker_id'], 'Wrong audio participant')
    require(audio['source_result_file'] == src['source_result'] and audio['source_result_sha256'] == src['source_result_sha256'], 'Audio source reference mismatch')
    require(audio['clip_index'] == src['source_clip_index'], 'Wrong source clip')
    require(audio['start_s'] == src['source_start_s'] and audio['end_s'] == src['source_end_s'], 'Audio time mismatch')
    require(frame['source'] == src['source'] and frame['source_time_s'] == src['source_start_s'], 'First-frame source mismatch')
    require(frame['participant_id'] == src['listener_id'], 'Wrong frame participant')
    require(frame['crop_revision'] == REVISION and frame['output_size'] == [512,512], 'Wrong crop version/size')
    require(frame['measurement_file'] == rec['quality']['measurement_file'] and frame['measurement_sha256'] == rec['quality']['measurement_sha256'], 'Crop reference mismatch')
    require(canonical_hash(crop_config) == row['provenance']['crop_config_sha256'], 'Crop config changed')
    applied, crop = effective_row({'sample_id': row['sample_id'], 'source': original}, crop_config, decision=rec['quality'])
    pair = checked_json(audio['source_result_file'], audio['source_result_sha256'])
    clip = pair['clips'][audio['clip_index']]
    require(clip['sample_id'] == row['sample_id'] and clip['split'] == row['split'], 'Pair identity/split mismatch')
    require(clip['listener_id'] == src['listener_id'] and clip['speaker_id'] == src['speaker_id'], 'Pair participant mismatch')
    require(abs(clip['start']-audio['start_s']) < 1e-7 and abs(clip['end']-audio['end_s']) < 1e-7, 'Pair time mismatch')
    require(clip['sources']['listener'] == src['source'], 'Listener video mismatch')
    a = clip['normalized_audio']['speaker']
    require(a['contract']['source'] == clip['sources']['speaker'], 'Speaker audio contract mismatch')
    require(Path(clip['sources']['speaker']['member']).stem == src['speaker_id'], 'Speaker source identity mismatch')
    require(Path(src['source']['member']).stem == src['listener_id'], 'Listener source identity mismatch')
    return dict(sample_id=row['sample_id'], split=row['split'], source=applied['source'],
                speaker_audio=a, crop_provenance=crop,
                annotation_crop=cap['source']['video_transform']['crop'],
                annotation_crop_differs=cap['source']['video_transform']['crop'] != applied['source']['video_transform']['crop'],
                target_text=target_text)


def first_frame(src):
    """Nearest decoded PTS to clip start, same rule as the 24-sample review."""
    import av
    from listen_visual_quality.measurement import SourceBytes
    from listen_benchmark.frame_crop import crop_original_frame_reused
    start = src['source_start_s']
    with SourceBytes(src['source']) as member, av.open(member) as container:
        stream=container.streams.video[0]
        stream.thread_type='SLICE';stream.codec_context.thread_count=1
        require([stream.width,stream.height] == src['video_transform']['source_size'], 'Source size mismatch')
        rate=float(stream.average_rate)
        container.seek(int(max(0,start-1)/float(stream.time_base)),stream=stream,backward=True,any_frame=False)
        previous=None;previous_t=None
        for frame in container.decode(video=0):
            require(frame.pts is not None, 'Missing source PTS')
            t=float(frame.pts*frame.time_base)
            if t+1e-8>=start:
                chosen=previous if previous is not None and abs(previous_t-start)<=abs(t-start) else frame
                actual=float(chosen.pts*chosen.time_base)
                require(abs(actual-start)<=1/rate+1e-5,'Source first-frame gap')
                return crop_original_frame_reused(chosen,src['video_transform']),actual
            previous=frame;previous_t=t
    raise ValueError('No frame at requested clip start')


def load_example(row, crop_config):
    """Return only speaker waveform and a single image as inputs; text is target.

    Waveform is mono float32 at its stored rate (currently 48kHz). A model's
    audio processor must resample explicitly to its required rate. No listener
    waveform, future video, transcript, or target-text feature is an input.
    """
    import numpy as np
    import soundfile as sf
    r=resolve_example(row,crop_config);src=r['source'];a=r['speaker_audio']
    check_file(a['path'],a['sha256'])
    with sf.SoundFile(a['path']) as f:
        sr=f.samplerate
        require(sr==a['sample_rate'] and f.channels==1,'Wrong speaker audio format')
        first,last=round(src['source_start_s']*sr),round(src['source_end_s']*sr)
        require(0<=first<last<=len(f),'Speaker audio bounds invalid')
        f.seek(first);wave=f.read(last-first,dtype='float32')
    require(len(wave)==last-first and np.isfinite(wave).all(),'Incomplete speaker audio')
    require(abs(len(wave)/sr-row['duration_s'])<=1/sr,'Speaker duration mismatch')
    image,pts=first_frame(src)
    return dict(sample_id=row['sample_id'],split=row['split'],
                inputs=dict(speaker_audio=wave,speaker_sample_rate=sr,listener_first_frame=image),
                target_text=r['target_text'],
                audit=dict(speaker_audio_path=a['path'],speaker_audio_sha256=a['sha256'],
                    speaker_first_sample=first,speaker_last_sample=last,
                    source_frame_time_s=pts,crop_provenance=r['crop_provenance'],
                    annotation_crop_differs=r['annotation_crop_differs'],
                    future_listener_media_in_inputs=False))
