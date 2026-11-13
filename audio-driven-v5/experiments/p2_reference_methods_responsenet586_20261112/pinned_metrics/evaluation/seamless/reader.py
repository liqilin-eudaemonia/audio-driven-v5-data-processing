"""Reuse validated audio-driven-v5 readers; never feed future GT to generation."""
import hashlib
import json
import math
from pathlib import Path
import sys
import numpy as np
from ..video.paired_metrics.media import sha256

DATA_ROOT=Path(__file__).resolve().parents[3]/'audio-driven-v5'
SCHEMA='seamless-lazy-gt-v1'
FPS=25


def backend():
    src=str(DATA_ROOT/'src')
    if src not in sys.path:sys.path.insert(0,src)
    from listen_response_sft import data
    return data


def reader_contract():
    return {name:sha256(DATA_ROOT/name) for name in (
        'src/listen_response_sft/data.py','src/listen_benchmark/j_omni_crop.py',
        'src/listen_benchmark/frame_crop.py','src/listen_benchmark/j_omni.py',
        'src/listen_visual_quality/measurement.py','src/listen_visual_quality/io.py')}


def upright_reader_contract():
    return {name:sha256(DATA_ROOT/name) for name in (
        'scripts/j_upright_response_sft_data.py', 'scripts/p2_j_current_qwen_hidden.py')}


def upright_backend():
    backend()
    scripts=str(DATA_ROOT/'scripts')
    if scripts not in sys.path:sys.path.insert(0,scripts)
    import j_upright_response_sft_data
    return j_upright_response_sft_data


def frame_count(duration):
    if not math.isfinite(duration) or duration<=0:raise ValueError('Invalid duration')
    return math.ceil(duration*FPS-1e-7)


def load_record(reference):
    if reference['schema']!=SCHEMA:raise ValueError('Unknown lazy GT schema')
    if reference['offset']<0 or reference['bytes']<=0:raise ValueError('Invalid row byte range')
    with open(reference['manifest'],'rb') as f:
        f.seek(reference['offset']);raw=f.read(reference['bytes'])
    if hashlib.sha256(raw).hexdigest()!=reference['row_sha256']:raise ValueError('Frozen Seamless row changed')
    row=json.loads(raw)
    if row['sample_id']!=reference['sample_id'] or row['split']!='test':raise ValueError('Wrong Seamless identity/split')
    config_path=Path(reference['crop_config'])
    if sha256(config_path)!=reference['crop_config_sha256']:raise ValueError('Crop configuration changed')
    if reader_contract()!=reference['reader_contract']:raise ValueError('Upstream media reader changed; prepare/review a new dataset version')
    if row.get('schema')=='j-upright-response-sft-reference-v1':
        if reference.get('upright_reader_contract')!=upright_reader_contract():
            raise ValueError('Upright reader changed; prepare a new dataset version')
    return row,json.loads(config_path.read_text())


def resolve(reference):
    row,cfg=load_record(reference)
    if row.get('schema')=='j-upright-response-sft-reference-v1':
        item=upright_backend().resolve(row)
        return row,dict(source=item['source'],speaker_audio=item['speaker'],
            crop_provenance=dict(policy='published_upright_source_and_final_crop'),
            annotation_crop_differs=False,target_text=item['caption'])
    return row,backend().resolve_example(row,cfg)


def decode_source(src):
    """Same nearest-PTS 25Hz and crop-before-bicubic policy as training review.

    Returns RGB directly; no intermediate lossy MP4 or long-video extraction.
    """
    backend()
    import av
    from listen_visual_quality.measurement import SourceBytes
    from listen_benchmark.frame_crop import crop_original_frame_reused
    start=src['source_start_s'];count=frame_count(src['duration_s'])
    desired=[start+i/FPS for i in range(count)]
    frames=[];times=[];previous=None;previous_time=None;last_pts=None;cropped=None
    with SourceBytes(src['source']) as member,av.open(member) as container:
        stream=container.streams.video[0]
        stream.thread_type='SLICE';stream.codec_context.thread_count=1
        if [stream.width,stream.height]!=src['video_transform']['source_size']:raise ValueError('Source size changed')
        native_fps=float(stream.average_rate)
        if native_fps<=0:raise ValueError('Invalid source FPS')
        container.seek(int(max(0,start-1)/float(stream.time_base)),stream=stream,backward=True,any_frame=False)
        for frame in container.decode(video=0):
            if frame.pts is None:raise ValueError('Missing source PTS')
            timestamp=float(frame.pts*frame.time_base)
            while len(frames)<count and timestamp+1e-8>=desired[len(frames)]:
                target=desired[len(frames)]
                chosen=previous if previous is not None and abs(previous_time-target)<=abs(timestamp-target) else frame
                actual=float(chosen.pts*chosen.time_base)
                if abs(actual-target)>1/native_fps+1e-5:raise ValueError('Source video gap')
                if last_pts!=chosen.pts:
                    cropped=np.asarray(crop_original_frame_reused(chosen,src['video_transform']))
                    last_pts=chosen.pts
                frames.append(cropped);times.append(actual)
            if len(frames)==count:break
            previous,previous_time=frame,timestamp
    # Derived upright clips retain original PTS but stop at the clip boundary.
    # The final target can follow the final decoded PTS by less than one frame;
    # use that nearest frame, with the same gap tolerance as the main loop.
    while len(frames)<count and previous is not None:
        target=desired[len(frames)]
        if abs(previous_time-target)>1/native_fps+1e-5:break
        if last_pts!=previous.pts:
            cropped=np.asarray(crop_original_frame_reused(previous,src['video_transform']))
            last_pts=previous.pts
        frames.append(cropped);times.append(previous_time)
    if len(frames)!=count:raise ValueError(f'Incomplete source decode: {len(frames)}/{count}')
    array=np.stack(frames)
    return array,dict(fps=FPS,native_fps=native_fps,frames=count,width=512,height=512,
        source_pts_s=times,maximum_grid_error_s=max(abs(a-b) for a,b in zip(times,desired)),
        rgb_sha256=hashlib.sha256(memoryview(array)).hexdigest(),
        video_transform=src['video_transform'],source_start_s=start,source_end_s=src['source_end_s'],
        policy='nearest_source_pts_25hz_crop_then_bicubic_512_no_codec_roundtrip')


def decode_reference(reference):
    row,resolved=resolve(reference)
    frames,audit=decode_source(resolved['source'])
    audit.update(sample_id=row['sample_id'],crop_provenance=resolved['crop_provenance'],
                 annotation_crop_differs=resolved['annotation_crop_differs'],row_sha256=reference['row_sha256'])
    return frames,float(FPS),audit


def load_conditioning(reference):
    """Only speaker audio + single listener first frame, using existing reader."""
    row,cfg=load_record(reference)
    if row.get('schema')=='j-upright-response-sft-reference-v1':
        example=upright_backend().load_example(row)
        sr=example['inputs']['speaker_sample_rate'];audio=row['inputs']['speaker_audio']
        example['audit'].update(speaker_first_sample=round(audio['start_s']*sr),
            speaker_last_sample=round(audio['end_s']*sr),upright_reference=True,
            reference_row_sha256=reference['row_sha256'])
    else:example=backend().load_example(row,cfg)
    return example['inputs'],example['audit']


def load_listener_audio(reference):
    """Optional GT audio: apply existing official union mute recipe on demand."""
    row,resolved=resolve(reference)
    data=backend()
    caption=data.checked_json(row['target']['source_file'],row['target']['source_sha256'])
    recipe=caption['audio_recipe']
    if recipe['listener_id']!=resolved['source']['listener_id']:raise ValueError('Wrong listener audio')
    from listen_benchmark.j_omni import read_audio
    wave,sr,audit=read_audio(recipe)
    if abs(len(wave)/sr-row['duration_s'])>1/sr:raise ValueError('Listener audio duration mismatch')
    if audit['processed_waveform_sha256']!=caption['input_evidence']['audio_identity']['processed_waveform_sha256']:
        raise ValueError('Cleaned listener audio differs from original annotation input')
    return wave,sr,audit
