"""Read only the explicitly identified listener from source-backed Seamless manifests."""
import base64,hashlib,io,math
from pathlib import Path
import av
import numpy as np
import soundfile as sf
from p2_listen.contracts import digest,safe_id
from .media import crop_resize
from .seamless import integer_crop

def compact_source(row,manifest,line_number):
    safe_id(row['sample_id'])
    if row['admission_pool']!='primary' or row['audio_final']['pool']!='keep_candidate':
        raise ValueError('Outside authorized primary audio-passed pool')
    if row['split'] not in ('train','dev','test'):raise ValueError('Unknown split')
    if row['listener_id']==row['speaker_id'] or row['listener_index']==row['speaker_index']:
        raise ValueError('Role identity conflict')
    source=row['sources']['listener'];audio=row['normalized_audio']['listener']
    if Path(source['member']).stem!=row['listener_id'] or audio['contract']['source']!=source:
        raise ValueError('Listener video and audio provenance mismatch')
    if any(type(source[k]) is not int or source[k]<0 for k in ('offset','byte_size')) or source['byte_size']==0:
        raise ValueError('Invalid source member byte range')
    geometry=row['face_geometry']['listener']
    transform=integer_crop(row['crop_proposals']['listener'],geometry['width'],geometry['height'])
    start,end=row['start'],row['end']
    if not all(type(v) in (int,float) and math.isfinite(v) for v in (start,end)) or not 0<=start<end:
        raise ValueError('Invalid source time interval')
    if abs(end-start-row['duration_s'])>2e-6:raise ValueError('Inconsistent source duration')
    return dict(schema='seamless-listener-source-v1',sample_id=row['sample_id'],pair=row['pair'],split=row['split'],
        listener_id=row['listener_id'],speaker_id=row['speaker_id'],listener_index=row['listener_index'],
        source=source,normalized_audio={k:audio[k] for k in ('path','sha256','sample_rate')},
        source_start_s=start,source_end_s=end,duration_s=end-start,video_transform=transform,
        source_manifest=str(manifest),source_line_1based=line_number,source_row_sha256=digest(row),
        boundary_review_flags=row.get('boundary_review_flags',[]),training_admitted=False)

def source_uri(source):
    return f"subfile,,start,{source['offset']},end,{source['offset']+source['byte_size']},,:{source['archive']}"

def prepare_source(row,cfg):
    """No speaker media, ASR, VAD pruning, second normalization, or physical clip export."""
    with sf.SoundFile(row['normalized_audio']['path']) as wav:
        rate=wav.samplerate
        if wav.channels!=1 or rate!=row['normalized_audio']['sample_rate']:raise ValueError('Listener waveform format mismatch')
        a,b=round(row['source_start_s']*rate),round(row['source_end_s']*rate)
        if not 0<=a<b<=len(wav):raise ValueError('Audio range outside source support')
        wav.seek(a);samples=wav.read(b-a,dtype='float32')
    if not np.isfinite(samples).all() or np.max(np.abs(samples),initial=0)>1:raise ValueError('Invalid normalized waveform')
    duration=(b-a)/rate;origin=a/rate
    if duration>cfg['max_audio_seconds']:raise ValueError('Audio budget exceeded')
    buf=io.BytesIO();sf.write(buf,samples,rate,format='WAV',subtype='PCM_16');audio_bytes=buf.getvalue()
    audio=[dict(type='text',text='Isolated LISTENER audio; sample zero is clip time 0 seconds.'),
        dict(type='input_audio',input_audio=dict(data=base64.b64encode(audio_bytes).decode(),format='wav'))]
    am=dict(samplerate=rate,samples=len(samples),wav_sha256=hashlib.sha256(audio_bytes).hexdigest(),transport='PCM_16',source_start_sample=a,source_end_sample=b,source_waveform_sha256_declared=row['normalized_audio']['sha256'])
    fps=cfg['fps'];limit=cfg['max_frames']
    if math.ceil(duration*fps)>limit:raise ValueError('Frame budget exceeded; no FPS reduction')
    visual=[];records=[];next_grid=0;last_end=None;previous=None;decoded=0
    with av.open(source_uri(row['source']),timeout=(30,60)) as container:
        stream=container.streams.video[0];stream.codec_context.thread_count=1
        if [stream.width,stream.height]!=row['video_transform']['source_size']:raise ValueError('Listener video dimensions mismatch')
        source_fps=float(stream.average_rate)
        container.seek(int(origin/stream.time_base),stream=stream,backward=True,any_frame=False)
        for frame in container.decode(stream):
            if frame.pts is None:raise ValueError('Missing source video PTS')
            source_time=float(frame.pts*frame.time_base);t=source_time-origin;decoded+=1
            if previous is not None and source_time<=previous:raise ValueError('Non-monotonic source PTS')
            previous=source_time
            if t>=duration:break
            last_end=t+(float(frame.duration*frame.time_base) if frame.duration else 1/source_fps)
            if t<0 or t+1e-8<next_grid/fps:continue
            if len(records)>=limit:raise ValueError('Frame budget exceeded')
            im=crop_resize(frame.to_image(),row['video_transform']);buf=io.BytesIO();im.save(buf,format='JPEG',quality=85);jpeg=buf.getvalue()
            visual.extend([dict(type='text',text=f'listener frame at {t:.6f} seconds:'),dict(type='image_url',image_url=dict(url='data:image/jpeg;base64,'+base64.b64encode(jpeg).decode()))])
            records.append(dict(time_s=t,source_time_s=source_time,source_pts=frame.pts,jpeg_sha256=hashlib.sha256(jpeg).hexdigest(),size=[512,512]))
            next_grid=math.floor((t+1e-8)*fps)+1
    if not records or last_end is None or abs(last_end-duration)>1/source_fps+1e-5:
        raise ValueError('Listener video interval not fully supported')
    vm=dict(frames=records,requested_fps=fps,source_time_origin_s=origin,decoded_frames=decoded,video_transform=row['video_transform'],source=row['source'])
    return dict(duration_s=duration,visual=(visual,dict(video=vm)),audio=(audio,dict(audio=am)))
