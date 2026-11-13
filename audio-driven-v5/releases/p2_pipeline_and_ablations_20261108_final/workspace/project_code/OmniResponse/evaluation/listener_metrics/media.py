"""Temporary evaluation media only; original waveforms and video never edited."""
import hashlib
import json
import math
from pathlib import Path
import subprocess
import numpy as np
from scipy.signal import resample_poly
from .core import union,intervals_from_mask,mask
from ..seamless.reader import backend,load_record,load_listener_audio


def read_audio(path,duration):
    info=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','a:0','-show_streams','-of','json',str(path)]))
    streams=info.get('streams',[])
    if len(streams)!=1:raise ValueError('Missing generated listener audio is not silence')
    if abs(float(streams[0].get('start_time',0)))>1e-4:raise ValueError('Nonzero audio origin')
    wave=np.frombuffer(subprocess.check_output(['ffmpeg','-v','error','-threads','1','-i',str(path),'-map','0:a:0','-vn','-ac','1','-ar','16000','-f','f32le','pipe:1']),np.float32).copy()
    if not len(wave) or not np.isfinite(wave).all():raise ValueError('Empty/nonfinite audio')
    if abs(len(wave)/16000-duration)>.05:raise ValueError('Audio/pair duration mismatch (>50ms)')
    return wave


def audio_pair(row,duration):
    official=[];source='GT waveform; ASR recorded as auxiliary support'
    if 'gt_reference' in row:
        wave,sr,audit=load_listener_audio(row['gt_reference'])
        gt=np.asarray(wave).reshape(-1)
        div=math.gcd(sr,16000);gt=resample_poly(gt,16000//div,sr//div).astype(np.float32)
        record,_=load_record(row['gt_reference']);cap=backend().checked_json(record['target']['source_file'],record['target']['source_sha256'])
        recipe=cap['audio_recipe'];source='official words/sentences/VAD union; existing cleaned waveform'
        for items in recipe['official_listener'].values():
            for item in items:
                if isinstance(item,dict):official.append([item['start'],item['end']])
                else:official.append(item[:2])
    else:
        gt=read_audio(row['gt_audio'],duration);audit={}
        if row.get('gt_asr'):
            asr=json.loads(Path(row['gt_asr']).read_text());origin=row['source_start']
            for item in asr.get('words',[])+asr.get('segments',[]):
                official.append([float(item['start'])-origin,float(item['end'])-origin])
    from ..video.paired_metrics.media import sha256
    generated_path=row.get('generated_audio') or row['generated']
    pred=read_audio(generated_path,duration)
    n=min(len(gt),len(pred),round(duration*16000));original_samples=[len(gt),len(pred)];gt,pred=gt[:n],pred[:n]
    return gt,pred,dict(source=source,official_support=union(official,n/16000),cleaned_audio_audit=audit,
        generated_audio_source=str(generated_path),generated_audio_source_sha256=sha256(generated_path),
        gt_audio_source=row.get('gt_audio','lazy_cleaned_listener'),gt_audio_source_sha256=sha256(row['gt_audio']) if row.get('gt_audio') else None,
        evaluated_duration_s=n/16000,original_samples_16k=original_samples,trimmed_tail_samples=[x-n for x in original_samples],gt_rgb_independent=True,
        gt_waveform_sha256=hashlib.sha256(gt.tobytes()).hexdigest(),generated_waveform_sha256=hashlib.sha256(pred.tobytes()).hexdigest())


def activity(wave,options,vad):
    duration=len(wave)/16000;hop=160;length=320
    rms=np.array([np.sqrt(np.mean(wave[a:min(len(wave),a+length)].astype(float)**2)) for a in range(0,len(wave),hop)])
    energy=intervals_from_mask(rms>=10**(options['energy_dbfs']/20),duration=duration,
        min_s=options['energy_min_s'],gap_s=options['energy_gap_s'])
    speech=vad(wave)
    # Absolute energy floor prevents near-silent numerical noise from entering speech scores.
    speech_mask=mask(speech,duration)&mask(energy,duration)
    speech=intervals_from_mask(speech_mask,duration=duration,min_s=.04,gap_s=0.)
    return dict(acoustic=energy,speech=speech,rms=float(np.sqrt(np.mean(wave.astype(float)**2))),
        peak=float(np.max(np.abs(wave))),duration_s=duration)


class SpeechDetector:
    def __init__(self):
        import silero_vad
        from ..video.paired_metrics.media import sha256
        self.api=silero_vad;self.model=silero_vad.load_silero_vad(onnx=True)
        root=Path(silero_vad.__file__).parent
        self.identity=dict(name='silero-vad',version=silero_vad.__version__,onnx_sha256=sha256(root/'data/silero_vad.onnx'),
            source_sha256=sha256(root/'utils_vad.py'),threshold=.5,min_speech_duration_ms=40,min_silence_duration_ms=80,speech_pad_ms=0)
    def __call__(self,wave):
        import torch
        rows=self.api.get_speech_timestamps(torch.from_numpy(wave.copy()),self.model,sampling_rate=16000,
            threshold=.5,min_speech_duration_ms=40,min_silence_duration_ms=80,speech_pad_ms=0,return_seconds=False)
        return [[r['start']/16000,r['end']/16000] for r in rows]


def grid25(frames,fps,duration):
    # Common absolute time grid; no time stretching. Ties select earlier frame.
    count=math.ceil(duration*25-1e-7);times=np.arange(count)/25
    indices=np.ceil(times*fps-.5-1e-8).astype(int).clip(0,len(frames)-1)
    if np.max(np.abs(indices/fps-times))>1/fps+1e-5:raise ValueError('Missing source video tail')
    return frames[indices],dict(input_fps=fps,output_fps=25,source_frame_indices=indices.tolist(),duration_s=duration)


def write_video(frames,path):
    # The original helper fixes 512, so encode arbitrary original sizes here.
    import av
    from fractions import Fraction
    with av.open(str(path),'w') as c:
        s=c.add_stream('ffv1',rate=25);s.width=frames.shape[2];s.height=frames.shape[1];s.pix_fmt='bgr0';s.options={'threads':'1'}
        for i,rgb in enumerate(frames):
            f=av.VideoFrame.from_ndarray(rgb,format='rgb24');f.pts=i;f.time_base=Fraction(1,25)
            for p in s.encode(f):c.mux(p)
        for p in s.encode():c.mux(p)
