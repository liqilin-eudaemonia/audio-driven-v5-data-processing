"""Explicit listener audio, checked time axes, no hidden shortening/padding."""
from pathlib import Path
import json
import math
import subprocess
from .common import digest


def geometry(path):
    from p2_glare_eval.media import probe
    return probe(path)


def aligned(gt,generated):
    from p2_glare_eval.media import aligned as check
    check(gt,generated)


def window(sample_id,frames,options):
    import random
    need=options['fvd_frames']*options['fvd_stride']
    if frames<need:return None
    seed=int(digest([sample_id,options['crop_seed']])[:16],16)
    start=random.Random(seed).randint(0,frames-need)
    return list(range(start,start+need,options['fvd_stride']))


def audio(path,output,duration_s):
    import numpy as np
    import soundfile as sf
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','a:0','-show_streams','-of','json',str(path)],text=True))
    streams=probe.get('streams',[])
    if len(streams)!=1:raise ValueError('Explicit listener audio has no decodable audio stream')
    start=float(streams[0].get('start_time',0))
    if not math.isfinite(start) or abs(start)>1e-4:raise ValueError('Listener audio PTS must start at zero')
    wave=np.frombuffer(subprocess.check_output(['ffmpeg','-v','error','-threads','1','-i',str(path),'-map','0:a:0','-vn','-ac','1','-ar','16000','-f','f32le','pipe:1']),dtype=np.float32).copy()
    if not len(wave) or not np.isfinite(wave).all():raise ValueError('Empty/nonfinite listener audio')
    # Native fractional-duration windows can differ by less than one video frame.
    if abs(len(wave)/16000-duration_s)>1/25+1/16000:raise ValueError('Listener audio/video duration mismatch; no -shortest repair')
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    sf.write(output,wave,16000,subtype='PCM_16')
    return dict(samples=len(wave),rate=16000,duration_s=len(wave)/16000,rms=float(np.sqrt(np.mean(wave.astype(np.float64)**2))),peak=float(np.max(np.abs(wave))),path=str(output))


def mux_listener(video,wave,output):
    subprocess.run(['ffmpeg','-v','error','-y','-threads','1','-i',str(video),'-i',str(wave),
        '-map','0:v:0','-map','1:a:0','-c:v','copy','-c:a','pcm_s16le',str(output)],check=True)
