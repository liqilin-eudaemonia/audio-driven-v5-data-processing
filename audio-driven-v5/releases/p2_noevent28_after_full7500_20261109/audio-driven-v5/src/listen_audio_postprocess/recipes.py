"""CPU-only virtual audio: decode source, apply frozen masks, never write WAV."""
import hashlib
import math
from pathlib import Path
import numpy as np
import soundfile as sf
from p2_listen.contracts import digest, sha
from listen_benchmark.official_audio_gate import make_gate, POLICY as GATE_POLICY
from listen_benchmark.chinese_asr_filter import POLICY as CHINESE_POLICY

SCHEMA='listener-audio-recipe-v1'


def waveform_digest(wave,rate):
    wave=np.asarray(wave,dtype='<f4',order='C')
    h=hashlib.sha256();h.update(f'float32le:{rate}:{wave.shape}'.encode());h.update(wave.tobytes());return h.hexdigest()


def make_recipe(sample_id,source,source_sha256,official,extra_mute_intervals=(),provenance=None):
    source=Path(source).resolve()
    if sha(source)!=source_sha256:raise ValueError('Source audio SHA256 mismatch')
    info=sf.info(source)
    result=dict(schema=SCHEMA,sample_id=sample_id,
                source=dict(path=str(source),sha256=source_sha256,sample_rate=info.samplerate,samples=info.frames,channels=info.channels),
                official_listener={k:[{q:v for q,v in x.items() if q in ('id','start','end')} for x in official[k]] for k in ('words','sentences','vad')},
                extra_mute_intervals=[dict(start=float(x['start']),end=float(x['end'])) for x in extra_mute_intervals],
                gate_policy=GATE_POLICY,extra_mute_policy=CHINESE_POLICY,provenance=provenance or {})
    masks(result)  # Validate before publishing a recipe.
    return result


def masks(recipe):
    if recipe.get('schema')!=SCHEMA or recipe['gate_policy']!=GATE_POLICY or recipe['extra_mute_policy']!=CHINESE_POLICY:
        raise ValueError('Unsupported audio recipe/policy; regenerate explicitly')
    src=recipe['source'];n=src['samples'];sr=src['sample_rate']
    if type(n) is not int or n<=0 or type(sr) is not int or sr<=0 or src['channels'] not in (1,2):raise ValueError('Invalid source audio geometry')
    gate,info=make_gate(recipe['official_listener'],n,sr)
    from listen_benchmark.official_audio_gate import merge
    cut=[]
    for x in recipe['extra_mute_intervals']:
        a,b=x['start'],x['end']
        if not all(type(t) in (int,float) and math.isfinite(t) for t in (a,b)) or not 0<=a<b<=n/sr+1e-7:
            raise ValueError('Invalid extra mute interval')
        cut.append([max(0,math.floor(a*sr+1e-7)),min(n,math.ceil(b*sr-1e-7))])
    extra=np.ones(n,dtype=np.float32);fade=round(CHINESE_POLICY['fade_seconds']*sr)
    for a,b in merge(cut):
        extra[a:b]=0
        k=min(fade,a)
        if k>=2:extra[a-k:a]*=np.cos(np.linspace(0,np.pi/2,k,dtype=np.float64)).astype(np.float32)**2
        k=min(fade,n-b)
        if k>=2:extra[b:b+k]*=np.sin(np.linspace(0,np.pi/2,k,dtype=np.float64)).astype(np.float32)**2
    return gate,extra,info


def read_listener_audio(recipe):
    """Return native-rate float32 [samples,channels] and reproducibility metadata.

    Apply operations in the same order as the reviewed WAVs, before resampling
    or VAE padding. Hash physical source bytes on every encoding invocation.
    """
    source=recipe['source'];path=Path(source['path'])
    if sha(path)!=source['sha256']:raise ValueError('Source audio changed')
    wave,rate=sf.read(path,dtype='float32',always_2d=True)
    if rate!=source['sample_rate'] or wave.shape!=(source['samples'],source['channels']) or not np.isfinite(wave).all():
        raise ValueError('Source audio geometry/data changed')
    gate,extra,_=masks(recipe)
    # Two multiplies preserve float32 rounding of the two reviewed stages.
    processed=(wave*gate[:,None])*extra[:,None]
    return processed,rate,dict(recipe_sha256=digest(recipe),processed_waveform_sha256=waveform_digest(processed,rate),
                               source_waveform_sha256=waveform_digest(wave,rate),waveform_changed=not np.array_equal(wave,processed))


def implementation_identity():
    import listen_benchmark.official_audio_gate as gate
    import listen_benchmark.chinese_asr_filter as chinese
    return {str(Path(p).resolve()):sha(p) for p in (__file__,gate.__file__,chinese.__file__)}
