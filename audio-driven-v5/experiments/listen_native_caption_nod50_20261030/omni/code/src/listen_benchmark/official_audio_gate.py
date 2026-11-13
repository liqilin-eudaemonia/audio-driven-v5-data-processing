"""Versioned time-domain gating using official listener words/sentences/VAD only."""
import math
import numpy as np

POLICY = dict(version='official-listener-union-gate-v1',padding_seconds=.1,fade_seconds=.01,
              union_sources=['words','sentences','vad'],interval_convention='half_open_sample_indices',
              clip_before_padding=True,physical_clip_boundary_fades=False,
              fade_curve='sin_squared',empty_union='equal_length_silence')


def merge(intervals):
    out=[]
    for a,b in sorted(intervals):
        if b<=a:continue
        if out and a<=out[-1][1]:out[-1][1]=max(out[-1][1],b)
        else:out.append([int(a),int(b)])
    return out


def make_gate(official, samples, sr, padding_seconds=.1, fade_seconds=.01):
    if not isinstance(samples,int) or samples<=0 or sr<=0:raise ValueError('Invalid audio geometry')
    if not all(math.isfinite(v) and v>=0 for v in [padding_seconds,fade_seconds]):raise ValueError('Invalid gate parameters')
    duration=samples/sr;core=[];accepted=[];skipped=[]
    for kind in POLICY['union_sources']:
        if kind not in official or not isinstance(official[kind],list):raise ValueError(f'Missing official {kind}')
        for i,x in enumerate(official[kind]):
            a,b=x.get('start'),x.get('end');identity=dict(source=kind,id=x.get('id',str(i)),start=a,end=b)
            if isinstance(a,bool) or isinstance(b,bool) or not isinstance(a,(float,int)) or not isinstance(b,(float,int)) or not math.isfinite(a+b) or b<a:
                raise ValueError(f'Invalid official interval: {identity}')
            if b==a:skipped.append(dict(identity,reason='zero_length'));continue
            a,b=max(0.,a),min(duration,b)
            if b<=a:skipped.append(dict(identity,reason='outside_clip'));continue
            sa=max(0,int(math.floor(a*sr+1e-7)));sb=min(samples,int(math.ceil(b*sr-1e-7)))
            if sb<=sa:skipped.append(dict(identity,reason='empty_after_sample_conversion'));continue
            core.append([sa,sb]);accepted.append(dict(identity,clipped_start_sample=sa,clipped_end_sample=sb))
    core=merge(core);padding=round(padding_seconds*sr);keep=merge([[max(0,a-padding),min(samples,b+padding)] for a,b in core]);mask=np.zeros(samples,dtype=np.float32);fade=round(fade_seconds*sr)
    for a,b in keep:
        mask[a:b]=1
        n=min(fade,b-a)
        if n>=2:
            curve=np.sin(np.linspace(0,np.pi/2,n,dtype=np.float64))**2
            curve[0]=0.;curve[-1]=1.;curve=curve.astype(np.float32)
            if a>0:mask[a:a+n]*=curve
            if b<samples:mask[b-n:b]*=curve[::-1]
    positive=[]
    # Identify actual nonzero support, including the zero-valued fade endpoints.
    changes=np.diff(np.pad((mask>0).astype(np.int8),(1,1)));starts=np.flatnonzero(changes==1);ends=np.flatnonzero(changes==-1)
    positive=[[int(a),int(b)] for a,b in zip(starts,ends)]
    info=dict(policy=dict(POLICY,padding_seconds=padding_seconds,fade_seconds=fade_seconds),sample_rate=sr,samples=samples,duration_s=duration,
              core_intervals_samples=core,keep_intervals_samples=keep,positive_intervals_samples=positive,
              accepted_official_intervals=accepted,skipped_official_intervals=skipped,
              core_samples=sum(b-a for a,b in core),nominal_keep_samples=sum(b-a for a,b in keep),
              positive_samples=int(np.count_nonzero(mask)),zero_samples=int(np.count_nonzero(mask==0)),
              fade_samples=int(np.count_nonzero((mask>0)&(mask<1))),full_silence=not bool(positive))
    return mask,info


def gate_audio(audio, official, sr, **kwargs):
    wave=np.asarray(audio)
    if wave.ndim not in [1,2] or not np.isfinite(wave).all():raise ValueError('Invalid waveform')
    mask,info=make_gate(official,len(wave),sr,**kwargs)
    gated=wave*mask if wave.ndim==1 else wave*mask[:,None]
    return gated.astype(np.float32),mask,info


def adapt_events(events, mask, sr):
    kept=[];excluded=[];audit=[]
    for e in events:
        if e.get('modality')!='audio':kept.append(e);continue
        a,b=e.get('start'),e.get('end')
        if not isinstance(a,(float,int)) or not isinstance(b,(float,int)) or not math.isfinite(a+b) or b<=a:raise ValueError('Invalid audio event boundary')
        start=min(len(mask),max(0,int(math.floor(a*sr+1e-7))));end=max(0,min(len(mask),int(math.ceil(b*sr-1e-7))))
        support=mask[start:end]>0;changes=np.diff(np.pad(support.astype(np.int8),(1,1)))
        ranges=[dict(start=(start+int(x))/sr,end=(start+int(y))/sr) for x,y in zip(np.flatnonzero(changes==1),np.flatnonzero(changes==-1))]
        positive=int(np.count_nonzero(support));length=max(0,end-start)
        status='fully_muted' if positive==0 else 'partial' if positive<length else 'kept'
        record=dict(event_id=e['event_id'],status=status,retained_intervals=ranges,retained_fraction=positive/max(length,1),
                    source_interval=dict(start=a,end=b),text_rewritten=False)
        audit.append(record)
        if status=='fully_muted':excluded.append(dict(original_event=e,reason='outside_official_union_gate',gate=record))
        else:kept.append(dict(e,audio_gate=record))
    return dict(events=kept,excluded_audio_events=excluded,audio_event_gate_audit=audit,
                original_text_and_times_preserved=True,training_applied=False,
                partial_events_require_retained_interval_mask=True)
