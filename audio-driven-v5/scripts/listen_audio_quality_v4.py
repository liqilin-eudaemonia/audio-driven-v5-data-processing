"""Additional review gates, not waveform repair or calibrated echo attribution.

V8 boundaries and QC3 evidence stay intact. This module never promotes a
previous review/exclude to a candidate and never removes or modifies audio.
"""
from collections import Counter
import numpy as np
from scipy import signal

POLICY=dict(revision='qc4-reference-and-role-1',window_seconds=1.,stride_seconds=.5,
    max_delay_seconds=1.,sample_rate=8000,reference_rms_min_dbfs=-45.,listener_rms_min_dbfs=-55.,
    broad_correlation_review=.45,band_correlation_review=.55,strong_correlation_review=.65,
    consistent_windows=2,delay_tolerance_ms=25.,untranscribed_speech_ast=.25,
    untranscribed_speech_panns=.5,minimum_speech_event_seconds=.15,
    denoising=False,hard_rejection_validated=False,
    note='Provisional review gates. Speech without transcript is not labeled noise. No gain, boundary, or waveform edits.')
SPEECH={'Speech','Male speech, man speaking','Female speech, woman speaking','Child speech, kid speaking','Whispering'}
LAUGHTER={'Laughter','Baby laughter','Giggle','Snicker','Chuckle, chortle','Belly laugh'}


def rms_db(wave):
    return float(20*np.log10(max(float(np.sqrt(np.mean(np.asarray(wave,dtype=float)**2))),1e-12)))


def matched_window(x,y,start,end,sr,pad):
    i=round(start*sr);j=min(len(y),round(end*sr));n=j-i
    lo=max(0,i-pad);hi=min(len(x),j+pad)
    if n<sr*.15 or hi-lo<n:return None
    target=y[i:j].astype(float);target-=target.mean()
    norm=float(target@target)
    if norm<1e-16:return None
    ref=x[lo:hi].astype(float)
    cross=signal.correlate(ref,target,mode='valid',method='fft')
    sums=np.r_[0.,np.cumsum(ref)];sq=np.r_[0.,np.cumsum(ref*ref)]
    energy=np.maximum(sq[n:]-sq[:-n]-(sums[n:]-sums[:-n])**2/n,1e-20)
    scores=np.abs(cross)/np.sqrt(energy*norm)
    k=int(np.argmax(scores));aligned=ref[k:k+n]
    return dict(correlation=float(min(1.,scores[k])),delay_ms=float((i-lo-k)*1000/sr),
        reference_rms_dbfs=rms_db(aligned),listener_rms_dbfs=rms_db(target))


def reference_scan(speaker,listener,sr=48000):
    """Scan all windows, including gaps in transcript/VAD, over +/-1 second."""
    target_sr=POLICY['sample_rate']
    waves=[signal.resample_poly(np.asarray(w,dtype=float),target_sr,sr) for w in (speaker,listener)]
    duration=min(map(len,waves))/target_sr
    if duration<.2:return dict(windows=[],reasons=[],status='review',unavailable='too_short')
    bands=[(200,3200),(200,800),(800,1600),(1600,3200)]
    filtered=[]
    for low,high in bands:
        sos=signal.butter(3,[low,high],btype='bandpass',fs=target_sr,output='sos')
        filtered.append([signal.sosfiltfilt(sos,w) for w in waves])
    size=min(POLICY['window_seconds'],duration)
    starts=list(np.arange(0,max(0,duration-size)+1e-7,POLICY['stride_seconds']))
    if not starts or duration-size-starts[-1]>1e-6:starts.append(duration-size)
    windows=[];strong=[]
    for a in starts:
        b=min(duration,a+size)
        matches=[matched_window(*ch,a,b,target_sr,round(POLICY['max_delay_seconds']*target_sr)) for ch in filtered]
        broad=matches[0]
        if broad is None:continue
        audible=broad['reference_rms_dbfs']>POLICY['reference_rms_min_dbfs'] and broad['listener_rms_dbfs']>POLICY['listener_rms_min_dbfs']
        agreeing=[m for m in matches[1:] if m and m['correlation']>=POLICY['band_correlation_review']
            and abs(m['delay_ms']-broad['delay_ms'])<=POLICY['delay_tolerance_ms']]
        suspicious=audible and (broad['correlation']>=POLICY['broad_correlation_review'] or len(agreeing)>=2)
        row=dict(start=float(a),end=float(b),**broad,bands=matches[1:],suspicious=suspicious)
        windows.append(row)
        if suspicious:strong.append(row)
    reasons=[]
    if any(w['correlation']>=POLICY['strong_correlation_review'] for w in strong):reasons.append('strong_reference_match')
    stable=max((sum(abs(x['delay_ms']-w['delay_ms'])<=POLICY['delay_tolerance_ms'] for x in strong) for w in strong),default=0)
    if stable>=POLICY['consistent_windows']:reasons.append('consistent_reference_leakage')
    if strong and not reasons:reasons.append('isolated_reference_match_requires_review')
    return dict(windows=windows,reasons=reasons,status='review' if reasons else 'keep',
        max_correlation=max((w['correlation'] for w in windows),default=0.),consistent_windows=stable)


def top_score(prediction,labels):
    return max((x['score'] for x in prediction.get('top_events',[]) if x['label'] in labels),default=0.)


def speech_identity(row):
    """Do not mistake recognized speech type for a confirmed listener voice."""
    flagged=[]
    for event in row['qc3']['listener']['events']:
        ast=event['ast'];panns=event['panns'];f=event['features']
        if event.get('word_support') or event['end']-event['start']<POLICY['minimum_speech_event_seconds']:continue
        audible=f['rms_dbfs']>-45 or f['peak_dbfs']>-30
        speech=ast.get('speech_score',0.)>=POLICY['untranscribed_speech_ast'] and top_score(panns,SPEECH)>=POLICY['untranscribed_speech_panns']
        # Clear laughter is not made suspicious merely for having no transcript.
        laughter=top_score(ast,LAUGHTER)>=.2 and top_score(panns,LAUGHTER)>=.2
        if audible and speech and not laughter:
            flagged.append(dict(start=event['start'],end=event['end'],reason='listener_speech_identity_unconfirmed',
                ast_speech=ast.get('speech_score',0.),panns_speech=top_score(panns,SPEECH)))
    return flagged


def assess(row,speaker,listener,feedback=()):
    reference=reference_scan(speaker,listener)
    identity=speech_identity(row)
    content=row.get('listener_compatibility',{}).get('uncertain_short_speech',[])
    substantive=row.get('listener_compatibility',{}).get('substantive',[])
    task_reasons=['listener_content_not_confirmed_backchannel'] if content or substantive else []
    audio_reasons=reference['reasons']+(['listener_speech_identity_unconfirmed'] if identity else [])
    automatic=('exclude' if row['v8_combined_status']=='exclude' else
        'review' if row['v8_combined_status']=='review' or task_reasons or audio_reasons else 'candidate')
    # An interval-level human hold is a separate override, never proof of
    # automatic detection or a reason to reject the whole source conversation.
    hits=[f for f in feedback if f['listener_id']==row['listener_id'] and
          min(f['end'],row['end'])>max(f['start'],row['start'])+1e-6]
    status='held_by_human' if hits else automatic
    return dict(policy_revision=POLICY['revision'],automatic_status=automatic,status=status,
        task_reasons=task_reasons,audio_reasons=sorted(set(audio_reasons)),
        content_events=content+substantive,speech_identity_windows=identity,
        reference=reference,human_holds=[f['sample_id'] for f in hits],
        original_qc3_status=row['audio_qc_status'],original_combined_status=row['v8_combined_status'],
        boundary_unchanged=True,waveform_unchanged=True,training_admitted=False)
