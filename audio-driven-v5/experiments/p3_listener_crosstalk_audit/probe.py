"""Read-only paired-track similarity diagnostics; not a validated rejection filter."""
import json
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy import signal
ROOT=Path(__file__).resolve().parents[2]

def similarity(s,l,a,b,sr=16000):
    i=max(0,round(a*sr));j=min(len(l),round(b*sr));pad=round(.15*sr)
    y=l[i:j].astype(float);y-=y.mean()
    lo=max(0,i-pad);hi=min(len(s),j+pad);x=s[lo:hi].astype(float)
    n=len(y)
    if n<800 or np.linalg.norm(y)<1e-8:return None
    c=signal.correlate(x,y,mode='valid',method='fft')
    sums=np.r_[0,np.cumsum(x)];sq=np.r_[0,np.cumsum(x*x)]
    energy=np.maximum(sq[n:]-sq[:-n]-(sums[n:]-sums[:-n])**2/n,1e-20)
    scores=c/np.sqrt(energy*np.sum(y*y));k=int(np.argmax(np.abs(scores)))
    aligned=x[k:k+n];aligned-=aligned.mean()
    gain=float(np.dot(aligned,y)/max(np.dot(aligned,aligned),1e-20))
    residual=y-gain*aligned
    return dict(start=round(a,3),end=round(b,3),abs_wave_correlation=float(abs(scores[k])),listener_delay_ms=float((i-lo-k)/sr*1000),linear_fit_explained_energy=float(1-np.sum(residual*residual)/np.sum(y*y)))

rows=[]
for directory in ['seamless_listen_qc2_review','seamless_listen_qc_review']:
    out=ROOT/'data_previews'/directory
    for r in json.loads((out/'samples.json').read_text())['samples']:
        target=r['sample_id']=='naturalistic_V01_S0065_I00000126_qc2_p001_00'
        normal=r.get('human_review',{}).get('decision')=='retain_valid_listener_response'
        if not (target or normal):continue
        tracks=[]
        for role in ['speaker','listener']:
            x,sr=sf.read(out/r['preview']/(role+'.wav'),dtype='float32');assert sr==48000
            x=signal.resample_poly(x,1,3)
            x=signal.sosfiltfilt(signal.butter(4,[200,3500],btype='bandpass',fs=16000,output='sos'),x)
            tracks.append(x)
        s,l=tracks
        events=[similarity(s,l,max(0,e['start']-.1),min(r['duration_s'],e['end']+.1)) for e in r['audio_qc']['listener']['events']]
        rows.append(dict(sample_id=r['sample_id'],label='user_reported_speaker_bleed' if target else 'user_confirmed_valid_listener_response',whole_clip_zero_lag_correlation=float(np.corrcoef(s,l)[0,1]),events=events))
result=dict(note='Bandpass 200–3500 Hz, lag search +/-150 ms. Similarity is corroborating evidence only; low similarity does not rule out nonlinear bleed. Three previously labeled response clips are development comparisons, not an accuracy benchmark.',rows=rows)
(ROOT/'experiments/p3_listener_crosstalk_audit/paired_similarity.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
for r in rows:
 print(r['sample_id'],'whole',round(r['whole_clip_zero_lag_correlation'],3))
 print([(e['start'],e['end'],round(e['abs_wave_correlation'],3),round(e['listener_delay_ms'],1)) for e in r['events'] if e])
