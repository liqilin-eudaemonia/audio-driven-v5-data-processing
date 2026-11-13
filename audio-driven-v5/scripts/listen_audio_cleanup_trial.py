"""Six user-selected cases: preserve originals, create matched-time audio comparisons."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
from scipy import signal, ndimage
import soundfile as sf

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from listen_benchmark.backchannel_face_stats import atomic_json,read,sha,SourceReader

CASES=[
('naturalistic_V03_S1855_I00000323_local_15ae9c67d316','noisy'),
('naturalistic_V03_S0300_I00000453_local_e9fe4a285c77','noisy'),
('naturalistic_V03_S0193_I00000376_local_9f5e9514749c','noisy'),
('naturalistic_V03_S1252_I00000422_local_79317dc5efd6','noisy'),
('naturalistic_V03_S0866_I00000035_local_9f576b380455','clean_control'),
('naturalistic_V03_S1921_I00000267_local_10fad8a578fc','clean_control')]


def db(x):return float(20*np.log10(np.sqrt(np.mean(np.asarray(x,dtype=float)**2))+1e-12))


def prepare(run):
    dataset=ROOT/'experiments/p2_train50h_backchannel_face_20261026'
    obs={x['sample_id']:x for x in read(dataset/'manifests/observations.json')}
    targets={x['sample_id']:x for x in read(dataset/'manifests/targets.json')}
    preview=ROOT/'data_previews/listen_benchmark_review/train50h_random'
    feature=ROOT/'experiments/p2_features_full_crop384_20261025'
    reader=SourceReader(feature);cases=[]
    for sid,feedback in CASES:
        dest=run/'cases'/sid;dest.mkdir(parents=True,exist_ok=True)
        t,o=targets[sid],obs[sid]
        for name,source in [('original.wav',t['target_audio']),('speaker.wav',o['speaker_audio']),('video.mp4',t['target_video']),('poster.png',o['listener_image'])]:
            shutil.copy2(source,dest/name)
        label=read(preview/'media'/sid/'label.json');atomic_json(dest/'original_labels.json',label)
        full=reader.get(label['source'])
        y,sr=sf.read(dest/'original.wav',dtype='float32');x,srx=sf.read(dest/'speaker.wav',dtype='float32')
        if sr!=srx or len(x)!=len(y) or y.ndim!=1:raise ValueError('Track geometry mismatch')
        # Context comes from the exact normalized source tracks, not another recording.
        begin=max(0,full['start']-8);stop=full['end']+8
        contexts={}
        for role in ['speaker','listener']:
            source=full['normalized_audio'][role]['path']
            with sf.SoundFile(source) as f:
                if f.samplerate!=sr:raise ValueError('Context sample rate mismatch')
                first=round(begin*sr);last=min(len(f),round(stop*sr));f.seek(first)
                contexts[role]=f.read(last-first,dtype='float32')
        offset=round(full['start']*sr)-round(begin*sr)
        n=min(len(contexts['speaker']),len(contexts['listener']))
        contexts={k:v[:n] for k,v in contexts.items()}
        matches=offset+len(y)<=n and np.max(np.abs(contexts['listener'][offset:offset+len(y)]-y))<1e-4 and np.max(np.abs(contexts['speaker'][offset:offset+len(x)]-x))<1e-4
        if not matches:
            contexts={'speaker':x,'listener':y};offset=0
        for role,audio in contexts.items():sf.write(dest/f'context_{role}.wav',audio,sr,subtype='FLOAT')
        case=dict(id=sid,user_feedback=feedback,sample_rate=sr,samples=len(y),duration=len(y)/sr,
            context_offset_samples=offset,context_source_matched=bool(matches),context_samples=n if matches else len(y),
            source_hashes={k:sha(dest/k) for k in ['original.wav','speaker.wav','video.mp4']},
            original_source=dict(listener=t['target_audio'],speaker=o['speaker_audio']),
            listener_dbfs=db(y),speaker_dbfs=db(x),events=label['events'])
        cases.append(case)
    reader.close();atomic_json(run/'cases.json',cases)
    return cases


def reference_filter(y,x,sr):
    """Conservative, frozen per-recording linear-reference experiment (not a pretrained AEC).

    Fit on speaker-active, low listener/reference-ratio frames; hold-out frames
    measure transfer coherence. Suppression is disabled with insufficient evidence.
    """
    # Coarse alignment is estimated from context, not backchannel labels.
    yd=signal.resample_poly(y,1,3);xd=signal.resample_poly(x,1,3)
    corr=signal.correlate(yd,xd,mode="full",method="fft")
    lags=signal.correlation_lags(len(yd),len(xd));valid=np.abs(lags)<=8000
    lag=int(lags[valid][np.argmax(abs(corr[valid]))])*3
    aligned=np.zeros_like(x)
    if lag>0: aligned[lag:]=x[:-lag]
    elif lag<0: aligned[:lag]=x[-lag:]
    else: aligned=x.copy()
    x=aligned
    nfft=4096 if sr==48000 else 2048;hop=nfft//4
    _,times,X=signal.stft(x,fs=sr,nperseg=nfft,noverlap=nfft-hop,boundary='zeros')
    _,_,Y=signal.stft(y,fs=sr,nperseg=nfft,noverlap=nfft-hop,boundary='zeros')
    px=np.mean(abs(X)**2,axis=0);py=np.mean(abs(Y)**2,axis=0)
    active=px>max(float(np.quantile(px,.45)),1e-9)
    ratio=py/(px+1e-12)
    cal=active & (ratio <= np.quantile(ratio[active],.5))
    info=dict(method='held-out-coherence-gated frequency-domain reference subtraction',calibration_frames=int(cal.sum()),
              fft=nfft,hop=hop,estimated_reference_delay_ms=lag/sr*1000,delay_correction='context cross-correlation within +/-500 ms, then frequency-domain transfer')
    if cal.sum()<12:return y.copy(),dict(info,bypassed=True,reason='insufficient_calibration')
    indices=np.flatnonzero(cal);train=np.zeros(len(px),bool);test=train.copy()
    train=cal & ((times/.5).astype(int)%2==0);test=cal & ~train
    if train.sum()<6 or test.sum()<6:return y.copy(),dict(info,bypassed=True,reason="insufficient_held_out_frames")
    def fit(mask):
        sxx=np.mean(abs(X[:,mask])**2,axis=1)
        sxy=np.mean(Y[:,mask]*X[:,mask].conj(),axis=1)
        return sxy/(sxx+np.quantile(sxx,.6)*.01+1e-12)
    H=fit(train)
    E=H[:,None]*X[:,test];V=Y[:,test]
    num=np.abs(np.mean(E.conj()*V,axis=1))**2
    coh=num/(np.mean(abs(E)**2,axis=1)*np.mean(abs(V)**2,axis=1)+1e-30)
    coherence=float(np.sum(coh*np.mean(abs(V)**2,axis=1))/(np.sum(np.mean(abs(V)**2,axis=1))+1e-30))
    # Independent-frame coherence gates each frequency; a target speaker uncorrelated
    # with the reference is not intentionally projected out.
    gate=np.clip((ndimage.uniform_filter1d(coh,size=7)-.45)/.4,0,1)
    gate[:max(1,round(80*nfft/sr))]=0
    info.update(held_out_weighted_coherence=coherence,frequency_gate_mean=float(np.mean(gate)))
    if coherence<.35:return y.copy(),dict(info,bypassed=True,reason='weak_reference_evidence')
    H=fit(cal)*gate
    estimate=H[:,None]*X
    # Limit subtraction to a fraction of observed magnitude in each time-frequency bin.
    scale=np.minimum(1.,.8*abs(Y)/(abs(estimate)+1e-12))
    out=Y-.85*scale*estimate
    _,wave=signal.istft(out,fs=sr,nperseg=nfft,noverlap=nfft-hop,boundary=True)
    return wave[:len(y)].astype(np.float32),dict(info,bypassed=False)


def process_reference(run):
    for c in read(run/'cases.json'):
        p=run/'cases'/c['id'];y,sr=sf.read(p/'context_listener.wav',dtype='float32');x,_=sf.read(p/'context_speaker.wav',dtype='float32')
        z,info=reference_filter(y,x,sr);start=c['context_offset_samples'];z=z[start:start+c['samples']]
        sf.write(p/'reference.wav',z,sr,subtype='FLOAT');atomic_json(p/'reference_info.json',info)
        print(c['id'],json.dumps(info),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','reference']);p.add_argument('--run',type=Path,required=True)
    a=p.parse_args();a.run.mkdir(parents=True,exist_ok=True)
    if a.command=='prepare':prepare(a.run)
    else:process_reference(a.run)
