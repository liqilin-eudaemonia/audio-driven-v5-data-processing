"""Conservative screening, not denoising. Model scores are not calibrated truth."""
from pathlib import Path
import math
import numpy as np
from scipy import signal,stats

POLICY=dict(revision='audio-qc-1',frame_seconds=.01,event_rms_dbfs=-55.,event_peak_dbfs=-30.,
    quiet_rms_dbfs=-50.,minimum_event_seconds=.06,maximum_event_seconds=2.5,
    human_score=.20,noise_score=.15,unexplained_loud_rms_dbfs=-42.,
    note='Conservative provisional thresholds; kept and rejected examples require human review.')
SPEECH_LABELS={'Speech','Male speech, man speaking','Female speech, woman speaking','Child speech, kid speaking','Whispering'}
VOCAL_LABELS=SPEECH_LABELS|{'Laughter','Baby laughter','Giggle','Snicker','Chuckle, chortle','Belly laugh',
    'Breathing','Sigh','Cough','Sniff','Throat clearing','Groan','Grunt','Humming'}
NOISE_LABELS={'Noise','Environmental noise','White noise','Pink noise','Static','Crackle','Crack',
    'Clicking','Clickety-clack','Burst, pop','Thump, thud','Rustle','Scratch','Scrape','Tap','Knock','Bang',
    'Wind noise (microphone)','Distortion','Hum','Mains hum','Plop','Rumble','Sine wave','Chirp tone','Pulse',
    'Hubbub, speech noise, speech babble','Speech synthesizer'}


def db(x):return float(20*np.log10(max(float(x),1e-15)))


def merge(intervals,gap=.12):
    out=[]
    for a,b in sorted(intervals):
        if b<=a:continue
        if out and a<=out[-1][1]+gap:out[-1][1]=max(b,out[-1][1])
        else:out.append([a,b])
    return out


def find_events(wave,sr,vad=()):
    hop=round(sr*.01);n=math.ceil(len(wave)/hop)
    chunks=np.pad(wave,(0,n*hop-len(wave))).reshape(n,hop)
    rms=np.sqrt(np.mean(chunks.astype(float)**2,axis=1));peak=np.max(np.abs(chunks),axis=1)
    floor=float(np.quantile(rms,.2))
    threshold=max(10**(POLICY['event_rms_dbfs']/20),floor*10**(.6))
    active=(rms>threshold)|(peak>10**(POLICY['event_peak_dbfs']/20))
    edges=np.diff(np.r_[False,active,False].astype(int));starts=np.where(edges==1)[0];ends=np.where(edges==-1)[0]
    bounds=merge([[a*.01,min(b*.01,len(wave)/sr)] for a,b in zip(starts,ends)]+
                 [[max(0,a),min(len(wave)/sr,b)] for a,b in vad],gap=.12)
    # Continuous audible noise may have no above-floor events and no supplied VAD.
    if not bounds and db(np.sqrt(np.mean(wave.astype(float)**2)))>POLICY['quiet_rms_dbfs']:
        bounds=[[0.,len(wave)/sr]]
    result=[]
    for a,b in bounds:
        if b-a<POLICY['minimum_event_seconds']:
            # Short clicks can be audible despite low integrated energy.
            if np.max(np.abs(wave[round(a*sr):round(b*sr)]),initial=0)<10**(-25/20):continue
            a=max(0,a-.03);b=min(len(wave)/sr,b+.03)
        parts=math.ceil((b-a)/POLICY['maximum_event_seconds'])
        for i in range(parts):
            result.append([a+(b-a)*i/parts,a+(b-a)*(i+1)/parts])
    return result


def features(wave,sr):
    y=wave.astype(np.float64);y-=y.mean()
    rms=float(np.sqrt(np.mean(y*y)));peak=float(np.max(np.abs(y)))
    fq,p=signal.welch(y,sr,nperseg=min(2048,len(y)));p+=1e-30
    hop=round(sr*.01);n=len(y)//hop
    env=np.sqrt(np.mean(y[:n*hop].reshape(n,hop)**2,axis=1)) if n else np.array([rms])
    rises=np.diff(20*np.log10(np.maximum(env,1e-9)))
    return dict(rms_dbfs=db(rms),peak_dbfs=db(peak),crest_db=db(peak/max(rms,1e-15)),
        kurtosis=float(stats.kurtosis(y,fisher=False)) if rms>1e-12 else 0.,
        low150=float(p[fq<150].sum()/p.sum()),low1000=float(p[fq<1000].sum()/p.sum()),
        flatness=float(np.exp(np.mean(np.log(p)))/p.mean()),
        top5_bins_fraction=float(np.sort(p)[-5:].sum()/p.sum()),
        maximum_frame_rise_db=float(np.max(rises,initial=0)))


def acoustic_flags(f):
    flags=[]
    if (f['maximum_frame_rise_db']>=18 and f['crest_db']>=12 and f['kurtosis']>=7
            and (f['top5_bins_fraction']>=.35 or f['flatness']>=.15)):
        flags.append('impulsive_burst_or_processing_artifact')
    if f['top5_bins_fraction']>=.55 and f['flatness']<.003 and f['low1000']>=.80:
        flags.append('abnormally_concentrated_narrowband_activity')
    if f['low150']>=.55 and f['rms_dbfs']>-45:
        flags.append('strong_low_frequency_activity')
    return flags


def decision(f,prediction,role,word_support):
    flags=acoustic_flags(f);human=prediction['human_score'];noise=prediction['noise_score']
    reasons=[]
    if flags and (role=='listener' or noise>=human or not word_support):reasons+=flags
    if noise>=POLICY['noise_score'] and noise>human:reasons.append('audio_event_model_noise_evidence')
    # No blanket VAD-only acceptance; quiet background and audible unexplained events differ.
    audible=f['rms_dbfs']>=POLICY['unexplained_loud_rms_dbfs'] or f['peak_dbfs']>=-28
    if audible and human<POLICY['human_score']:
        reasons.append('audible_event_without_confident_human_vocal_evidence')
    if role=='listener' and prediction['speech_score']>=.25 and not word_support and human==prediction['speech_score']:
        reasons.append('untranscribed_speech_or_possible_bleed')
    return dict(accepted=not reasons,reasons=reasons,acoustic_flags=flags,
                word_support=word_support,**prediction)


class EventModel:
    def __init__(self,model_path,device='cuda:0',batch_size=8):
        import torch
        from transformers import ASTFeatureExtractor,ASTForAudioClassification
        self.torch=torch;self.device=device;self.batch_size=batch_size
        self.extractor=ASTFeatureExtractor.from_pretrained(model_path,local_files_only=True)
        self.model=ASTForAudioClassification.from_pretrained(model_path,local_files_only=True).to(device).eval()
        self.labels=[self.model.config.id2label[i] for i in range(self.model.config.num_labels)]
        self.human_ids=[i for i,s in enumerate(self.labels) if s in VOCAL_LABELS]
        self.noise_ids=[i for i,s in enumerate(self.labels) if s in NOISE_LABELS]
        self.speech_ids=[i for i,s in enumerate(self.labels) if s in SPEECH_LABELS]

    def predict(self,waves,sr):
        answer=[]
        for start in range(0,len(waves),self.batch_size):
            batch=[signal.resample_poly(w.astype(np.float32),16000,sr) for w in waves[start:start+self.batch_size]]
            inputs=self.extractor(batch,sampling_rate=16000,return_tensors='pt')
            with self.torch.inference_mode():
                scores=self.model(**{k:v.to(self.device) for k,v in inputs.items()}).logits.sigmoid().float().cpu().numpy()
            for score in scores:
                top=np.argsort(score)[-8:][::-1]
                answer.append(dict(human_score=float(score[self.human_ids].max()),
                    speech_score=float(score[self.speech_ids].max()),noise_score=float(score[self.noise_ids].max()),
                    top_events=[dict(label=self.labels[i],score=float(score[i])) for i in top]))
        return answer


def assess(wave,sr,vad,words,model,role='listener'):
    events=find_events(wave,sr,vad);chunks=[];stats_out=[]
    for a,b in events:
        raw=wave[round(a*sr):round(b*sr)]
        f=features(raw,sr)
        supported=any(min(b,w['end'])-max(a,w['start'])>=.06 and w.get('score',0)>=.65
                      and w['end']-w['start']<=1.2 for w in words)
        chunks.append(wave[round(max(0,a-.35)*sr):round(min(len(wave)/sr,b+.45)*sr)])
        stats_out.append(dict(start=a,end=b,features=f,word_support=supported))
    predictions=model.predict(chunks,sr) if chunks else []
    for e,p in zip(stats_out,predictions):e['decision']=decision(e['features'],p,role,e['word_support'])
    if not events and role=='speaker':
        return dict(accepted=False,reasons=['missing_audible_speaker'],events=[],policy=POLICY)
    reasons=sorted({reason for e in stats_out for reason in e['decision']['reasons']})
    return dict(accepted=not reasons,reasons=reasons,events=stats_out,policy=POLICY,
                rms_dbfs=db(np.sqrt(np.mean(wave.astype(float)**2))),peak_dbfs=db(np.max(np.abs(wave))))
