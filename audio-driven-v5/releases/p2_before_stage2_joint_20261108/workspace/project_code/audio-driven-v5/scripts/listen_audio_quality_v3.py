"""QC3 paired-track and independent background checks. Review only, no denoising."""
import importlib.util
import json
import math
import sys
from pathlib import Path
import numpy as np
from scipy import signal
from listen_audio_quality import db,features

ROOT=Path(__file__).resolve().parents[1]
MODEL=ROOT/'models/audio_qc_panns'
POLICY=dict(revision='audio-qc-3',scope='same 56 QC2 candidates; development regression separate',
    hard_rejection_validated=False,denoising=False,
    cross_track_high_correlation=.75,cross_track_review_correlation=.45,
    model_noise_strong=.45,model_noise_review=.20,
    vocal_two_model=.20,vocal_panns_strong=.60,
    background_window_seconds=2.,background_stride_seconds=1.,
    note='Human voice and contamination are independent evidence; thresholds provisional and frozen before this run.')

# Explicit physical/environmental classes. Animal-like predictions can be processing artifacts;
# those remain uncertain rather than automatically being called real environmental sounds.
ENV_LABELS={'Noise','Environmental noise','White noise','Pink noise','Static','Crackle','Crack',
    'Clicking','Clickety-clack','Burst, pop','Thump, thud','Rustle','Scratch','Scrape','Tap','Knock','Bang',
    'Wind noise (microphone)','Distortion','Hum','Mains hum','Plop','Rumble','Sine wave','Chirp tone','Pulse',
    'Mechanisms','Mechanical fan','Air conditioning','Vacuum cleaner','Power tool','Drill',
    'Engine','Engine starting','Idling','Vehicle','Traffic noise, roadway noise','Car','Car passing by',
    'Vehicle horn, car horn, honking','Siren','Alarm','Bell','Telephone bell ringing',
    'Door','Slam','Sliding door','Doorbell','Dishes, pots, and pans','Cutlery, silverware',
    'Typing','Computer keyboard','Keys jangling','Glass','Shatter','Footsteps',
    'Water','Running water','Water tap, faucet','Rain','Rain on surface','Wind','Thunder',
    'Music','Musical instrument','Television','Radio'}
VOCAL_LABELS={'Speech','Male speech, man speaking','Female speech, woman speaking','Child speech, kid speaking',
    'Whispering','Laughter','Baby laughter','Giggle','Snicker','Chuckle, chortle','Belly laugh',
    'Breathing','Sigh','Cough','Sniff','Throat clearing','Groan','Grunt','Humming','Gasp','Yawn'}


class PannsModel:
    def __init__(self,device='cuda:0',batch_size=16):
        import torch
        sys.path.insert(0,str(MODEL/'vendor'))
        # Import the model file without package __init__, which otherwise downloads to HOME.
        import types
        package=types.ModuleType('qc3_panns');package.__path__=[str(MODEL/'vendor/panns_inference')]
        sys.modules['qc3_panns']=package
        spec=importlib.util.spec_from_file_location('qc3_panns.models',MODEL/'vendor/panns_inference/models.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        self.torch=torch;self.device=device;self.batch_size=batch_size
        self.model=module.Cnn14(sample_rate=32000,window_size=1024,hop_size=320,
                               mel_bins=64,fmin=50,fmax=14000,classes_num=527)
        checkpoint=torch.load(MODEL/'Cnn14_mAP=0.431.pth',map_location='cpu',weights_only=True)
        self.model.load_state_dict(checkpoint['model'],strict=True)
        self.model.to(device).eval()
        # AST uses the same AudioSet class index order; verify against packaged class CSV.
        import csv
        with open(MODEL/'class_labels_indices.csv') as f:
            classes=list(csv.DictReader(f))
        self.labels=[r['display_name'] for r in classes]
        assert len(self.labels)==527 and all(int(r['index'])==i for i,r in enumerate(classes))
        ast=json.loads((ROOT/'models/audio_qc_ast/config.json').read_text())['id2label']
        assert all(ast[str(i)]==x for i,x in enumerate(self.labels))
        self.noise_ids=[i for i,x in enumerate(self.labels) if x in ENV_LABELS]
        self.vocal_ids=[i for i,x in enumerate(self.labels) if x in VOCAL_LABELS]
        assert len(self.noise_ids)>40 and len(self.vocal_ids)>15

    def predict(self,waves,sr=48000):
        result=[]
        for offset in range(0,len(waves),self.batch_size):
            batch=[signal.resample_poly(w.astype(np.float32),2,3) if sr==48000 else
                   signal.resample_poly(w.astype(np.float32),32000,sr) for w in waves[offset:offset+self.batch_size]]
            size=max(max(len(w) for w in batch),64000)
            x=np.stack([np.pad(w,(0,size-len(w))) for w in batch])
            with self.torch.inference_mode():
                scores=self.model(self.torch.from_numpy(x).to(self.device))['clipwise_output'].float().cpu().numpy()
            for s in scores:
                top=np.argsort(s)[-8:][::-1]
                ni=max(self.noise_ids,key=lambda i:s[i]);vi=max(self.vocal_ids,key=lambda i:s[i])
                result.append(dict(human_score=float(s[vi]),human_label=self.labels[vi],
                    noise_score=float(s[ni]),noise_label=self.labels[ni],
                    top_events=[dict(label=self.labels[i],score=float(s[i])) for i in top]))
        return result


def prepare_pair(speaker,listener,sr=48000):
    filtered=[]
    for x in (speaker,listener):
        if x is None:filtered.append(None);continue
        x=signal.resample_poly(x,16000,sr)
        x=signal.sosfiltfilt(signal.butter(4,[200,3500],btype='bandpass',fs=16000,output='sos'),x)
        filtered.append(x)
    return filtered


def paired_similarity(speaker,listener,start,end,sr=16000):
    if speaker is None:return dict(available=False)
    i=max(0,round(start*sr));j=min(len(listener),round(end*sr));pad=round(.15*sr)
    y=listener[i:j].astype(float);y-=y.mean()
    lo=max(0,i-pad);hi=min(len(speaker),j+pad);x=speaker[lo:hi].astype(float)
    n=len(y)
    if n<800 or np.linalg.norm(y)<1e-8:return dict(available=False)
    c=signal.correlate(x,y,mode='valid',method='fft')
    sums=np.r_[0,np.cumsum(x)];sq=np.r_[0,np.cumsum(x*x)]
    energy=np.maximum(sq[n:]-sq[:-n]-(sums[n:]-sums[:-n])**2/n,1e-20)
    scores=c/np.sqrt(energy*np.sum(y*y));k=int(np.argmax(np.abs(scores)))
    aligned=x[k:k+n];aligned-=aligned.mean()
    return dict(available=True,abs_wave_correlation=float(abs(scores[k])),
        listener_delay_ms=float((i-lo-k)/sr*1000),window_seconds=n/sr,
        speaker_rms_dbfs=db(np.sqrt(np.mean(aligned*aligned))),
        listener_rms_dbfs=db(np.sqrt(np.mean(y*y))))


def full_coverage_windows(duration,size=2.,stride=1.):
    if duration<=size:return [(0.,duration)]
    starts=list(np.arange(0,duration-size+1e-8,stride))
    if abs(starts[-1]-(duration-size))>1e-6:starts.append(duration-size)
    return [(float(a),float(min(a+size,duration))) for a in starts]


def ast_environment(prediction):
    # Cached AST only has top eight scores. Missing is unknown, never proof of zero noise.
    scores=[p['score'] for p in prediction.get('top_events',[]) if p['label'] in ENV_LABELS]
    return max(scores,default=0.)


def decide_event(f,ast,panns,paired):
    reasons=[];warnings=[];status='keep'
    audible=f['rms_dbfs']>=-45 or f['peak_dbfs']>=-30
    corr=paired.get('abs_wave_correlation',0.)
    paired_audible=paired.get('speaker_rms_dbfs',-100)>-45 and paired.get('listener_rms_dbfs',-100)>-55
    # These are exclusion candidates, not ground-truth labels. No waveform mutation.
    if paired_audible and corr>=POLICY['cross_track_high_correlation']:
        status='exclude';reasons.append('strong_local_speaker_match')
    elif paired_audible and corr>=POLICY['cross_track_review_correlation']:
        status='review';reasons.append('possible_local_speaker_match')
    pn=panns['noise_score'];an=ast_environment(ast)
    if audible and pn>=POLICY['model_noise_strong'] and an>=.20:
        status='exclude';reasons.append('two_model_environmental_contamination')
    elif audible and pn>=POLICY['model_noise_review']:
        if status=='keep':status='review'
        reasons.append('environmental_sound_evidence')
    human=(panns['human_score']>=.20 and ast.get('human_score',0)>=.20) or (
        panns['human_score']>=.60 and ast.get('human_score',0)>=.10)
    if audible and not human:
        if status=='keep':status='review'
        reasons.append('reaction_identity_uncertain')
    if not audible:warnings.append('weak_activity_not_confirmed_reaction')
    return dict(status=status,reasons=reasons,warnings=warnings,human_evidence=human,
                panns_noise_independent_of_human=True,ast_environment_top8_score=an)


def background_decision(f,panns):
    # Entire-track scan, including VAD gaps. Human score never cancels a noise flag.
    reasons=[];audible=f['rms_dbfs']>=-45 or f['peak_dbfs']>=-30
    if audible and panns['noise_score']>=.20:reasons.append('background_environmental_evidence')
    if f['rms_dbfs']>-45 and f['flatness']>.15:reasons.append('audible_broadband_background')
    return dict(status='review' if reasons else 'keep',reasons=reasons)


def scan_track(wave,old_check,model,paired=None,sr=48000,role='listener'):
    duration=len(wave)/sr;events=[];contexts=[]
    for old in old_check['events']:
        a,b=old['start'],old['end'];lo=max(0,a-.4);hi=min(duration,b+.4)
        contexts.append(wave[round(lo*sr):round(hi*sr)])
        events.append(dict(start=a,end=b,features=old['features'],ast=old['decision'],
            word_support=old.get('word_support',False),paired=paired_similarity(*paired,max(0,a-.1),min(duration,b+.1)) if paired else dict(available=False)))
    windows=full_coverage_windows(duration)
    contexts.extend(wave[round(a*sr):round(b*sr)] for a,b in windows)
    predictions=model.predict(contexts,sr)
    for e,p in zip(events,predictions[:len(events)]):
        e['panns']=p;e['decision']=decide_event(e['features'],e['ast'],p,e['paired'])
    background=[]
    for (a,b),p in zip(windows,predictions[len(events):]):
        f=features(wave[round(a*sr):round(b*sr)],sr)
        background.append(dict(start=a,end=b,features=f,panns=p,decision=background_decision(f,p)))
    hop=round(.02*sr);n=len(wave)//hop
    rms=np.sqrt(np.mean(wave[:n*hop].astype(float).reshape(n,hop)**2,axis=1))
    floor=db(np.quantile(rms,.2));reasons=sorted({r for e in events+background for r in e['decision']['reasons']})
    states=[e['decision']['status'] for e in events+background]
    status='exclude' if 'exclude' in states else 'review' if 'review' in states else 'keep'
    if floor>-45 and role=='listener':
        reasons.append('elevated_between_event_floor_requires_review')
        if status=='keep':status='review'
    if not events and db(np.sqrt(np.mean(wave.astype(float)**2)))>-45:
        reasons.append('audible_track_without_identified_events')
        if status=='keep':status='review'
    return dict(status=status,reasons=reasons,events=events,background=background,
                floor20_dbfs=floor,rms_dbfs=db(np.sqrt(np.mean(wave.astype(float)**2))),
                has_human_evidence=any(e['decision']['human_evidence'] for e in events))
