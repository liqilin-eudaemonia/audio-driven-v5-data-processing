from pathlib import Path
import json
import numpy as np
import soundfile as sf
from scipy import signal,stats
R=Path('/private/liqi_workspace/project_code/audio-driven-v5');P=R/'data_previews/seamless_listen_v6_audio_ab'
d=json.loads((P/'samples.json').read_text());rows=d['samples'];out=[]
for idx in [0,1,10,11,12,13]:
 r=rows[idx];x,sr=sf.read(P/r['path']/'baseline.wav')
 events=r['events']
 if idx==1: events=events[:1]+events[-1:]
 elif idx>=10: events=[e for e in events if abs(e['start']-.75)<.01]
 for ev in events:
  a,b=ev['start'],ev['end'];y=x[round(a*sr):round(b*sr)];y=y-y.mean()
  fq,p=signal.welch(y,sr,nperseg=min(2048,len(y)))
  _,_,z=signal.stft(y,sr,nperseg=960,noverlap=480);power=abs(z)**2
  frame_rms=np.sqrt(np.mean(y[:len(y)//480*480].reshape(-1,480)**2,axis=1))
  p=p+1e-30
  out.append(dict(sample=r['sample_id'],start=a,end=b,duration=b-a,
   provisional_note='transcribed-feedback comparison; clip also has user-reported noise' if idx==1 and a<5 else 'user-reported-noisy clip; precise event identity not independently labeled',
   rms_dbfs=float(20*np.log10(np.sqrt(np.mean(y*y))+1e-15)),
   crest_db=float(20*np.log10(np.max(abs(y))/(np.sqrt(np.mean(y*y))+1e-15))),
   kurtosis=float(stats.kurtosis(y,fisher=False)),
   low150=float(p[fq<150].sum()/p.sum()),low1000=float(p[fq<1000].sum()/p.sum()),
   flatness=float(np.exp(np.mean(np.log(p)))/p.mean()),
   top5_bins_fraction=float(np.sort(p)[-5:].sum()/p.sum()),
   maximum_frame_rise_db=float(np.max(np.diff(20*np.log10(frame_rms+1e-10))))))
(R/'experiments/p3_listen_v6_audio_quality/features.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
for v in out:print(v)
