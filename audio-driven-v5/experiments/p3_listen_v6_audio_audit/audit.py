from pathlib import Path
import json,sys,tarfile,os
os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np
import soundfile as sf
from scipy import signal
ROOT=Path('/private/liqi_workspace/project_code/audio-driven-v5')
sys.path.insert(0,str(ROOT/'src'))
from p3_listen import media
OUT=ROOT/'experiments/p3_listen_v6_audio_audit'
preview=ROOT/'data_previews/seamless_listen_v6_review'
samples=json.loads((preview/'samples.json').read_text())['samples']
row=next(s for s in samples if s['sample_id']=='naturalistic_V03_S1324_I00000303_v6_p014_03')
folder=preview/row['preview']
norm=json.loads((ROOT/'data_corpora/seamless_listen_300h/normalized_pairs'/f"{row['pair']}.json").read_text())
def db(x):return float(20*np.log10(max(float(x),1e-15)))
def stats(x):return dict(rms_dbfs=db(np.sqrt(np.mean(x*x))),peak_dbfs=db(np.max(np.abs(x))))
tracks={};results={}
for role in ['listener','speaker']:
 src=row['sources'][role]
 with tarfile.open(src['archive'],'r:') as tar:
  with sf.SoundFile(tar.extractfile(Path(src['member']).with_suffix('.wav').name)) as f:
   sr=f.samplerate;channels=f.channels;f.seek(round(row['start']*sr));raw=f.read(round(row['duration_s']*sr),dtype='float64')
 assert sr==48000 and channels==1
 sf.write(OUT/f'{role}_raw.wav',raw,sr,subtype='PCM_24')
 x,sr=sf.read(folder/f'{role}.wav',dtype='float64')
 tr=norm['tracks'][row[role+'_index']]
 with sf.SoundFile(tr['path']) as f:
  f.seek(round(row['start']*sr));original_norm=f.read(len(x),dtype='float64')
 gain=10**(tr['gain_db']/20);matched=raw*gain
 sf.write(OUT/f'{role}_raw_gain_matched.wav',matched,sr,subtype='FLOAT')
 fitted=float(np.dot(raw,x)/np.dot(raw,raw));res=x-fitted*raw
 fq,power=signal.welch(raw,sr,nperseg=8192)
 bands={f'below_{cut}_hz':float(power[fq<cut].sum()/power.sum()) for cut in [80,150,300,1000]}
 block=4800;n=len(raw)//block
 energies=np.sqrt(np.mean(raw[:n*block].reshape(n,block)**2,axis=1))
 peaks=signal.find_peaks(energies,distance=5)[0]
 peaks=sorted(peaks,key=lambda p:energies[p],reverse=True)[:6]
 media.command(['ffmpeg','-v','error','-nostdin','-y','-threads','1','-i',str(folder/f'{role}.mp4'),'-map','0:a:0','-ar','48000','-c:a','pcm_f32le',str(OUT/f'{role}_aac_decoded.wav')])
 aac,_=sf.read(OUT/f'{role}_aac_decoded.wav');aac=aac[:len(x)]
 result=dict(original=stats(raw),normalized=stats(x),gain_db=tr['gain_db'],amplitude_multiplier=gain,
  fitted_gain_db=db(fitted),raw_normalized_correlation=float(np.corrcoef(raw,x)[0,1]),
  residual_relative_db=db(np.linalg.norm(res)/np.linalg.norm(x)),exact_normalized_slice=bool(np.array_equal(x,original_norm)),
  source_frequency_energy=bands,top_noise_or_activity_times=[dict(start=round(p*.1,2),end=round((p+1)*.1,2),raw_rms_dbfs=db(energies[p]),normalized_rms_dbfs=db(energies[p]*gain)) for p in peaks],
  aac_correlation=float(np.corrcoef(x[:len(aac)],aac)[0,1]),aac_error_relative_db=db(np.linalg.norm(aac-x[:len(aac)])/np.linalg.norm(x[:len(aac)])))
 results[role]=result;tracks[role]=raw
# Cross-channel waveform similarity after highpass: evidence only, not a speech-separation score.
sos=signal.butter(4,200,fs=48000,btype='highpass',output='sos')
a=signal.sosfilt(sos,tracks['speaker'])[::3];b=signal.sosfilt(sos,tracks['listener'])[::3]
c=signal.correlate(a,b,mode='full',method='fft');lags=signal.correlation_lags(len(a),len(b));keep=np.abs(lags)<=3200
sel=np.argmax(np.abs(c[keep]));corr=float(c[keep][sel]/np.sqrt(np.dot(a,a)*np.dot(b,b)))
results['cross_channel']=dict(max_highpass_correlation_within_200ms=corr,lag_seconds=float(lags[keep][sel]/16000))
results['sample']=row;results['interpretation']='Objective signal comparison; noise identity is not confirmed by listening.'
(OUT/'report.json').write_text(json.dumps(results,indent=2,ensure_ascii=False)+'\n')
print(json.dumps(results,ensure_ascii=False,indent=2),flush=True)
