"""Apply approved official union gate to the frozen 100-case review, without training mutations."""
import json,sys,hashlib,csv,shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import soundfile as sf
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from listen_benchmark.official_audio_gate import gate_audio,adapt_events,POLICY
from listen_benchmark.backchannel_face_stats import sha
SOURCE=ROOT/'experiments/listen_official_random100_20261026'
SOURCE_HTML=ROOT/'data_previews/listen_benchmark_review/audio_official_random100'
RUN=ROOT/'experiments/listen_official_gate100_20261026'
OUT=ROOT/'data_previews/listen_benchmark_review/audio_official_gate100'

def regions(items,sr,text):return [dict(id=f'{text}-{i}',start=a/sr,end=b/sr,text=text) for i,(a,b) in enumerate(items)]
def envelope(x,sr):
 step=round(.02*sr);return [round(float(np.max(abs(x[i:i+step]))),6) for i in range(0,len(x),step)]
def write_json(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2))

def main():
 RUN.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True);(RUN/'labels').mkdir(exist_ok=True)
 data=json.loads((SOURCE/'comparison.json').read_text());assert len(data['cases'])==100
 def work(c):
  sid=c['sample_id'];src=SOURCE_HTML/'media'/sid;dst=OUT/'media'/sid;dst.mkdir(parents=True,exist_ok=True)
  before=sha(src/'listener.wav');assert before==c['media_hashes']['target_audio']
  x,sr=sf.read(src/'listener.wav',dtype='float32');assert x.ndim==1 and sr==48000
  y,mask,gate=gate_audio(x,c['official']['listener'],sr)
  sf.write(dst/'processed.wav',y,sr,subtype='FLOAT');sf.write(dst/'removed.wav',x-y,sr,subtype='FLOAT')
  decoded,dsr=sf.read(dst/'processed.wav',dtype='float32');assert dsr==sr and len(decoded)==len(x)
  assert np.isfinite(decoded).all() and np.all(decoded[mask==0]==0);np.testing.assert_array_equal(decoded[mask==1],x[mask==1])
  residual,_=sf.read(dst/'removed.wav',dtype='float32');assert np.max(abs(decoded+residual-x))<1e-7
  assert sha(src/'listener.wav')==before
  # Originals and pictures are referenced from the earlier immutable review, not overwritten.
  label=json.loads((src/'label.json').read_text());assert hashlib.sha256((src/'label.json').read_bytes()).hexdigest()==c['label_sha256']
  derived=adapt_events(label['events'],mask,sr)
  original_visual=[e for e in label['events'] if e.get('modality')!='audio'];new_visual=[e for e in derived['events'] if e.get('modality')!='audio'];assert original_visual==new_visual
  derived.update(schema='official-gated-candidates-v1',sample_id=sid,source_label_sha256=c['label_sha256'],policy_version=POLICY['version'],source_audio_sha256=before,processed_audio_sha256=sha(dst/'processed.wav'),gate_metadata=str(dst/'gate.json'))
  write_json(RUN/'labels'/f'{sid}.json',derived);write_json(dst/'gated_candidates.json',derived);write_json(dst/'gate.json',gate)
  c['media']={k:'../audio_official_random100/'+v for k,v in c['media'].items()}
  c['media'].update(processed=f'media/{sid}/processed.wav',removed=f'media/{sid}/removed.wav')
  c['gate']=gate;c['gate']['source_audio_sha256']=before;c['gate']['processed_audio_sha256']=derived['processed_audio_sha256']
  c['gate']['intervals']=regions(gate['keep_intervals_samples'],sr,'保留区间（含保护边界）');c['gate']['core_intervals']=regions(gate['core_intervals_samples'],sr,'官方词/句/VAD并集')
  keep=gate['keep_intervals_samples'];gaps=[];left=0
  for a,b in keep:
   if a>left:gaps.append([left,a])
   left=b
  if left<len(x):gaps.append([left,len(x)])
  c['gate']['muted_intervals']=regions(gaps,sr,'置零区间')
  c['gate']['events']=derived['audio_event_gate_audit'];c['gate']['event_status_counts']={k:sum(e['status']==k for e in derived['audio_event_gate_audit']) for k in ['kept','partial','fully_muted']}
  c['gate']['label_url']=f'media/{sid}/gated_candidates.json';c['gate']['metadata_url']=f'media/{sid}/gate.json'
  c['waves']={k:envelope(z,sr) for k,z in [('original',x),('processed',y),('removed',x-y)]};c['wave_peak']=float(max(np.max(abs(x)),1e-8))
  c['verification']=dict(samples=len(x),sample_rate=sr,source_unchanged=True,output_duration_unchanged=True,outside_zero=True,unity_gain_samples_unchanged=True,residual_reconstruction_max_error=float(np.max(abs(decoded+residual-x))),visual_events_unchanged=True)
  return c
 with ThreadPoolExecutor(max_workers=8) as pool:
  cases=[]
  for i,c in enumerate(pool.map(work,data['cases']),1):
   cases.append(c)
   if i%20==0:print('gated',i,flush=True)
 summary=dict(samples=100,duration_s=sum(c['duration_s'] for c in cases),full_silence_clips=sum(c['gate']['full_silence'] for c in cases),clips_with_retained_audio=sum(not c['gate']['full_silence'] for c in cases),official_core_seconds=sum(c['gate']['core_samples']/c['gate']['sample_rate'] for c in cases),nominal_keep_seconds=sum(c['gate']['nominal_keep_samples']/c['gate']['sample_rate'] for c in cases),zeroed_seconds=sum(c['gate']['zero_samples']/c['gate']['sample_rate'] for c in cases),fade_seconds=sum(c['gate']['fade_samples']/c['gate']['sample_rate'] for c in cases),original_audio_events=sum(c['summary']['ours_audio_events'] for c in cases),audio_events_fully_muted=sum(c['gate']['event_status_counts']['fully_muted'] for c in cases),audio_events_partially_retained=sum(c['gate']['event_status_counts']['partial'] for c in cases),audio_events_fully_supported=sum(c['gate']['event_status_counts']['kept'] for c in cases),training_applied=False)
 data.update(schema='official-union-gate-review100-v1',run_id=RUN.name,gate_policy=POLICY,gate_summary=summary,cases=cases,source_comparison_sha256=sha(SOURCE/'comparison.json'))
 write_json(RUN/'results.json',data);write_json(RUN/'summary.json',summary);write_json(RUN/'policy.json',POLICY);write_json(RUN/'verification.json',[dict(sample_id=c['sample_id'],**c['verification']) for c in cases]);write_json(OUT/'review.json',data)
 with (RUN/'summary.csv').open('w') as f:
  keys=['sample_rank','sample_id','duration_s','full_silence','nominal_keep_seconds','zero_seconds','kept_events','partial_events','muted_events'];w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
  for c in cases:
   g=c['gate'];w.writerow(dict(sample_rank=c['sample_rank'],sample_id=c['sample_id'],duration_s=c['duration_s'],full_silence=g['full_silence'],nominal_keep_seconds=g['nominal_keep_samples']/g['sample_rate'],zero_seconds=g['zero_samples']/g['sample_rate'],kept_events=g['event_status_counts']['kept'],partial_events=g['event_status_counts']['partial'],muted_events=g['event_status_counts']['fully_muted']))
 shutil.copy2(RUN/'summary.csv',OUT/'summary.csv')
 from fontTools.ttLib import TTFont
 from fontTools import subset
 html=(ROOT/'scripts/listener_official_gate100.html').read_text();(OUT/'index.html').write_text(html)
 font=TTFont('/tmp/v8_browser_libs/extracted/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',fontNumber=2);sub=subset.Subsetter();sub.populate(text=html+json.dumps(data,ensure_ascii=False));sub.subset(font);font.flavor='woff';font.save(OUT/'review.woff')
 print(json.dumps(summary,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
