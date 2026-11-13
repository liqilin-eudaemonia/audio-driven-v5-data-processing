"""Build an audible, aligned review of existing fixed/guided generation outputs."""
import concurrent.futures
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'experiments/p2_av_review_20261027';sys.path.insert(0,str(ROOT/'src'))
from p2_listen.contracts import read,sha,digest,atomic_json
from p2_glare_eval.common import job_key
ARMS=['E0','E1','E0F','E1F'];NAMES={'E0':'AV','E1':'AV＋Event','E0F':'AV＋Feature','E1F':'AV＋Event＋Feature'}
EN_COORD=ROOT/'experiments/p2_guided_en_full_eval_20261027'
USE_ENGLISH=(EN_COORD/'publication_ready.json').exists() and read(EN_COORD/'publication_ready.json').get('complete') is True
GUIDED_COORD=EN_COORD if USE_ENGLISH else ROOT/'experiments/p2_guided_full_eval_20261027'
MEDIA=OUT/'media';MEDIA.mkdir(parents=True,exist_ok=True)


def link(path, suffix=None):
 path=Path(path).resolve();name=hashlib.sha256(str(path).encode()).hexdigest()[:24]+(suffix or path.suffix);dest=MEDIA/name
 if dest.is_symlink():
  if dest.resolve()!=path:raise ValueError('Media link identity differs')
 elif dest.exists():raise ValueError('Unexpected media file collision')
 else:dest.symlink_to(path)
 return 'media/'+name


def waveform(path,duration,rate=48000):
 value,sr=sf.read(path,dtype='float32',always_2d=True)
 if value.shape[1]==1:value=np.repeat(value,2,axis=1)
 if value.shape[1]!=2:raise ValueError('Expected mono/stereo audio')
 if sr!=rate:
  g=math.gcd(sr,rate);value=resample_poly(value,rate//g,sr//g,axis=0).astype(np.float32)
 count=round(duration*rate);return np.pad(value,((0,max(0,count-len(value))),(0,0)))[:count]


def aligned_mix(listener,speaker,duration):
 # Original relative level: no listener normalization or amplification.
 a=waveform(listener,duration);b=waveform(speaker,duration);mixed=a+b;peak=float(np.max(abs(mixed)));gain=min(1.,.95/max(peak,1e-12))
 return mixed*gain,dict(alignment='sample zero aligned; no offsets',listener_gain=gain,speaker_gain=gain,peak_before_gain=peak,samples=len(mixed),rate=48000)


def mux(video,audio,duration,speaker=None):
 ident=dict(video=str(video),video_sha256=sha(video),audio=str(audio),audio_sha256=sha(audio),duration_s=duration,
            speaker=str(speaker) if speaker else None,speaker_sha256=sha(speaker) if speaker else None,
            policy='equal-gain-addition-headroom095-v1' if speaker else 'exact-target-audio-remux-v1')
 key=digest(ident);dest=MEDIA/(key+'.mp4');marker=MEDIA/(key+'.json')
 if marker.exists():
  m=read(marker)
  if m['identity']!=ident or m['sha256']!=sha(dest):raise ValueError('Review media cache changed')
  return 'media/'+dest.name
 with tempfile.TemporaryDirectory(prefix='p2_review_mix_') as temp:
  if speaker:
   wave,mix=aligned_mix(audio,speaker,duration);wav=Path(temp)/'mix.wav';sf.write(wav,wave,48000,subtype='FLOAT')
  else:wav=Path(audio);mix=None
  target=Path(temp)/'video.mp4'
  command=['ffmpeg','-v','error','-nostdin','-threads','1','-i',str(video),'-i',str(wav),'-map','0:v:0','-map','1:a:0','-c:v','copy','-c:a','aac','-b:a','192k','-threads','1','-t',str(duration),'-movflags','+faststart',str(target)]
  subprocess.run(command,check=True,capture_output=True)
  streams=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries','stream=codec_type','-of','json',str(target)],text=True))['streams']
  if {s['codec_type'] for s in streams}!={'video','audio'}:raise ValueError('Missing AV stream')
  shutil.copyfile(target,dest)
 atomic_json(marker,dict(identity=ident,mix=mix,sha256=sha(dest)))
 return 'media/'+dest.name


def metric_rows(comp,values,generated,matched=None):
 rows=[]
 for arm in ARMS:
  scope=('固定/指导共八个条件、两个seed共同有效样本；各seed分别计算后等权平均；支持数：'+str(matched['coverage'])) if matched else '四组共同有效样本；两个生成seed分别计算后等权平均'
  rows.append(dict(id=arm+'-fixed',name=NAMES[arm],variant='固定prompt',outputs=48,values=matched['values'][arm+'-fixed'] if matched else values[arm],scope=scope))
  count=sum(x['arm']==arm and x['variant']=='guided' for x in generated)
  if count or (ROOT/'experiments/p2_guided_full_eval_20261027/jobs.json').exists():rows.append(dict(id=arm+'-guided',name=NAMES[arm],variant='英文标注指导' if USE_ENGLISH else '原文标注指导',outputs=count if count==48 else f'{count}/48',values=matched['values'][arm+'-guided'] if matched else {},scope=scope if matched else '完整生成及评测正在补齐'))
 if matched:
  for row in rows:
   arm,variant=row['id'].split('-')
   section=matched.get('within_'+variant)
   row['within_values']=section['values'][arm] if section else (values[arm] if variant=='fixed' else {})
   row['within_scope']='仅此prompt内四组权重及双seed的共同支持；不同prompt的支持集合不能直接作差'
   diagnostic=matched.get('glare_pair_coverage',{}).get(row['id'],{})
   row['glare_valid_pairs']=diagnostic.get('valid_pairs',diagnostic.get('strict_valid_pairs'))
   row['glare_diagnostic']=diagnostic
 return rows


def make_dataset(kind):
 base=ROOT/f'experiments/p2_{kind}24_fourway_eval_20261027';obs=read(base/'observations.json');targets={r['sample_id']:r for r in read(base/'targets.json')};metrics={a:{(r['sample_id'],r['sample_seed']):r for r in read(base/a/'omni/results.json')['rows']} for a in ARMS};comp=read(base/'comparison.json')['metrics'];values=read(base/'table_values.json')
 guided=read(ROOT/'experiments/p2_guided_prompt_train24_20261027/validation_summary.json')['rows'] if kind=='train' else []
 guided=[r for r in guided if r['variant']=='guided'];samples=[];tasks=[]
 full=ROOT/f'experiments/p2_guided_{"en_" if USE_ENGLISH else ""}{kind}24_fourway_eval_20261027';comparison_path=full/('comparison_rgb_v2.json' if (full/'comparison_rgb_v2.json').exists() else 'comparison.json');matched=read(comparison_path) if comparison_path.exists() else None;guided_metrics={}
 jobs_path=GUIDED_COORD/'jobs.json'
 if jobs_path.exists():
  from p2_glare_eval.common import artifact_valid
  guided=[]
  for job in read(jobs_path):
   if job['kind']!=kind or not artifact_valid(job['output'],job['identity']):continue
   wave,_=sf.read(Path(job['output'])/'listener.wav');rms=float(np.sqrt(np.mean(wave*wave)))
   guided.append(dict(**job,variant='guided',rms=rms,rms_dbfs=20*math.log10(max(rms,1e-15))))
  for arm in ARMS:
   path=full/arm/'omni/results.json'
   if path.exists():guided_metrics[arm]={(r['sample_id'],r['sample_seed']):r for r in read(path)['rows']}
  if matched:
   expected={(o['sample_id'],seed) for o in obs for seed in (137,251)}
   for arm in ARMS:
    actual={(r['sample_id'],r['seed']) for r in guided if r['arm']==arm}
    if actual!=expected or set(guided_metrics.get(arm,{}))!=expected:
     raise ValueError('Completed comparison requires all 48 guided outputs and metric records: '+kind+' '+arm)
 for i,o in enumerate(obs):
  sid=o['sample_id'];t=targets[sid];assert sha(o['speaker_audio'])==o['media_hashes']['speaker_audio'];assert sha(t['target_audio'])==t['media_hashes']['target_audio'];assert sha(t['target_video'])==t['media_hashes']['target_video']
  asr_file=OUT/'speaker_asr'/(o['media_hashes']['speaker_audio']+'.json');asr=read(asr_file) if asr_file.exists() else None
  if asr and asr['sha256']!=o['media_hashes']['speaker_audio']:raise ValueError('Speaker ASR identity differs')
  s=dict(index=i+1,id=sid,duration=o['requested_duration_s'],image=link(o['listener_image']),speaker=link(o['speaker_audio']),asr=asr,
         gt_audio=link(t['target_audio']),gt_label='训练实际listener目标' if kind=='train' else '独立test原始listener参考（未用于训练）',
         sources=dict(speaker=o['speaker_audio'],gt_video=t['target_video'],gt_audio=t['target_audio']),outputs=[])
  tasks.append((s,'gt_video',t['target_video'],t['target_audio'],s['duration'],None))
  if kind=='test':
   processed=next(r for r in read(base/'processed_targets.json') if r['sample_id']==sid);s['metric_gt_audio']=link(processed['target_audio'])
  for seed in [137,251]:
   for arm in ARMS:
    p=base/arm/'generation/generated'/job_key(sid,seed);r=metrics[arm][sid,seed];mr=r['metrics'];sync=mr['sync']['generated'];utmos=mr['utmos']['generated'];wave,_=sf.read(p/'listener.wav');rms=float(np.sqrt(np.mean(wave*wave)))
    entry=dict(arm=arm,name=NAMES[arm],variant='fixed',seed=seed,video=link(p/'generated.mp4'),audio=link(p/'listener.wav'),
               metrics=dict(utmos=utmos.get('value'),lse_d=sync.get('lse_d'),lse_c=sync.get('lse_c'),rms_dbfs=20*math.log10(max(rms,1e-15))),rms=rms,metric_status='existing_evaluation',prompt=None)
    s['outputs'].append(entry);tasks.append((entry,'mixed_video',str(p/'generated.mp4'),str(p/'listener.wav'),s['duration'],o['speaker_audio']))
  for row in guided:
   if row['sample_id']!=sid:continue
   p=Path(row['output']);entry=dict(arm=row['arm'],name=NAMES[row['arm']],variant='guided',seed=row['seed'],video=link(p/'generated.mp4'),audio=link(p/'listener.wav'),
                                 metrics=dict(utmos=None,lse_d=None,lse_c=None,rms_dbfs=row['rms_dbfs']),rms=row['rms'],metric_status='not_evaluated',prompt=(p/'prompt.txt').read_text())
   if USE_ENGLISH:
    from p2_guided_prompt_en.formatting import validate_guidance,is_english_script
    record_path=Path(row['work'])/'prompts'/(sid+'.json');record=validate_guidance(read(record_path))
    assert is_english_script(entry['prompt']) and entry['prompt']==record['prompt']
    entry.update(guidance_language='English',translation_records=record['translations'],guidance_record=link(record_path))
   score=guided_metrics.get(row['arm'],{}).get((sid,row['seed']))
   if score:
    mr=score['metrics'];entry['metrics'].update(utmos=mr['utmos']['generated'].get('value'),lse_d=mr['sync']['generated'].get('lse_d'),lse_c=mr['sync']['generated'].get('lse_c'));entry['metric_status']='evaluated'
    entry['metric_details']={k:{s:mr[k][s] for s in ('gt','generated')} for k in ('utmos','sync')}
   s['outputs'].append(entry);tasks.append((entry,'mixed_video',str(p/'generated.mp4'),str(p/'listener.wav'),s['duration'],o['speaker_audio']))
  samples.append(s)
 with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
  futures={pool.submit(mux,v,a,d,sp):(obj,key) for obj,key,v,a,d,sp in tasks}
  for n,f in enumerate(concurrent.futures.as_completed(futures),1):
   obj,key=futures[f];obj[key]=f.result()
   if n%40==0:print(kind,'review media',n,'/',len(tasks),flush=True)
 old_coverage=dict(fvd=comp['fvd']['samples_per_seed'],utmos=comp['utmos']['samples_per_seed'],sync=comp['sync']['samples_per_seed'],glare=comp['glare']['scope']['samples_per_seed'])
 protocols={}
 if matched:
  for mode,filename,reportname in [('strict','comparison_rgb_v2.json','report_rgb_v2.md'),('unfiltered','comparison_unfiltered_v1.json','report_unfiltered_v1.md')]:
   path=full/filename
   if not path.exists():continue
   value=read(path)
   protocols[mode]=dict(metrics=metric_rows(comp,values,guided,value),coverage=value['coverage'],
       within_coverage=dict(fixed=value.get('within_fixed',{}).get('coverage',old_coverage),guided=value['within_guided']['coverage']),
       detector_version=value['detector_version'],report=link(full/reportname),comparison=link(path))
 return dict(id=kind,title='原训练集24条' if kind=='train' else '独立test24条',samples=samples,
             guidance_language='English' if USE_ENGLISH else 'mixed',glare_protocols=protocols,
             metrics=metric_rows(comp,values,guided,matched),coverage=matched['coverage'] if matched else dict(fvd=comp['fvd']['samples_per_seed'],utmos=comp['utmos']['samples_per_seed'],sync=comp['sync']['samples_per_seed'],glare=comp['glare']['scope']['samples_per_seed']),
             comparison_scope='eight_conditions' if matched else 'four_fixed_conditions',
             detector_version=matched.get('detector_version','bgr_v1') if matched else 'bgr_v1',
             within_coverage=dict(fixed=matched.get('within_fixed',{}).get('coverage',old_coverage),guided=matched['within_guided']['coverage']) if matched else None,
             report=link(full/('report_rgb_v2.md' if matched.get('detector_version')=='rgb_v2' else 'report.md')) if matched else link(base/'report.md'),comparison=link(comparison_path) if matched else link(base/'comparison.json'))


def main():
 datasets=[make_dataset(k) for k in ['train','test']];atomic_json(OUT/'data.json',dict(schema='p2-av-review-v2',datasets=datasets))
 assets=ROOT/'scripts/p2_review_assets'
 for name in ['index.html','app.js','style.css']:shutil.copy2(assets/name,OUT/name)
 summary=dict(samples=sum(len(d['samples']) for d in datasets),outputs=sum(len(s['outputs']) for d in datasets for s in d['samples']),asr=sum(s['asr'] is not None for d in datasets for s in d['samples']),media_files=len(list(MEDIA.glob('*.mp4'))))
 atomic_json(OUT/'build_summary.json',summary);print(summary)
if __name__=='__main__':main()
