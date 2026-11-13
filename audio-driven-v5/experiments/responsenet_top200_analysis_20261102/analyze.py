"""Exploratory ranking of four-checkpoint mean per-pair visual metrics."""
import csv,hashlib,json,math
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
BASE=Path(__file__).resolve().parent
RUN=ROOT/'experiments/p2_testsets_4ckpt_24gpu_20261102'
ARMS=['av_event_6000','av_event_8927','av_only_6000','av_only_8927']
DIRECTIONS={'psnr':1,'ssim':1,'lpips':-1,'emo_acc':1,'delta_resp':-1}
EMOTIONS=['angry','disgust','fear','happy','sad','surprise','neutral']

def read(p):return json.loads(p.read_text())
def rows(p):return [json.loads(x) for x in p.read_text().splitlines() if x]
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def jsonl(p,rs):p.write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in rs))
def percentile(values):return (pd.Series(values).rank(method='average').to_numpy()-1)/(len(values)-1)*100

def aggregate(rs):
 out={}
 for m in ('psnr','ssim','lpips'):
  out[m]=sum(r['metrics'][m]*r['media']['aligned_frames'] for r in rs)/sum(r['media']['aligned_frames'] for r in rs)
 out['emo_acc']=sum(r['emotion']['matches'] for r in rs)/sum(r['emotion']['valid_frames'] for r in rs)
 phases=[p for r in rs for p in r['response']['phases'] if p['delta'] is not None]
 out['delta_resp']=sum(p['delta']*p['duration'] for p in phases)/sum(p['duration'] for p in phases)
 out['emo_div_gt']=float(np.mean([r['metrics']['emo_div_gt'] for r in rs]))
 out['emo_div_generated']=float(np.mean([r['metrics']['emo_div_generated'] for r in rs]))
 return out

def main():
 protocol=read(RUN/'protocol.json');refpath=Path(protocol['datasets']['responsenet']['prepared'])/'references.jsonl'
 assert hashlib.sha256(refpath.read_bytes()).hexdigest()==protocol['datasets']['responsenet']['references_sha256']
 refs=rows(refpath);ref={r['id']:r for r in refs};ids=sorted(ref)
 assert len(ids)==len(refs)==586
 auditpath=ROOT/'experiments/planning_sft_responsenet_handoff_20261102/responsenet_gt_audio_audit.jsonl'
 audit={r['id']:r for r in rows(auditpath)};assert set(audit)==set(ids)
 inputs=[refpath,RUN/'RESULTS.json',auditpath];visual={};audio={}
 for arm in ARMS:
  paths=[RUN/'metrics'/arm/'responsenet'/f for f in ('pairs.jsonl','audio_pairs.jsonl')];inputs+=paths
  raw=rows(paths[0]);visual[arm]={r['id']:r for r in raw};assert len(raw)==len(visual[arm])==586 and set(visual[arm])==set(ids)
  raw=rows(paths[1]);audio[arm]={r['id']:r for r in raw};assert len(raw)==len(audio[arm])==586 and set(audio[arm])==set(ids)
  for sid in ids:
   r=visual[arm][sid];assert not r['errors'] and not audio[arm][sid]['errors']
   assert r['input']['gt']==ref[sid]['gt']
   assert all(abs(r['input'][k]-ref[sid][k])<1e-9 for k in ('source_start','source_end'))
   assert all(math.isfinite(r['metrics'][m]) for m in DIRECTIONS)
   assert bool(audio[arm][sid]['audio']['gt']['peak']>0)==audit[sid]['nonzero']
   for kind in ('acoustic','speech'):
    assert audio[arm][sid]['audio']['gt'][kind]==audio[ARMS[0]][sid]['audio']['gt'][kind]
  full=aggregate(list(visual[arm].values()));official=read(RUN/'RESULTS.json')[arm+'/responsenet']['metrics']
  assert all(abs(full[m]-official[m])<1e-9 for m in full),(arm,full,official)
 matrix=np.array([[[visual[a][sid]['metrics'][m] for m in DIRECTIONS] for a in ARMS] for sid in ids])
 mean=matrix.mean(axis=1);percentiles=np.stack([percentile(mean[:,i]*direction) for i,direction in enumerate(DIRECTIONS.values())],axis=1)
 scores=percentiles.mean(axis=1)
 perarm=np.stack([np.mean(np.stack([percentile(matrix[:,j,i]*direction) for i,direction in enumerate(DIRECTIONS.values())]),axis=0) for j in range(4)],axis=1)
 armranks=np.stack([pd.Series(-perarm[:,j]).rank(method='min').to_numpy() for j in range(4)],axis=1)
 records=[]
 for ix,sid in enumerate(ids):
  r=ref[sid];vs=[visual[a][sid] for a in ARMS];gt=audio[ARMS[0]][sid]['audio']['gt']
  hist=np.sum([v['emotion']['histogram_gt'] for v in vs],axis=0)
  motion=[];genmotion=[]
  for v in vs:
   phases=[p for p in v['response']['phases'] if p['delta'] is not None];den=sum(p['duration'] for p in phases)
   motion.append(sum(p['duration']*p['motion_gt'] for p in phases)/den)
   genmotion.append(sum(p['duration']*p['motion_generated'] for p in phases)/den)
  record={'id':sid,'source_id':r['sample_id'],'source_start':r['source_start'],'source_end':r['source_end'],'duration_s':r['duration_s'],
   'score':float(scores[ix]),'metric_mean':dict(zip(DIRECTIONS,map(float,mean[ix]))),'metric_percentiles':dict(zip(DIRECTIONS,map(float,percentiles[ix]))),
   'gt_nonzero':audit[sid]['nonzero'],'gt_acoustic':bool(gt['acoustic']),'gt_speech':bool(gt['speech']),
   'gt_speech_seconds':sum(b-a for a,b in gt['speech']),'gt_acoustic_seconds':sum(b-a for a,b in gt['acoustic']),
   'asr_group':r['group'],'gt_dominant_emotion_proxy':EMOTIONS[int(hist.argmax())],
   'gt_neutral_frame_fraction_proxy':float(hist[-1]/hist.sum()),'gt_emotion_diversity_mean':float(np.mean([v['metrics']['emo_div_gt'] for v in vs])),
   'gt_motion_proxy':float(np.mean(motion)),'generated_motion_proxy':float(np.mean(genmotion)),
   'gt_face_coverage':float(np.mean([v['face_detection']['gt_valid']/v['face_detection']['total'] for v in vs])),
   'generated_face_coverage':float(np.mean([v['face_detection']['generated_valid']/v['face_detection']['total'] for v in vs])),
   'per_arm_metrics':{a:visual[a][sid]['metrics'] for a in ARMS},
   'per_arm_score':dict(zip(ARMS,map(float,perarm[ix]))),'per_arm_rank':dict(zip(ARMS,map(int,armranks[ix]))),
   'gt_video':r['gt'],'generated_videos':{a:visual[a][sid]['input']['generated'] for a in ARMS},
   'generated_audios':{a:visual[a][sid]['input']['generated_audio'] for a in ARMS},
   'speaker_audio':r['speaker']['audio'],'gt_audio':r['listener']['audio']}
  records.append(record)
 records.sort(key=lambda r:(-r['score'],r['id']))
 for rank,r in enumerate(records,1):r['rank']=rank
 top=records[:200];topids={r['id'] for r in top};others=records[200:]
 motion_cut=np.quantile([r['gt_motion_proxy'] for r in records],[.25,.5,.75]);duration_cut=np.quantile([r['duration_s'] for r in records],[.25,.5,.75])
 def describe(group):
  counts=Counter(r['source_id'] for r in group);motion=np.array([r['gt_motion_proxy'] for r in group]);dur=np.array([r['duration_s'] for r in group])
  perf={a:aggregate([visual[a][r['id']] for r in group]) for a in ARMS}
  audio_stats={}
  for a in ARMS:
   audio_stats[a]={}
   for kind in ('acoustic','speech'):
    comps=[audio[a][r['id']]['audio']['comparison'][kind] for r in group]
    audio_stats[a][kind]={'gt_active_pairs':sum(c['any_reference_activity'] for c in comps),
     'missed_entire_response':sum(c['missed_entire_response'] for c in comps),
     'false_activity_on_silent_clip':sum(c['false_activity_on_silent_clip'] for c in comps),
     'gt_active_s':sum(c['gt_active_s'] for c in comps),'generated_active_s':sum(c['generated_active_s'] for c in comps),
     'extra_active_s':sum(c['extra_active_s'] for c in comps)}
  return {'pairs':len(group),'source_count':len(counts),'total_minutes':float(dur.sum()/60),'duration_mean_s':float(dur.mean()),'duration_median_s':float(np.median(dur)),
   'duration_min_max_s':[float(dur.min()),float(dur.max())],'duration_quartiles_global':np.bincount(np.searchsorted(duration_cut,dur,side='right'),minlength=4).tolist(),
   'score_range':[min(r['score'] for r in group),max(r['score'] for r in group)],'gt_zero':sum(not r['gt_nonzero'] for r in group),
   'gt_nonzero':sum(r['gt_nonzero'] for r in group),'gt_acoustic':sum(r['gt_acoustic'] for r in group),'gt_speech':sum(r['gt_speech'] for r in group),
   'gt_speech_total_s':sum(r['gt_speech_seconds'] for r in group),'asr_groups':dict(Counter(r['asr_group'] for r in group)),
   'dominant_emotion_proxy':dict(Counter(r['gt_dominant_emotion_proxy'] for r in group)),
   'gt_motion_mean':float(motion.mean()),'gt_motion_median':float(np.median(motion)),
   'gt_motion_quartiles_global':np.bincount(np.searchsorted(motion_cut,motion,side='right'),minlength=4).tolist(),
   'gt_emotion_diversity_mean':float(np.mean([r['gt_emotion_diversity_mean'] for r in group])),
   'gt_face_coverage_mean':float(np.mean([r['gt_face_coverage'] for r in group])),
   'generated_face_coverage_mean':float(np.mean([r['generated_face_coverage'] for r in group])),
   'top_sources':counts.most_common(12),'top5_source_share':sum(c for _,c in counts.most_common(5))/len(group),
   'metric_original_aggregation_per_arm':perf,'metric_original_aggregation_mean4':{m:float(np.mean([perf[a][m] for a in ARMS])) for m in perf[ARMS[0]]},
   'audio_per_arm':audio_stats}
 balanced=(percentiles[:,:3].mean(axis=1)+percentiles[:,3]+percentiles[:,4])/3
 balanced_ids={ids[i] for i in np.argsort(-balanced,kind='stable')[:200]}
 pct_first=perarm.mean(axis=1);pct_ids={ids[i] for i in np.argsort(-pct_first,kind='stable')[:200]}
 sensitivity={'balanced_visual_emotion_motion_top200_overlap':len(topids&balanced_ids),
  'percentile_per_arm_before_average_top200_overlap':len(topids&pct_ids),
  'top200_in_all_four_individual_top200':sum(all(v<=200 for v in r['per_arm_rank'].values()) for r in top),
  'individual_top200_overlaps':{a:sum(r['per_arm_rank'][a]<=200 for r in top) for a in ARMS}}
 summary={'method':{'arms':ARMS,'metric_directions':DIRECTIONS,'metric_weights':{m:.2 for m in DIRECTIONS},
  'formula':'average raw metric across 4 arms per pair; convert each oriented metric to tie-aware percentile 100*(average_rank-1)/(586-1); average five percentiles; tie-break by ID',
  'scope':'exploratory visual composite, not an official metric or frozen evaluation subset',
  'excluded_metrics':{'fid_fvd':'distribution metrics, no per-pair value','emo_div':'generated class count has no standalone higher-is-better relationship to GT','utmos':'all unavailable'},
  'aggregation_note':'Ranking uses pair-level means; comparison metrics below reproduce original frame/duration weighted aggregation, then average 4 arms.',
  'motion_proxy_note':'GT eyebrow/eye landmark step motion on common valid support, averaged over arms; not manual gesture labels.',
  'emotion_proxy_note':'DeepFace machine labels; not manual emotion ground truth.'},
  'all586':describe(records),'top200':describe(top),'remaining386':describe(others),'sensitivity':sensitivity,
  'motion_global_quartile_cutoffs':motion_cut.tolist(),'duration_global_quartile_cutoffs':duration_cut.tolist(),
  'top10_ids':[r['id'] for r in top[:10]],'input_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}}
 jsonl(BASE/'ranked_all586.jsonl',records);jsonl(BASE/'top200.jsonl',top)
 jsonl(BASE/'top200_references_analysis_only.jsonl',[ref[r['id']] for r in top])
 write(BASE/'summary.json',summary)
 fields=['rank','id','source_id','score','duration_s','source_start','source_end',*DIRECTIONS,'gt_nonzero','gt_acoustic','gt_speech','gt_motion_proxy','gt_dominant_emotion_proxy',*[a+'_rank' for a in ARMS]]
 with (BASE/'top200.csv').open('w',encoding='utf-8-sig',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
  for r in top:
   writer.writerow({k:({**r,**r['metric_mean'],**{a+'_rank':r['per_arm_rank'][a] for a in ARMS}})[k] for k in fields})
 write(BASE/'validation.json',{'passed':True,'arms':4,'pairs_per_arm':586,'unique_top_ids':len(topids),'all_5_metrics_finite':True,'GT_audio_intervals_identical_across_arms':True,'reproduced_original_aggregations':True,'original_files_not_modified':True})
 print(json.dumps({k:summary[k] for k in ('top200','sensitivity')},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
