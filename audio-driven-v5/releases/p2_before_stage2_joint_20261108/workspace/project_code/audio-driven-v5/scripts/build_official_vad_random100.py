"""Uniform 100 new train50h clips, official paired metadata comparison, review only."""
import csv,hashlib,json,random,shutil,sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import soundfile as sf
import compare_listener_official_six as cmp
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from listen_benchmark.backchannel_face_stats import SourceReader,sha
DATASET=ROOT/'experiments/p2_train50h_backchannel_face_20261026'
RUN=ROOT/'experiments/listen_official_random100_20261026'
OUT=ROOT/'data_previews/listen_benchmark_review/audio_official_random100'
SEED=2026102602

def merge_intervals(items,duration):
 out=[]
 for a,b in sorted((max(0,x['start']),min(duration,x['end'])) for x in items):
  if b<=a:continue
  if out and a<=out[-1][1]:out[-1][1]=max(out[-1][1],b)
  else:out.append([a,b])
 return out

def comparisons(label,off):
 es=[];events=[e for e in label['events'] if e.get('modality')=='audio']
 for e in events:
  a=dict(e);cp={}
  for role in ['listener','speaker']:
   cp[role]={k:cmp.match(e,off[role][k]) for k in ['words','sentences','vad']}
   cv=cmp.covered(e,off[role]['vad']);cp[role].update(vad_coverage_s=round(cv,6),vad_coverage_fraction=round(cv/max(e['end']-e['start'],1e-9),4))
  lw,lv=cp['listener']['words'],cp['listener']['vad'];wo=any(x['kind']=='overlap' for x in lw);vo=any(x['kind']=='overlap' for x in lv)
  cp['status']='word_overlap' if wo else 'vad_only_overlap' if vo else 'no_listener_overlap'
  cp['near_listener_evidence']=any(x['kind']=='nearby_only' for x in lw+lv)
  cp['lexical_equal_word_ids']=[w['id'] for w in lw if e.get('text') and cmp.textnorm(e['text'])==cmp.textnorm(w['text'] or '')]
  cp['speaker_overlap_without_listener_overlap']=not wo and not vo and any(x['kind']=='overlap' for x in cp['speaker']['vad'])
  cp['other_our_event_ids_nearby']=[z['event_id'] for z in events if z is not e and cmp.gap(e,z)<=cmp.TOL+1e-8]
  cp['decision']='undetermined_difference_not_noise_label';a['comparison']=cp;es.append(a)
 other=[e for e in label['events'] if e.get('modality')!='audio'];missed=[]
 for w in off['listener']['words']:
  if not w['in_clip']:continue
  w['our_event_matches']=cmp.match(w,es)
  if not any(x['kind']=='overlap' for x in w['our_event_matches']):missed.append(w['id'])
 s=dict(ours_audio_events=len(es),ours_visual_events_excluded=len(other),official_listener_words=sum(w['in_clip'] for w in off['listener']['words']),official_listener_sentences=sum(w['in_clip'] for w in off['listener']['sentences']),official_listener_vad_segments=sum(w['in_clip'] for w in off['listener']['vad']),word_overlap=sum(e['comparison']['status']=='word_overlap' for e in es),vad_only_overlap=sum(e['comparison']['status']=='vad_only_overlap' for e in es),no_listener_overlap=sum(e['comparison']['status']=='no_listener_overlap' for e in es),official_word_ids_without_our_overlap=missed,ours_audio_events_without_vad_overlap=sum(e['comparison']['listener']['vad_coverage_s']==0 for e in es),official_words_without_vad_overlap=sum(w['in_clip'] and cmp.covered(w,off['listener']['vad'])==0 for w in off['listener']['words']))
 return es,other,s

def main():
 RUN.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True);(RUN/'raw_metadata').mkdir(exist_ok=True)
 complete=json.loads((DATASET/'COMPLETE.json').read_text())
 for name in ['samples.csv','manifests/observations.json','manifests/targets.json','summary.json']:
  assert sha(DATASET/name)==complete['files'][name],name
 rows=list(csv.DictReader((DATASET/'samples.csv').open()));byid={r['sample_id']:r for r in rows};assert len(rows)==len(byid)==30415 and all(r['split']=='train' for r in rows)
 previous=json.loads((ROOT/'data_previews/listen_benchmark_review/train50h_random/review.json').read_text());excluded={c['id'] for c in previous['samples']}
 ids=random.Random(SEED).sample(sorted(set(byid)-excluded),100)
 sampling=dict(seed=SEED,method='uniform_without_replacement_from_sorted_ids',source_count=len(rows),excluded_previous_review_ids=sorted(excluded),eligible_count=len(set(byid)-excluded),sample_ids_in_draw_order=ids,quality_or_vad_or_event_filter_applied=False,draw_count=100,dataset_completion_sha256=sha(DATASET/'COMPLETE.json'))
 if (RUN/'sampling.json').exists():assert json.loads((RUN/'sampling.json').read_text())==sampling
 (RUN/'sampling.json').write_text(json.dumps(sampling,indent=2))
 obs={x['sample_id']:x for x in json.loads((DATASET/'manifests/observations.json').read_text())};targets={x['sample_id']:x for x in json.loads((DATASET/'manifests/targets.json').read_text())}
 reader=SourceReader(ROOT/'experiments/p2_features_full_crop384_20261025');jobs=[]
 for rank,sid in enumerate(ids,1):
  row=byid[sid]
  with Path(row['label_file']).open('rb') as f:f.seek(int(row['label_offset']));raw=f.read(int(row['label_bytes']))
  assert hashlib.sha256(raw).hexdigest()==row['label_sha256'];label=json.loads(raw);assert label['sample_id']==sid
  full=reader.get(label['source']);assert full['split']=='train';jobs.append((rank,row,label,raw,full,obs[sid],targets[sid]))
 reader.close()
 def build(job):
  rank,row,label,labelraw,full,o,t=job;sid=row['sample_id'];duration=float(o['requested_duration_s']);start=full['start'];off={};provenance={}
  assert o['official_split']==o['experiment_split']=='train'
  dest=OUT/'media'/sid;dest.mkdir(parents=True,exist_ok=True)
  for filename,path,expected in [('listener.wav',t['target_audio'],t['media_hashes']['target_audio']),('speaker.wav',o['speaker_audio'],o['media_hashes']['speaker_audio']),('video.mp4',t['target_video'],t['media_hashes']['target_video']),('poster.png',o['listener_image'],o['media_hashes']['listener_image'])]:
   assert sha(Path(path))==expected;shutil.copy2(path,dest/filename);assert sha(dest/filename)==expected
  for filename in ['listener.wav','speaker.wav']:
   info=sf.info(dest/filename);assert info.samplerate==48000 and abs(info.frames/info.samplerate-duration)<1/48000+1e-9
  (dest/'label.json').write_bytes(labelraw)
  for role in ['listener','speaker']:
   raw,p=cmp.load_metadata(full['sources'][role]);off[role]=cmp.official(raw,role,start,duration);provenance[role]=p
   (RUN/'raw_metadata'/f'{sid}_{role}.json').write_text(json.dumps(raw,ensure_ascii=False))
  events,other,summary=comparisons(label,off)
  merged=merge_intervals(off['listener']['vad'],duration);gaps=[];left=0
  for a,b in merged:
   if a>left+1e-6:gaps.append([left,a])
   left=max(left,b)
  if left<duration-1e-6:gaps.append([left,duration])
  iv=[dict(id=f'vad-union-{i}',start=a,end=b,text='官方 VAD 内') for i,(a,b) in enumerate(merged)]
  ov=[dict(id=f'vad-gap-{i}',start=a,end=b,text='VAD 外，待复核') for i,(a,b) in enumerate(gaps)]
  vadseconds=sum(b-a for a,b in merged);summary.update(vad_union_seconds=vadseconds,vad_outside_seconds=duration-vadseconds,vad_coverage_fraction=vadseconds/duration)
  result=dict(sample_id=sid,short_id=f'{rank:03d} · '+sid.split('_')[2]+' · '+sid.split('_')[3],sample_rank=rank,human_group='unreviewed',clip_absolute_start=start,clip_absolute_end=start+duration,duration_s=duration,official=off,ours_audio_events=events,ours_other_events=other,summary=summary,vad_inside_intervals=iv,vad_outside_intervals=ov,provenance=provenance,selection_reason=row['selection_reason'],face_p10=float(row['face_short_side_p10']),face_p50=float(row['face_short_side_p50']),label_sha256=row['label_sha256'],media_hashes=dict(**t['media_hashes'],**o['media_hashes']),media=dict(video=f'media/{sid}/video.mp4',poster=f'media/{sid}/poster.png',listener=f'media/{sid}/listener.wav',speaker=f'media/{sid}/speaker.wav'))
  return result
 with ThreadPoolExecutor(max_workers=8) as pool:
  cases=[]
  for i,c in enumerate(pool.map(build,jobs),1):
   cases.append(c)
   if i%20==0:print('built',i,flush=True)
 agg=dict(samples=len(cases),duration_s=sum(c['duration_s'] for c in cases),audio_events=sum(c['summary']['ours_audio_events'] for c in cases),clips_without_vad=sum(c['summary']['official_listener_vad_segments']==0 for c in cases),clips_without_audio_events=sum(c['summary']['ours_audio_events']==0 for c in cases),audio_events_without_vad_overlap=sum(c['summary']['ours_audio_events_without_vad_overlap'] for c in cases),official_words_without_vad_overlap=sum(c['summary']['official_words_without_vad_overlap'] for c in cases),human_reviewed=0,noise_or_vad_accuracy_not_estimated=True)
 data=dict(schema='listener-official-random100-v1',run_id=RUN.name,sampling=sampling,tolerance_s=.2,aggregate=agg,cases=cases)
 (RUN/'comparison.json').write_text(json.dumps(data,ensure_ascii=False,indent=2));(OUT/'review.json').write_text(json.dumps(data,ensure_ascii=False))
 with (RUN/'summary.csv').open('w') as f:
  keys=['sample_rank','sample_id','duration_s','ours_audio_events','official_listener_words','official_listener_vad_segments','vad_union_seconds','ours_audio_events_without_vad_overlap','official_words_without_vad_overlap'];w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
  for c in cases:w.writerow({k:c.get(k,c['summary'].get(k)) for k in keys})
 shutil.copy2(RUN/'summary.csv',OUT/'summary.csv')
 from fontTools.ttLib import TTFont
 from fontTools import subset
 html=(ROOT/'scripts/listener_official_random100.html').read_text();(OUT/'index.html').write_text(html)
 font=TTFont('/tmp/v8_browser_libs/extracted/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',fontNumber=2);sub=subset.Subsetter();sub.populate(text=html+json.dumps(data,ensure_ascii=False));sub.subset(font);font.flavor='woff';font.save(OUT/'review.woff')
 (RUN/'build_check.json').write_text(json.dumps(dict(samples=100,unique_ids=len(set(ids)),overlap_with_previous=len(set(ids)&excluded),all_train=True,dataset_hashes_verified=True,source_label_hashes_verified=True,source_and_preview_media_hashes_verified=True,wav_lengths_match=True),indent=2))
 print(json.dumps(agg,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
