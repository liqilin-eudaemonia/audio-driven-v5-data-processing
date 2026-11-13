"""Compare six existing listener timelines against official paired transcript/VAD.
No relabeling or quality decisions; all source text and absolute times are retained.
"""
import json,tarfile,sys,hashlib,csv,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from listen_benchmark.backchannel_face_stats import SourceReader
RUN=ROOT/'experiments/listen_official_compare6_20261026'
SOURCE=ROOT/'experiments/listen_audio_cleanup_6_20261026'
PREVIEW=ROOT/'data_previews/listen_benchmark_review/audio_official_compare_6'
TOL=.2

def intersect(a,b):return max(0.,min(a['end'],b['end'])-max(a['start'],b['start']))
def gap(a,b):return max(0.,max(a['start'],b['start'])-min(a['end'],b['end']))
def textnorm(s):return re.sub(r'[^\w]+','',s.casefold())
def match(a,items):
 return [dict(id=b.get('id',b.get('event_id')),start=b['start'],end=b['end'],text=b.get('text'),overlap_s=round(intersect(a,b),6),gap_s=round(gap(a,b),6),kind='overlap' if intersect(a,b)>1e-7 else 'nearby_only',score=b.get('score')) for b in items if intersect(a,b)>1e-7 or gap(a,b)<=TOL+1e-8]
def union_duration(intervals):
 out=[]
 for a,b in sorted(intervals):
  if b<=a:continue
  if out and a<=out[-1][1]:out[-1][1]=max(out[-1][1],b)
  else:out.append([a,b])
 return sum(b-a for a,b in out)
def covered(a,items):
 return union_duration([(max(a['start'],b['start']),min(a['end'],b['end'])) for b in items])
def load_metadata(source):
 name=str(Path(source['member']).with_suffix('.json'))
 with tarfile.open(source['archive']) as tar:
  for m in tar:
   if m.name==name:
    raw=tar.extractfile(m).read();return json.loads(raw),dict(archive=source['archive'],member=name,sha256=hashlib.sha256(raw).hexdigest())
 raise FileNotFoundError(name)
def official(raw,role,start,duration):
 result=dict(words=[],sentences=[],vad=[],unaligned_words=[])
 def item(kind,i,obj,text=None,parent=None):
  a=obj.get('start');b=obj.get('end')
  if not isinstance(a,(float,int)) or not isinstance(b,(float,int)) or b<a:return None
  # Include +/-0.2s context for boundary-neighbor checks; visibly mark out-of-clip.
  if b<start-TOL or a>start+duration+TOL:return None
  return dict(id=f'{role}-{kind}-{i}',start=round(a-start,6),end=round(b-start,6),absolute_start=a,absolute_end=b,text=text,score=obj.get('score'),parent=parent,in_clip=min(b,start+duration)>max(a,start),source_record=obj)
 for i,s in enumerate(raw.get('metadata:transcript',[])):
  sentence=item('sentence',i,s,s.get('transcript',s.get('text','')))
  if sentence:result['sentences'].append(sentence)
  for j,w in enumerate(s.get('words',[])):
   word=item('word',f'{i}-{j}',w,w.get('word',w.get('text','')),f'{role}-sentence-{i}')
   if word:result['words'].append(word)
   elif sentence and (not isinstance(w.get('start'),(float,int)) or not isinstance(w.get('end'),(float,int))):result['unaligned_words'].append(w)
 for i,v in enumerate(raw.get('metadata:vad',[])):
  z=item('vad',i,v)
  if z:result['vad'].append(z)
 return result

def main():
 RUN.mkdir(parents=True,exist_ok=True);PREVIEW.mkdir(parents=True,exist_ok=True);(RUN/'raw_metadata').mkdir(exist_ok=True)
 cases=json.loads((SOURCE/'cases.json').read_text());reader=SourceReader(ROOT/'experiments/p2_features_full_crop384_20261025');results=[];rows=[]
 for c in cases:
  label=json.loads((SOURCE/'cases'/c['id']/'original_labels.json').read_text());full=reader.get(label['source']);start=full['start'];duration=c['duration'];off={};prov={}
  for role in ['listener','speaker']:
   raw,p=load_metadata(full['sources'][role]);prov[role]=p;off[role]=official(raw,role,start,duration)
   (RUN/'raw_metadata'/f'{c["id"]}_{role}.json').write_text(json.dumps(raw,ensure_ascii=False,indent=2))
  es=[];label_audio=[e for e in label['events'] if e.get('modality')=='audio'];excluded=[e for e in label['events'] if e.get('modality')!='audio']
  for e in label_audio:
   a=dict(e);a['comparison']={}
   for role in ['listener','speaker']:
    a['comparison'][role]={k:match(e,off[role][k]) for k in ['words','sentences','vad']}
    a['comparison'][role]['vad_coverage_s']=round(covered(e,off[role]['vad']),6)
    a['comparison'][role]['vad_coverage_fraction']=round(covered(e,off[role]['vad'])/max(e['end']-e['start'],1e-9),4)
   lw=a['comparison']['listener']['words'];lv=a['comparison']['listener']['vad']
   wo=any(x['kind']=='overlap' for x in lw);vo=any(x['kind']=='overlap' for x in lv)
   status='word_overlap' if wo else 'vad_only_overlap' if vo else 'no_listener_overlap'
   a['comparison']['status']=status;a['comparison']['near_listener_evidence']=any(x['kind']=='nearby_only' for x in lw+lv)
   a['comparison']['lexical_equal_word_ids']=[w['id'] for w in lw if e.get('text') and textnorm(e['text'])==textnorm(w['text'] or '')]
   a['comparison']['speaker_overlap_without_listener_overlap']=(not wo and not vo and any(x['kind']=='overlap' for x in a['comparison']['speaker']['vad']))
   a['comparison']['other_our_event_ids_nearby']=[z['event_id'] for z in label_audio if z is not e and gap(e,z)<=TOL+1e-8]
   a['comparison']['decision']='undetermined_difference_not_noise_label'
   es.append(a)
   rows.append(dict(sample_id=c['id'],human_group=c['user_feedback'],event_id=e['event_id'],start=e['start'],end=e['end'],description=e['description'],status=status,listener_words=json.dumps(lw,ensure_ascii=False),listener_vad_coverage=a['comparison']['listener']['vad_coverage_fraction'],speaker_words=json.dumps(a['comparison']['speaker']['words'],ensure_ascii=False),speaker_vad_coverage=a['comparison']['speaker']['vad_coverage_fraction']))
  missed=[]
  for w in off['listener']['words']:
   if not w['in_clip']:continue
   m=match(w,es);w['our_event_matches']=m
   if not any(x['kind']=='overlap' for x in m):missed.append(w['id'])
  summary=dict(ours_audio_events=len(es),ours_visual_events_excluded=len(excluded),official_listener_words=sum(w['in_clip'] for w in off['listener']['words']),official_listener_sentences=sum(w['in_clip'] for w in off['listener']['sentences']),official_listener_vad_segments=sum(w['in_clip'] for w in off['listener']['vad']),word_overlap=sum(e['comparison']['status']=='word_overlap' for e in es),vad_only_overlap=sum(e['comparison']['status']=='vad_only_overlap' for e in es),no_listener_overlap=sum(e['comparison']['status']=='no_listener_overlap' for e in es),official_word_ids_without_our_overlap=missed)
  result=dict(sample_id=c['id'],short_id=c['id'].split('_')[2],human_group=c['user_feedback'],clip_absolute_start=start,clip_absolute_end=start+duration,duration_s=duration,official=off,ours_audio_events=es,ours_other_events=excluded,summary=summary,provenance=prov,media=dict(video=f'../audio_cleanup_6/media/{c["id"]}/video.mp4',poster=f'../audio_cleanup_6/media/{c["id"]}/poster.png',listener=f'../audio_cleanup_6/media/{c["id"]}/original.wav',speaker=f'../audio_cleanup_6/media/{c["id"]}/speaker.wav'))
  results.append(result);print(c['id'].split('_')[2],json.dumps(summary),flush=True)
 reader.close()
 out=dict(schema='listener-official-comparison-six-v1',tolerance_s=TOL,rules=dict(overlap='positive interval intersection; no tolerance inflation',nearby='interval gap <= .2s; not counted as actual overlap',content='original descriptions preserved; lexical equal only ignores case/punctuation, no translation or semantic rewrite',scope='ours audio events only; visual events excluded from transcript/VAD difference counts',interpretation='all unmatched items are differences, not confirmed noise'),cases=results)
 (RUN/'comparison.json').write_text(json.dumps(out,ensure_ascii=False,indent=2));(PREVIEW/'review.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
 with (RUN/'events.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 # Build the review page with the same text and timestamps as the JSON output.
 import shutil
 from fontTools.ttLib import TTFont
 from fontTools import subset
 html=(ROOT/'scripts/listener_official_compare.html').read_text()
 (PREVIEW/'index.html').write_text(html);shutil.copy2(RUN/'events.csv',PREVIEW/'events.csv')
 font=TTFont('/tmp/v8_browser_libs/extracted/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',fontNumber=2)
 sub=subset.Subsetter();sub.populate(text=html+json.dumps(out,ensure_ascii=False));sub.subset(font);font.flavor='woff';font.save(PREVIEW/'review.woff')
 print('Events',len(rows),'cases',len(results),flush=True)
if __name__=='__main__':main()
