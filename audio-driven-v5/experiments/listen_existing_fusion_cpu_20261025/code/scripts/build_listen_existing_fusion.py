"""Build the 50-clip CPU-only automatic integration review. No feedback input."""
import argparse,hashlib,io,json,shutil,sys
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from listen_benchmark.existing_result_fusion import fuse,VERSION
from listen_benchmark.response_review import alignment_track

def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def build(source,align,out,preview,font=None):
 for p in [out,preview]:
  if p.exists() and any(p.iterdir()):raise ValueError('Use empty output: '+str(p))
 raw=(source/'observations.jsonl').read_bytes();orig={s['sample_id']:s for s in map(json.loads,raw.splitlines())};old=json.loads((source/'review.json').read_text());samples=[];hashes={}
 for s in old['samples']:
  tracks=[]
  for p in sorted(align.glob(s['id']+'*.json')):
   b=p.read_bytes();hashes[str(p)]=hashlib.sha256(b).hexdigest();tracks.append(alignment_track(json.loads(b),s['duration']))
  r=fuse(orig[s['id']],s['decisions'],tracks);r.update(id=s['id'],video=s['video'],audio=s['audio'],original_events=orig[s['id']]['events'],source_decisions=s['decisions']);samples.append(r)
 events=[e for s in samples for e in s['events']];tokens=sum(len(t['tokens']) for s in samples for t in s['alignment_tracks']);clusters=sum(len(s['token_clusters']) for s in samples)
 summary=dict(samples=len(samples),original_observations=sum(len(s['events']) for s in orig.values()),mapped_observations_before=sum(e['mapping_status']=='mapped_candidate' for s in orig.values() for e in s['events']),independent_event_candidates=len(events),visual_candidates=sum(e['modality']=='visual' for e in events),audio_candidates=sum(e['modality']=='audio' for e in events),events_with_unknown_time=sum(e.get('start') is None for e in events),samples_without_independent_candidates=sum(not s['events'] for s in samples),asr_token_observations=tokens,asr_token_clusters=clusters,asr_cross_window_duplicate_observations=tokens-clusters,unlocalized_units=sum(len(s['automatic_fusion']['alignment_failures']) for s in samples),human_feedback_used_for_generation=False,media_inference_rerun=False,gpu_used=False,full_corpus_applied=False,training_admitted=False)
 policy=dict(generation_inputs=['existing_semantic_observations','existing_ASR_transcripts_and_alignments'],human_feedback_read=False,sample_specific_generation_branches=False,gpu_used=False,media_read=False,new_boundaries_invented=False,method='select existing positive intervals with sequence and source consistency; retain conflicts separately',precision_guarantee=False)
 data=dict(schema=VERSION,run_id=out.name,summary=summary,samples=samples,policy=policy)
 html=(ROOT/'scripts/listen_existing_fusion.html').read_text().replace('__DATA__',json.dumps(data,ensure_ascii=False).replace('</','<\\/'))
 fontbytes=None
 if font:
  from fontTools.ttLib import TTFont
  from fontTools.subset import Options,Subsetter
  f=TTFont(str(font),fontNumber=2);o=Options();o.flavor='woff';sub=Subsetter(options=o);sub.populate(text=html);sub.subset(f);f.flavor='woff';buf=io.BytesIO();f.save(buf);fontbytes=buf.getvalue()
 else:html=html.replace('@font-face{font-family:Review;src:url(review.woff)}','')
 for p in [out,preview]:
  p.mkdir(parents=True,exist_ok=True);dump(p/'review.json',data);dump(p/'summary.json',summary);dump(p/'policy.json',policy);(p/'observations.jsonl').write_bytes(raw);(p/'index.html').write_text(html)
  (p/'labels.candidate.jsonl').write_text(''.join(json.dumps({k:v for k,v in s.items() if k not in ['original_events','source_decisions','alignment_tracks','pre_correction_events','video','audio']},ensure_ascii=False)+'\n' for s in samples))
  if fontbytes:(p/'review.woff').write_bytes(fontbytes)
 files=['src/listen_benchmark/existing_result_fusion.py','src/listen_benchmark/integration_pilot.py','src/listen_benchmark/response_review.py','scripts/build_listen_existing_fusion.py','scripts/listen_existing_fusion.html','tests/test_listen_existing_fusion.py']
 for name in files:
  dest=out/'code'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dest)
 dump(out/'manifest.json',dict(created_at=datetime.now(timezone.utc).isoformat(),source=str(source),alignment_source=str(align),raw_observations_sha256=hashlib.sha256(raw).hexdigest(),alignment_sha256=hashes,code_sha256={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files},human_feedback_input=None,gpu_used=False))
 print(json.dumps(summary,ensure_ascii=False))
if __name__=='__main__':
 p=argparse.ArgumentParser()
 for k in ['source','align','out','preview']:p.add_argument('--'+k,type=Path,required=True)
 p.add_argument('--font',type=Path);a=p.parse_args();build(a.source.resolve(),a.align.resolve(),a.out.resolve(),a.preview.resolve(),a.font)
