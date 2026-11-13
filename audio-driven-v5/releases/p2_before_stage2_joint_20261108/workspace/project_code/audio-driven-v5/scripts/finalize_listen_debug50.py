"""Create traceable positive-only weak-label fixtures; never infer unseen negatives."""
import argparse,collections,copy,hashlib,json,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from listen_benchmark.seven_class import TAXONOMY

def dump(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def rows(p,x):p.write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in x))
def union(values):
 result=[]
 for a,b in sorted(values):
  if result and a<=result[-1][1]:result[-1][1]=max(b,result[-1][1])
  else:result.append([a,b])
 return result
def complement(values,d):
 out=[];start=0.
 for a,b in values:
  if start<a:out.append([start,a])
  start=b
 if start<d:out.append([start,d])
 return out

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();run=a.run.resolve()
 data=json.loads((run/'integrated/review.json').read_text());targets=json.loads((run/'targets.candidate.json').read_text());byid={r['sample_id']:r for r in targets};selection=json.loads((run/'selection.json').read_text())
 assert len(data['samples'])==len(targets)==len(selection['samples'])==50
 version='open-to-visual6-audio1-semantic-v3+verbatim-v2';admission='debug-positive-only-weak-v1'
 policy=dict(schema=admission,requested_use='training code debug only',authorization='User requested 50 newly annotated samples and seven-class weak labels for training debug',formal_training_admitted=False,human_ground_truth=False,known_support='union of selected weak-positive intervals only',unlabeled_time_policy='unknown',weak_negatives_inferred=False,balanced_training_ready=False,classification='Qwen3-VL text-only semantic categories',mapping_version=version,taxonomy=list(TAXONOMY),warning='Positive-only fixtures cannot establish negative support. Production class_statistics may disable all classes; use the explicit BCE smoke check for plumbing, not model quality.')
 dump(run/'DEBUG_ADMISSION.json',policy)
 candidate=[];debug=[];mapping=[];counts=collections.Counter()
 for s in data['samples']:
  sid=s['id'];d=s['duration_s'];events=[]
  for e in s['events']:
   for k in e['categories']:
    assert k in TAXONOMY and ((e['modality']=='audio')==(k==TAXONOMY[-1]))
    event=dict(event_id=e['event_id']+'-'+k,type=k,start=e['start'],end=e['end'],description=e['description'],source=dict(candidate_event_id=e['event_id'],candidate_file='integrated/labels.candidate.jsonl',source_observation_ids=e['source_observation_ids'],description_source=e.get('description_source'),timing_evidence=e.get('timing_evidence'),review_flags=e['review_flags']),verification='weak_accepted',boundary_status='weak_accepted',boundary_censored=False)
    events.append(event);mapping.append(dict(sample_id=sid,candidate_event_id=e['event_id'],training_event_id=event['event_id'],type=k));counts[k]+=1
  coverage={k:union([[e['start'],e['end']] for e in events if e['type']==k]) for k in TAXONOMY}
  label=dict(schema='listen-events-v2-visual6-audio1',sample_id=sid,time_origin='clip_start_seconds',duration_s=d,taxonomy=list(TAXONOMY),events=events,coverage=coverage,unknown={k:complement(coverage[k],d) for k in TAXONOMY},training_admitted=False,provenance=dict(mapping_version=version,admission_policy=admission,human_verified=False,formal_training_admitted=False,debug_only=True,media_hashes=byid[sid]['media_hashes'],coverage_sources={k:dict(reference='integrated/labels.candidate.jsonl#'+sid,verification='weak_accepted',scope='positive intervals only') if coverage[k] else {} for k in TAXONOMY},admission_record=dict(reference='DEBUG_ADMISSION.json',scope='debug-positive-only',no_negative_inference=True)))
  candidate.append(label);label=copy.deepcopy(label);label['training_admitted']=True;debug.append(label)
 rows(run/'labels.canonical.candidate.jsonl',candidate);rows(run/'labels.debug-positive.jsonl',debug);dump(run/'event_mapping.json',mapping)
 targets_debug=[dict(t,admitted=True) for t in targets];dump(run/'targets.debug.json',targets_debug)
 shutil.copy2(run/'integrated/labels.candidate.jsonl',run/'labels.integrated.jsonl')
 summary=dict(samples=50,raw_observations=data['summary']['original_observations'],integrated_events=data['summary']['independent_event_candidates'],per_class={k:counts[k] for k in TAXONOMY},unlocalized_asr_units=data['summary']['unlocalized_units'],without_candidates=data['summary']['samples_without_independent_candidates'],all_required_raw_views_successful=True,source_scope='native listener crop >=384; primary/onset reserve train; 2-8.04 seconds; distinct source pairs; excludes old50',debug_only=True,formal_training_admitted=False,negative_labels_inferred=False,classification_gpu_used=True,media_inference_rerun=False,fusion_gpu_used=False)
 dump(run/'summary.json',summary)
 files=['scripts/prepare_listen_fullraw_debug50.py','scripts/export_listen_debug50_media.py','scripts/finalize_listen_debug50.py','scripts/build_listen_existing_fusion.py','scripts/prepare_listen_semantic_mapping.py','scripts/run_listen_semantic_mapping.py','scripts/build_listen_semantic_seven_class.py','src/listen_benchmark/existing_result_fusion.py','src/listen_benchmark/integration_pilot.py','src/listen_benchmark/response_review.py','src/listen_benchmark/open_merge.py','src/listen_benchmark/semantic_mapping.py','src/listen_benchmark/seven_class.py']
 for f in files:
  dest=run/'code'/f;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/f,dest)
 dump(run/'code_manifest.json',{f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files})
 preview=ROOT/'data_previews/listen_benchmark_review/debug50_fullraw';media_link=preview.parent/'debug50_fullraw_media'
 if not media_link.exists():media_link.symlink_to(run/'media',target_is_directory=True)
 # Old five-regression filter is irrelevant to this fresh sample; remove it from this copy.
 for html in [preview/'index.html',run/'integrated/index.html']:
  text=html.read_text();import re
  text=re.sub(r'<option value="human">.*?</option>','',text)
  html.write_text(text)
 print(json.dumps(summary,ensure_ascii=False))
if __name__=='__main__':main()
