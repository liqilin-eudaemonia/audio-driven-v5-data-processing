import json,sys
from pathlib import Path
EXP=Path(__file__).resolve().parent;ROOT=EXP.parents[1]
sys.path.insert(0,str(ROOT/'src'))
from p3_listen.common import sha
from compare import read,write
selection=read(EXP/'selection.json');comparison=read(EXP/'comparison.json');prod=ROOT/'data_corpora/seamless_listen_local_full'
assert all(sha(prod/'results'/f'{p}.json')==h for p,h in selection['production_results'].items())
contract=read(EXP/'localbc/contract.json');previous=read(ROOT/'experiments/pair_boundary_real_audit_20261029/localbc_pairs/contract.json')
assert contract['repair_sha256']==previous['repair_sha256']
assert all(sha(ROOT/p)==h for p,h in contract['frozen_code_sha256'].items())
for i,r in enumerate(comparison['pairs']):
 for phase in ['initial','terminal','kept']:
  c=r[phase];assert not c['new_word_crossing_edges'] and not c['overlaps'] and not c['duration_violations']
  if i<2:assert c['old_count']==c['new_count']==c['unchanged'] and not c['previous_speaker_words_lost'] and not c['added_speaker_words']
  else:
   assert [w['text'] for w in c['previous_speaker_words_lost']]==['Okay.']
   assert [w['text'] for w in c['added_speaker_words']]==["it's"]
counts={};qc={}
for variant in ['gap5','localbc']:
 m=read(EXP/'review'/variant/'manifest.json');counts[variant]=sum(len(d['rows']) for d in m['pairs'])
 for d in m['pairs']:
  for r in d['rows']:
   assert r['media_check']['decoded'] and r['media_check']['audio_channels']==2
   assert abs(r['media_check']['actual_seconds']-r['duration'])<.09
   assert r['added_context_seconds']==0
 qc[variant]={k:sum(d['counts'].get(k,0) for d in m['pairs']) for k in ['fresh','reused']}
report=dict(rule_module_identical_to_prior_two_source_trial=True,production_and_frozen_code_unchanged=True,decoded_media_records=counts,qc_counts=qc,all_new_boundaries_outside_valid_aligned_words=True,duration_and_nonoverlap_checks=True,first_two_sources_identical=True,third_source_known_regression='Okay. 213.343–213.963 seconds omitted in both initial and quality-passed result',third_source_added_word="it's 359.819–359.959 seconds; listener VAD overlap remains flagged",no_claim_of_full_semantic_or_acoustic_safety=True)
write(EXP/'data_validation.json',report);print(json.dumps(report,ensure_ascii=False))
