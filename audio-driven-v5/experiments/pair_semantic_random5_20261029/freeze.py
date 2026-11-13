"""Freeze the validated segmentation policy; this does not admit QC-held pairs."""
import hashlib,json,shutil
from pathlib import Path
EXP=Path(__file__).resolve().parent;ROOT=EXP.parents[1];DEST=ROOT/'releases/listen_pair_segmentation_j_v1_20261029'
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
contract=read(EXP/'results/contract.json');valid=read(EXP/'validation.json');browser=read(EXP/'browser_validation.json')
assert valid['same_code_as_user_reviewed_J'] and valid['all_three_version_final_changes_shown'] and browser['no_page_errors']
assert all(not s['new_word_crossing_edges'] and not s['same_interval_text_differences'] for s in valid['sources'])
initial=read(EXP/'initial_inspection.json');coverage=read(EXP/'production_candidate_coverage.json')
assert sum(r['metrics']['old_count'] for r in initial)==150
assert sum(r['metrics']['unchanged'] for r in initial)==140
for r in initial:
 for c in r['cases']:
  assert len(c['before'])==len(c['after'])==1
  a,b=c['before'][0],c['after'][0]
  assert a['start']==b['start'] and b['end']<a['end'] and b['semantic_end']=='sentence_terminal' and b.get('sentence_end_protection')
assert [(r['pair'],[w['text'] for w in r['metrics']['previous_speaker_words_lost']]) for r in coverage if r['metrics']['previous_speaker_words_lost']]==[('naturalistic_V03_S1634_I00000407',['He','I','I'])]
paths=list(contract['frozen_code_sha256'])+['scripts/'+n for n in ['listen_local_recovery_v2.py','listen_local_recovery_gap5.py','listen_local_recovery_localbc.py','listen_local_recovery_head_attach.py','listen_local_recovery_semantic_trial.py']]
paths+=['experiments/pair_boundary_real_audit_20261029/rerun_fixed_pairs.py','experiments/pair_semantic_random5_20261029/run.py','tests/test_listen_semantic_trial.py']
DEST.mkdir(parents=True,exist_ok=True)
for rel in paths:
 src=ROOT/rel;dst=DEST/'code'/rel;dst.parent.mkdir(parents=True,exist_ok=True)
 if dst.exists():assert sha(dst)==sha(src)
 else:shutil.copyfile(src,dst)
manifest=dict(version='listen-pair-segmentation-j-v1',status='fixed_segmentation_policy',date='2026-10-29',module=contract['repair_module'],revision=contract['revision'],module_sha256=contract['repair_sha256'],code_sha256={p:sha(ROOT/p) for p in paths},parameters=dict(hard_pause_seconds=5,min_pair_seconds=2,max_pair_seconds=11.95,initial_segmentation_before_audio_screening=True,audio_screening_rules_changed=False,uncertain_recovery='preserve_full_parent_with_original_QC_verdict'),validation=dict(prior_J_rule_and_presented_cases_accepted_by_user=True,random_seed=20261031,random_pairs=contract['pairs'],initial_h=150,initial_j=150,initial_exact_matches=140,initial_changed_groups=10,change_type='only preserving an existing sentence terminal instead of appending next-sentence fragments',old_full_kept=131,j_kept=113,old_full_exact_matches=92,old_full_words_missing_from_j_candidates=['I','I','He'],semantic_parent_holds=18,regression_tests_passed=5,comparison_playbacks=len(browser['played'])),scope='Fixed source segmentation and conservative recovery behavior; QC-held original candidates stay withheld, not silently admitted. No full-corpus processing or training replacement.',evidence={str(p.relative_to(ROOT)):sha(p) for p in [EXP/'selection.json',EXP/'comparison.json',EXP/'validation.json',EXP/'browser_validation.json',EXP/'production_candidate_coverage.json']},production_modified=False,full_run_started=False)
(DEST/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
(DEST/'verify.py').write_text('''import hashlib,json\nfrom pathlib import Path\np=Path(__file__).resolve().parent;r=p.parents[1];m=json.loads((p/'manifest.json').read_text())\nfor rel,h in m['code_sha256'].items():\n for f in [r/rel,p/'code'/rel]:\n  assert hashlib.sha256(f.read_bytes()).hexdigest()==h,str(f)\nprint(m['version'], 'verified', len(m['code_sha256']), 'source and archive files')\n''')
print('FROZEN',DEST,len(paths),'files')
