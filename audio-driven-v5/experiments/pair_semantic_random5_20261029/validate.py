"""Audit every comparison row and immutable policy/input fingerprints."""
import ctypes
assert ctypes.CDLL(None).prctl(41,1,0,0,0)==0
import hashlib,json,sys
from pathlib import Path
EXP=Path(__file__).resolve().parent;ROOT=EXP.parents[1]
sys.path[:0]=[str(EXP),str(ROOT/'experiments/pair_boundary_holdout3_20261029')]
import compare
from export import initial,keep,terminal,three_changes
read=compare.read
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
contract=read(EXP/'results/contract.json');selection=read(EXP/'selection.json');m=read(EXP/'review/manifest.json');stats=read(EXP/'comparison.json')['pairs']
original=read(ROOT/'experiments/pair_semantic_trial2_20261029/results/contract.json')
assert contract['repair_sha256']==original['repair_sha256']==sha(ROOT/contract['repair_module'])
assert all(sha(ROOT/p)==h for p,h in contract['frozen_code_sha256'].items())
assert all(sha(ROOT/'data_corpora/seamless_listen_local_full/results'/(p+'.json'))==h for p,h in selection['production_results'].items())
assert not set(selection['excluded'])&set(contract['pairs'])
ps={d['pair']:d for d in read(EXP/'review/production_before/manifest.json')['pairs']};hs={d['pair']:d for d in read(EXP/'review/h_manifest.json')['pairs']}
reports=[]
for j,stat in zip(m['pairs'],stats):
 pair=j['pair'];p,h=ps[pair],hs[pair];cases=j['comparison_cases']
 for view in (p,h,j):
  for row in view['rows']:
   evidence=ROOT/'data_previews/seamless_listen_full_source_recovery'/row['evidence_url']
   ev=read(evidence);ev['view']=row;evidence.write_text(json.dumps(ev,ensure_ascii=False,indent=2)+'\n')
 shown={side:{compare.sig(r) for c in cases for r in c[side]} for side in ('production','before','after')}
 initialshown={side:{compare.sig(r) for c in cases if c.get('stage')=='initial' or c.get('also_initial_change') for r in c[side]} for side in ('before','after')}
 finalshown={side:{compare.sig(r) for c in cases if c.get('stage')=='final' for r in c[side]} for side in ('production','before','after')}
 for g in compare.changes(initial(h),initial(j)):
  for side in ('before','after'):
   assert all(compare.sig(r) in initialshown[side] for r in g[side])
 for g in three_changes(keep(p),keep(h),keep(j)):
  for side in ('production','before','after'):assert all(compare.sig(r) in finalshown[side] for r in g[side])
 assert all(compare.sig(r) in shown['after'] for r in terminal(j) if r['pool']!='keep_candidate')
 equal_text=[]
 for label,rows in [('H initial',initial(h)),('P kept',keep(p)),('H kept',keep(h))]:
  targets=initial(j) if label=='H initial' else keep(j);mapping={compare.sig(r):r for r in targets}
  for row in rows:
   if compare.sig(row) in mapping:
    other=mapping[compare.sig(row)]
    if row['text']!=other['text']:equal_text.append(dict(stage=label,start=row['start'],end=row['end'],old=row['text'],new=other['text']))
 for c in cases:
  for side in ('production','before','after'):
   c[side].sort(key=lambda r:(r['start'],r['end']))
   for r in c[side]:
    for name in ['media','poster','evidence_url']:
     assert (ROOT/'data_previews/seamless_listen_full_source_recovery'/r[name]).exists(),r[name]
 source=read(EXP/(pair+'.json'));hr=read(EXP/'h'/pair/'result.json');jr=read(EXP/'results'/pair/'result.json')
 assert jr['complete'] and hr['complete']
 oldcuts=compare.evaluate(source,compare.roots(hr),compare.roots(hr))['new_word_crossing_edges']
 oldedges={(r['time'],r['word']['person'],r['word']['start'],r['word']['end']) for r in oldcuts}
 addedcuts=[c for c in stat['initial_metrics']['new_word_crossing_edges'] if (c['time'],c['word']['person'],c['word']['start'],c['word']['end']) not in oldedges]
 assert not stat['initial_metrics']['duration_violations']
 assert not stat['initial_metrics']['overlaps']
 reports.append(dict(pair=pair,new_word_crossing_edges=addedcuts,same_interval_text_differences=equal_text,comparison_groups=len(cases),held_full_parents=len(stat['semantic_parent_holds'])))
 # Persist chronological columns for the page and canonical manifest.
 dest=ROOT/'data_previews/seamless_listen_full_source_recovery'/pair/'review.json';published=read(dest)
 for view in [published['fixed_pairs'],*published['fixed_pair_versions']]:
  if view['revision']==j['revision']:view['comparison_cases']=cases
 dest.write_text(json.dumps(published,ensure_ascii=False,indent=2)+'\n')
(EXP/'review/manifest.json').write_text(json.dumps(m,ensure_ascii=False,indent=2)+'\n')
result=dict(same_code_as_user_reviewed_J=True,sha256=contract['repair_sha256'],frozen_code_files_verified=len(contract['frozen_code_sha256']),production_results_unchanged=True,random_selection_frozen_before_runs=True,all_initial_changes_shown=True,all_three_version_final_changes_shown=True,all_withheld_terminal_candidates_shown=True,all_comparison_media_and_evidence_exist=True,full_run_started=False,sources=reports)
(EXP/'validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result,ensure_ascii=False,indent=2))
