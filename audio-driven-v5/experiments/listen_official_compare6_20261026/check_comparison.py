import json
from pathlib import Path
r=Path('/private/liqi_workspace/project_code/audio-driven-v5/experiments/listen_official_compare6_20261026');d=json.loads((r/'comparison.json').read_text());n=0
for c in d['cases']:
 original=json.loads((r.parent/'listen_audio_cleanup_6_20261026/cases'/c['sample_id']/'original_labels.json').read_text())
 source_events={x['event_id']:x for x in original['events']}
 for event in c['ours_audio_events']:
  assert {k:v for k,v in event.items() if k!='comparison'}==source_events[event['event_id']]
 for role in ['listener','speaker']:
  for key in ['words','sentences','vad']:
   for x in c['official'][role][key]:
    assert abs(x['start']-(x['absolute_start']-c['clip_absolute_start']))<1e-6
    assert abs(x['end']-(x['absolute_end']-c['clip_absolute_start']))<1e-6
 for e in c['ours_audio_events']:
  n+=1;assert e['modality']=='audio'
  for role in ['listener','speaker']:
   for key in ['words','sentences','vad']:
    for x in e['comparison'][role][key]:
     overlap=max(0,min(e['end'],x['end'])-max(e['start'],x['start']));gap=max(0,max(e['start'],x['start'])-min(e['end'],x['end']))
     assert abs(overlap-x['overlap_s'])<1e-6
     assert abs(gap-x['gap_s'])<1e-6
     assert (x['kind']=='overlap')==(overlap>1e-7)
     assert gap<=.200001
 csum=c['summary'];assert csum['word_overlap']+csum['vad_only_overlap']+csum['no_listener_overlap']==len(c['ours_audio_events'])
assert n==10
lookup={c['short_id']:c for c in d['cases']}
assert lookup['S0866']['summary']['word_overlap']==2
assert all(e['comparison']['listener']['vad_coverage_s']==0 for e in lookup['S0866']['ours_audio_events'])
assert lookup['S1921']['summary']['no_listener_overlap']==1
assert lookup['S1252']['summary']['official_listener_sentences']==0 # nearby outside-clip sentence excluded
(r/'numeric_check.json').write_text(json.dumps(dict(cases=6,audio_events=10,timestamp_conversion_pass=True,all_overlap_and_gap_recomputed=True,visual_excluded=True,outside_clip_context_excluded_from_counts=True,source_text_preserved=True),indent=2))
print('PASS: exact timestamps, overlap, near-only, clip bounds, visual exclusion')
