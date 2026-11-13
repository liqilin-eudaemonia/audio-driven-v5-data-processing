import json,sys,hashlib
from pathlib import Path
root=Path('/private/liqi_workspace/project_code/audio-driven-v5');run=root/'experiments/listen_official_aligned100_20261026';data=json.loads((run/'results.json').read_text());old=json.loads((root/'experiments/listen_official_gate100_20261026/results.json').read_text());assert [c['sample_id'] for c in data['cases']]==[c['sample_id'] for c in old['cases']]
methods={};kept=0
for c in data['cases']:
 src=root/'data_previews/listen_benchmark_review/audio_official_random100/media'/c['sample_id']
 original=json.loads((src/'label.json').read_text());lookup={e['event_id']:e for e in original['events']}
 assert hashlib.sha256((src/'label.json').read_bytes()).hexdigest()==c['label_sha256']
 wav=root/'data_previews/listen_benchmark_review/audio_official_gate100/media'/c['sample_id']/'processed.wav'
 assert hashlib.sha256(wav.read_bytes()).hexdigest()==c['gate']['processed_audio_sha256']
 result=json.loads((run/'labels'/f'{c["sample_id"]}.json').read_text());audio=[e for e in result['events'] if e['modality']=='audio']
 assert len(audio)==len(c['aligned_audio_events']);assert c['label_group']==('has_audio_labels' if audio else 'no_audio_labels')
 assert [e for e in result['events'] if e['modality']!='audio']==[e for e in original['events'] if e['modality']!='audio']
 assert {e['event_id'] for e in result['events']}|{e['original_event']['event_id'] for e in result['excluded_audio_events']}==set(lookup)
 for a in result['audio_event_alignment']:
  methods[a['method']]=methods.get(a['method'],0)+1
  assert a['description']==lookup[a['event_id']]['description'] and a['text']==lookup[a['event_id']].get('text')
 for e in audio:
  kept+=1;source=lookup[e['event_id']]
  assert e['description']==source['description'] and e.get('text')==source.get('text') and e['categories']==source['categories']
  assert e['source_interval']==dict(start=source['start'],end=source['end'])
  for x in e['intervals']:
   assert any(a/c['gate']['sample_rate']-1e-9<=x['start'] and x['end']<=b/c['gate']['sample_rate']+1e-9 for a,b in c['gate']['positive_intervals_samples'])
  a=e['boundary_alignment']
  if a['method']=='official_words':
   ws=a['official_evidence'];assert e['start']==ws[0]['start'] and e['end']==ws[-1]['end']
  if a['method']=='official_vad':
   v=a['official_evidence'][0];assert e['start']==v['start'] and e['end']==v['end']
assert kept==66 and sum(methods.values())==158
r=dict(samples=100,same_fixed_sample_order=True,all_source_label_and_processed_waveform_hashes_unchanged=True,all_events_accounted_for=True,all_original_text_and_categories_preserved=True,visual_events_exactly_unchanged=True,final_ranges_inside_processed_audio_support=True,adopted_official_boundaries_exact=True,kept_audio_events=kept,methods=methods)
(run/'full_numeric_check.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
