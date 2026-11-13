import json,random,hashlib
from pathlib import Path
r=Path('/private/liqi_workspace/project_code/audio-driven-v5/experiments/listen_official_random100_20261026');d=json.loads((r/'comparison.json').read_text());out=r.parents[1]/'data_previews/listen_benchmark_review/audio_official_random100'
assert len(d['cases'])==len({c['sample_id'] for c in d['cases']})==100
import csv
rows=list(csv.DictReader((r.parent/'p2_train50h_backchannel_face_20261026/samples.csv').open()));eligible=sorted({x['sample_id'] for x in rows}-set(d['sampling']['excluded_previous_review_ids']))
assert random.Random(d['sampling']['seed']).sample(eligible,100)==[c['sample_id'] for c in d['cases']]
num_events=0;num_words_out=0;empty_vad=0
for c in d['cases']:
 label=json.loads((out/'media'/c['sample_id']/'label.json').read_text());source={e['event_id']:e for e in label['events']}
 for role in ['listener','speaker']:
  raw=json.loads((r/'raw_metadata'/f'{c["sample_id"]}_{role}.json').read_text());assert 'metadata:vad' in raw and 'metadata:transcript' in raw
  assert raw['id']==Path(c['provenance'][role]['member']).stem
  for kind in ['words','sentences','vad']:
   for x in c['official'][role][kind]:
    assert abs(x['start']-(x['absolute_start']-c['clip_absolute_start']))<=1e-6
    assert abs(x['end']-(x['absolute_end']-c['clip_absolute_start']))<=1e-6
 for e in c['ours_audio_events']:
  assert {k:v for k,v in e.items() if k!='comparison'}==source[e['event_id']]
  assert e['modality']=='audio';num_events+=1
  for role in ['listener','speaker']:
   for kind in ['words','sentences','vad']:
    for m in e['comparison'][role][kind]:
     overlap=max(0,min(e['end'],m['end'])-max(e['start'],m['start']));gap=max(0,max(e['start'],m['start'])-min(e['end'],m['end']))
     assert abs(overlap-m['overlap_s'])<=1e-6 and abs(gap-m['gap_s'])<=1e-6
     assert (overlap>1e-7)==(m['kind']=='overlap')
 inside=c['vad_inside_intervals'];outside=c['vad_outside_intervals'];intervals=sorted(inside+outside,key=lambda x:x['start']);end=0
 for x in intervals:assert abs(x['start']-end)<1e-6;end=x['end'];assert x['end']>x['start']
 assert abs(end-c['duration_s'])<1e-6
 s=c['summary'];assert sum(s[k] for k in ['word_overlap','vad_only_overlap','no_listener_overlap'])==len(c['ours_audio_events'])
 empty_vad+=not bool(s['official_listener_vad_segments']);num_words_out+=s['official_words_without_vad_overlap']
assert num_events==158 and num_words_out==15 and empty_vad==59
(r/'numeric_check.json').write_text(json.dumps(dict(samples=100,random_draw_reproduced=True,metadata_present_on_both_sides=True,participant_ids_match=True,all_time_offsets_checked=True,event_text_and_times_unchanged=True,all_overlaps_recomputed=True,vad_inside_and_outside_partition_entire_clips=True,events=num_events,official_words_outside_vad=num_words_out),indent=2))
print('PASS: random draw, 200 official metadata files, source IDs, original events, timing, VAD partitions')
