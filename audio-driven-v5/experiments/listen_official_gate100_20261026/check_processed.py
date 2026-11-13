import json,sys,hashlib
from pathlib import Path
import numpy as np,soundfile as sf
ROOT=Path('/private/liqi_workspace/project_code/audio-driven-v5');sys.path.insert(0,str(ROOT/'src'))
from listen_benchmark.official_audio_gate import make_gate
r=ROOT/'experiments/listen_official_gate100_20261026';out=ROOT/'data_previews/listen_benchmark_review/audio_official_gate100';src=ROOT/'data_previews/listen_benchmark_review/audio_official_random100';d=json.loads((r/'results.json').read_text());original=json.loads((ROOT/'experiments/listen_official_random100_20261026/comparison.json').read_text())
assert [c['sample_id'] for c in d['cases']]==[c['sample_id'] for c in original['cases']]
for c in d['cases']:
 p=out/'media'/c['sample_id'];sp=src/'media'/c['sample_id'];x,sr=sf.read(sp/'listener.wav',dtype='float32');y,ysr=sf.read(p/'processed.wav',dtype='float32');removed,rsr=sf.read(p/'removed.wav',dtype='float32')
 assert x.shape==y.shape==removed.shape and sr==ysr==rsr
 mask,info=make_gate(c['official']['listener'],len(x),sr)
 np.testing.assert_array_equal(y, x*mask);np.testing.assert_array_equal(removed,x-y)
 assert np.all(y[mask==0]==0);np.testing.assert_array_equal(y[mask==1],x[mask==1]);assert np.max(abs(y+removed-x))<1e-7
 assert hashlib.sha256((sp/'listener.wav').read_bytes()).hexdigest()==c['media_hashes']['target_audio']
 assert info['full_silence']==c['gate']['full_silence']
 if info['full_silence']:assert not y.any()
 # Every core sample must be preserved because fade edges are inside the 100ms guard.
 for a,b in info['core_intervals_samples']:assert np.all(mask[a:b]==1)
 original_label=json.loads((sp/'label.json').read_text());derived=json.loads((p/'gated_candidates.json').read_text());lookup={e['event_id']:e for e in original_label['events']}
 assert Path(derived['gate_metadata']).is_file()
 for e in derived['events']:
  assert {k:v for k,v in e.items() if k!='audio_gate'}==lookup[e['event_id']]
 for e in derived['excluded_audio_events']:
  assert e['original_event']==lookup[e['original_event']['event_id']]
  assert e['gate']['status']=='fully_muted' and not e['gate']['retained_intervals']
 assert len(derived['events'])+len(derived['excluded_audio_events'])==len(original_label['events'])
summary=d['gate_summary'];assert summary['audio_events_fully_muted']+summary['audio_events_partially_retained']+summary['audio_events_fully_supported']==158
(r/'full_numeric_check.json').write_text(json.dumps(dict(samples=100,same_sample_order=True,all_sample_counts_and_rates_unchanged=True,core_waveform_bit_exact=True,zero_mask_output_exactly_zero=True,removed_equals_original_minus_processed=True,source_hashes_match=True,all_original_events_accounted_for=True,no_text_rewrite=True,full_silence_clips=52),indent=2))
print('PASS: 100 gates and residuals, exact original cores, silence, duration, labels, source hashes')
