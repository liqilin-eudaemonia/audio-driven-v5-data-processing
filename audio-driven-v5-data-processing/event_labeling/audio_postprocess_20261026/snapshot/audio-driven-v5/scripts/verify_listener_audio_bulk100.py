import sys,json
from pathlib import Path
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
from listen_audio_postprocess.bulk import derive
from p2_listen.contracts import read
from listen_audio_postprocess.prepare import review_records
final=read(root/'experiments/listen_official_chinese100_20261026/results.json')
expected={r['sample_id']:r for r in review_records(root/'experiments/listen_official_chinese100_20261026/results.json')}
for c in final['cases']:
 sid=c['sample_id'];folder=root/'data_previews/listen_benchmark_review/audio_official_random100/media'/sid
 label=read(folder/'label.json');raw=read(root/'experiments/listen_official_random100_20261026/raw_metadata'/f'{sid}_listener.json')
 job=dict(sample_id=sid,target_audio=str(folder/'listener.wav'),source_audio_sha256=c['gate']['source_audio_sha256'],duration_s=c['duration_s'],start=c['clip_absolute_start'],integrated_sha256=c['label_sha256'],events=[e for e in label['events'] if e.get('modality')=='audio'])
 record,audit=derive(job,raw,{})
 for key in ['sample_id','source_audio_sha256','audio_events','extra_mute_intervals']:
  assert record[key]==expected[sid][key],(sid,key)
print('PASS: all 100 bulk-derived audio events, text, boundaries, exclusions and Chinese mute ranges match approved review')
