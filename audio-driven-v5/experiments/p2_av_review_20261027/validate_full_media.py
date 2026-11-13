import hashlib,json
from pathlib import Path
from urllib.request import Request,urlopen
OUT=Path(__file__).resolve().parent;data=json.loads((OUT/'data.json').read_text());counts={'samples':0,'gt':0,'asr':0,'mixed_outputs':0,'guided':0,'fixed':0}
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for dataset in data['datasets']:
 assert len(dataset['samples'])==24 and all(r['outputs']==48 for r in dataset['metrics'])
 for sample in dataset['samples']:
  counts['samples']+=1;counts['asr']+=1;assert sample['asr']['sha256']==sha((OUT/sample['speaker']).resolve())
  gt=OUT/sample['gt_video'];receipt=read(gt.with_suffix('.json'));identity=receipt['identity'];assert receipt['sha256']==sha(gt)
  assert identity['audio_sha256']==sha(Path(sample['sources']['gt_audio'])) and identity['video_sha256']==sha(Path(sample['sources']['gt_video']));assert identity['speaker'] is None;counts['gt']+=1
  assert len(sample['outputs'])==16
  for output in sample['outputs']:
   video=OUT/output['mixed_video'];receipt=read(video.with_suffix('.json'));identity=receipt['identity'];assert receipt['sha256']==sha(video)
   assert identity['speaker_sha256']==sha(Path(sample['sources']['speaker'])) and identity['audio_sha256']==sha((OUT/output['audio']).resolve()) and identity['video_sha256']==sha((OUT/output['video']).resolve())
   assert identity['duration_s']==sample['duration'];assert receipt['mix']['listener_gain']==receipt['mix']['speaker_gain']<=1
   counts['mixed_outputs']+=1;counts[output['variant']]+=1
url='http://127.0.0.1:8876/p2_av_review/'+data['datasets'][0]['samples'][0]['outputs'][0]['mixed_video']
with urlopen(Request(url,headers={'Range':'bytes=0-1023'})) as response:assert response.status==206 and len(response.read())==1024
counts.update(status='passed',range_response=206);(OUT/'full_media_validation.json').write_text(json.dumps(counts,indent=2)+'\n');print(counts)
