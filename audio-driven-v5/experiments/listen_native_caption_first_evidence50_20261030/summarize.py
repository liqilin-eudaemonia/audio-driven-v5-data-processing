import json,re
from pathlib import Path
from collections import Counter
ROOT=Path(__file__).resolve().parents[2]
RUN=Path(__file__).resolve().parent
rows=json.loads((RUN/'manifest.json').read_text())
runs={'first':ROOT/'experiments/listen_native_caption_clean50_20261030/text8', 'previous_v1':ROOT/'experiments/listen_native_caption_nod_final50_20261030', 'first_plus_evidence':RUN}
output={}
for name,folder in runs.items():
 counts=Counter();cases=[]
 for row in rows:
  sid=row['sample_id'];x=json.loads((folder/'results'/(sid+'.json')).read_text());t=x['caption']
  counts['samples']+=1;counts['normal_finish']+=x['finish_reason']=='stop';counts['retried_samples']+=len(x.get('attempts',[]))>1
  nod=bool(re.search(r'\bnod\w*\b',t,re.I));counts['nod_word_mentions']+=nod
  counts['with_cues_and_nod_word']+=nod and bool(row['nod_supplement']['events'])
  counts['possible_interpretation_wording']+=bool(re.search(r'\b(indicat\w*|suggest\w*|as if|thoughtful|contemplative|pensive|acknowledg\w*)\b',t,re.I))
  counts['assistant_prefix']+=t.startswith('Assistant:')
  timed=bool(re.search(r'\b(?:\d+(?:\.\d+)?|one|two|three|four|five|six|seven|eight|nine|ten)(?:[- ](?:point[- ]\w+|and[- ]\w+))?[- ]seconds?\b',t,re.I))
  counts['number_and_seconds_wording']+=timed
  risk=bool(row['audio_preflight']['full_silence'] and re.search(r'\b(?:while (?:speaking|talking|laughing)|as (?:he|she|they) (?:speaks|talks|laughs))\b',t,re.I))
  counts['silent_speech_pattern_review']+=risk
  cases.append(dict(sample_id=sid,caption=t,finish_reason=x['finish_reason'],full_silence=row['audio_preflight']['full_silence'],nod_word=nod,silent_speech_pattern_review=risk))
 output[name]=dict(counts=dict(counts),cases=cases)
report=dict(semantic_quality_verified=False,counters_are_text_screening_not_accuracy=True,runs=output)
(RUN/'comparison.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v['counts'] for k,v in output.items()},ensure_ascii=False))
