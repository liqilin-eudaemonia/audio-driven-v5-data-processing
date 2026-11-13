"""Prepare original nod sentences for text/visual review; never relabel events."""
import json,re,hashlib
from pathlib import Path
R=Path(__file__).resolve().parent
rows=[]
for arm in ['observation_v1','observed_actions_v2']:
 p=R/f'descriptions_{arm}.json'
 if not p.exists():continue
 for x in json.loads(p.read_text()):
  if not re.search(r'\bnod\w*',x['behavior'],re.I):continue
  sentences=[s for s in re.split(r'(?<=[.!?])\s+',x['behavior']) if re.search(r'\bnod\w*',s,re.I)]
  negative_cue=any(re.search(r'\b(no|not|without|never|absence|none)\b',s,re.I) for s in sentences)
  rows.append(dict(arm=arm,**x,nod_sentences=sentences,has_negation_cue=negative_cue,description_sha256=hashlib.sha256(x['behavior'].encode()).hexdigest(),semantic_polarity_not_automatically_assigned=True,media_truth_not_assigned=True))
(R/'nod_mentions.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
unique={x['description_sha256']:x for x in rows}
for x in unique.values():print(x['description_sha256'][:12],x['arm'],x['sample_id'],x['view'],'negation_cue='+str(x['has_negation_cue']),repr(x['nod_sentences']))
print('Total mentions',len(rows),'unique descriptions',len(unique))
