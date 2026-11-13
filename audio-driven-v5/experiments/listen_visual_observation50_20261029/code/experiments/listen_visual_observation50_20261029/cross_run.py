"""Cross-run input audit and descriptive statistics, with no GT assumptions."""
import collections,json,statistics
from pathlib import Path
R=Path(__file__).resolve().parent;P=R.parent/'listen_visual_observable50_20261029'
current=json.loads((R/'samples.json').read_text());previous=json.loads((P/'samples.json').read_text());assert current==previous
replays=[];durations=collections.defaultdict(list);per_request=collections.defaultdict(list);text_lengths=collections.defaultdict(list)
for p in sorted((R/'results').glob('*.json')):
 r=json.loads(p.read_text());old=json.loads((P/'results'/p.name).read_text());b={(v['arm'],v['view']):v for v in old['records']}
 for v in r['records']:
  original=b[('baseline',v['view'])]
  assert original['media']==v['media'] and original['start']==v['start'] and original['end']==v['end']
  if v['arm']=='baseline':
   assert original['prompt']==v['prompt'];replays.append(dict(sample_id=r['sample_id'],view=v['view'],raw_identical=original['raw']==v['raw'],status_identical=original['status']==v['status']))
  if v['status']=='valid':
   es=v['parsed']['events'];per_request[v['arm']].append(len(es))
   durations[v['arm']].extend(e['end']-e['start'] for e in es)
   text_lengths[v['arm']].extend(len(e['behavior']) for e in es)
out=dict(completed_samples=len(list((R/'results').glob('*.json'))),same_sample_manifest=True,same_media_as_previous=True,baseline_replayed_requests=len(replays),baseline_raw_identical=sum(r['raw_identical'] for r in replays),baseline_status_identical=sum(r['status_identical'] for r in replays),replays=replays,stats={a:dict(median_candidate_duration_s=statistics.median(v),median_candidates_per_valid_request=statistics.median(per_request[a]),mean_description_chars=statistics.mean(text_lengths[a])) for a,v in durations.items() if v})
(R/'cross_run.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k!='replays'},ensure_ascii=False,indent=2))
