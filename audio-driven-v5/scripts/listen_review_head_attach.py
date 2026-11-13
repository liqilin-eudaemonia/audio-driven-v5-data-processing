"""Overlay the latest completed H experiment on the existing review page."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
EXP=ROOT/'experiments/pair_head_attach5_20261029'
PREVIEW=ROOT/'data_previews/seamless_listen_full_source_recovery'
def read(p):return json.loads(p.read_text())
def write(p,d):
 tmp=p.with_suffix(p.suffix+'.pending');tmp.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
def overlay():
 seven=ROOT/'experiments/pair_head_seven_20261029/review/manifest.json'
 path=seven if seven.exists() else EXP/'review/head_attach/manifest.json'
 if not path.exists():return
 manifest=read(path);pairs=manifest['pairs'];before=manifest.get('before_versions',[])
 links=[dict(pair=d['pair'],label=f'来源 {i+1} · {d["source_seconds"]:.0f} 秒') for i,d in enumerate(pairs)]
 for latest in pairs:
  dest=PREVIEW/latest['pair']/'review.json'
  if not dest.exists():continue
  d=read(dest);old=d.get('fixed_pair_versions') or [d.get('fixed_pairs'),d.get('fixed_pairs_previous')]
  versions=[v for v in old if v and v['revision']!=latest['revision']]
  for previous in before:
   if previous['pair']==latest['pair'] and not any(v['revision']==previous['revision'] for v in versions):versions.append(previous)
  d.update(fixed_pairs=latest,fixed_pairs_previous=versions[0] if versions else None,fixed_pair_versions=[latest]+versions,fixed_source_links=links)
  write(dest,d)
 index=read(PREVIEW/'index.json');index['head_attachment_revision']=pairs[0]['revision']
 if 'full_run_gate' in manifest:index['head_seven_gate']=manifest['full_run_gate']
 for trial in [ROOT/'experiments/pair_semantic_trial2_20261029/review/manifest.json',
               ROOT/'experiments/pair_semantic_remaining5_20261029/review/manifest.json',
               ROOT/'experiments/pair_semantic_random5_20261029/review/manifest.json']:
  if trial.exists():
   for latest in read(trial)['pairs']:
    dest=PREVIEW/latest['pair']/'review.json'
    d=read(dest);versions=[v for v in d['fixed_pair_versions'] if v['revision']!=latest['revision']]
    d.update(fixed_pairs=latest,fixed_pairs_previous=versions[0],fixed_pair_versions=[latest]+versions)
    write(dest,d)
   index['semantic_trial_revision']=read(trial)['pairs'][0]['revision']
 write(PREVIEW/'index.json',index)
