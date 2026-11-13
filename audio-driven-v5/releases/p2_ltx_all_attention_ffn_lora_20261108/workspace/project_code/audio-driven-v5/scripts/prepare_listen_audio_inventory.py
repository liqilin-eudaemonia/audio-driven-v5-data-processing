import sys,json,re
from pathlib import Path
ROOT=Path('/private/liqi_workspace/project_code/audio-driven-v5');sys.path.insert(0,str(ROOT/'src'))
from p3_listen import media
out=ROOT/'experiments/p3_listen_v6_audio_ab';out.mkdir(exist_ok=True)
samples=json.loads((ROOT/'data_previews/seamless_listen_v6_review/samples.json').read_text())['samples'];records={}
for row in samples:
 norm=json.loads((ROOT/'data_corpora/seamless_listen_300h/normalized_pairs'/f"{row['pair']}.json").read_text())
 for role in ('listener','speaker'):
  ident=row[role+'_id'];ref=row['sources'][role];name=str(Path(ref['member']).with_suffix('.json'))
  if ident in records:continue
  offset,size=media.members(ref['archive'])[name]
  with open(ref['archive'],'rb') as f:f.seek(offset);meta=json.loads(f.read(size))
  transcript=meta.get('metadata:transcript',[])
  records[ident]=dict(id=ident,source=ref,metadata=meta,normalization=norm['tracks'][row[role+'_index']],pair=row['pair'])
  matches=[s for s in transcript if re.search(r'laugh|chuckl|giggl|ha ha|haha',s.get('transcript',''),re.I)]
  print(ident,'segments',len(transcript),'laugh_candidates',[(s.get('start'),s.get('end'),s.get('transcript')) for s in matches],flush=True)
(out/'source_inventory.json').write_text(json.dumps(records,ensure_ascii=False,indent=2))
