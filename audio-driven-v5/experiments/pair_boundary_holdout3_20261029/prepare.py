"""Freeze a seeded new sample before comparing segmentation results."""
import json,random,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];EXP=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
import listen_local_recovery as base
from p3_listen.common import sha
PROD=ROOT/'data_corpora/seamless_listen_local_full'
def read(p):return json.loads(p.read_text())
def write(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2,default=lambda x:x.item())+'\n')
def main():
 selection=EXP/'selection.json'
 if not selection.exists():
  rows=read(PROD/'catalog.json')['rows'];random.Random(20261030).shuffle(rows)
  excluded={p.stem for p in (ROOT/'experiments/pair_boundary_real_audit_20261029').glob('*.json')}
  excluded.update(r['pair'] for r in read(ROOT/'data_previews/seamless_listen_full_source_recovery/index.json')['pairs'])
  chosen=[]
  for r in rows:
   pair=r['pair'];ap=PROD/'audit'/f'{pair}.json'
   if pair in excluded or not ap.exists():continue
   audit=read(ap)
   if not 120<=audit['source_seconds']<=600:continue
   if not all((PROD/k/f'{pair}.json').exists() for k in ['results','normalized_pairs']):continue
   if not (ROOT/'data_corpora/seamless_listen_v8_audio_final_full/results'/f'{pair}.json').exists():continue
   chosen.append(r)
   if len(chosen)==3:break
  assert len(chosen)==3
  write(selection,dict(seed=20261030,criteria='first 3 after seeded catalog shuffle; source 120..600 seconds; production and normalized audio available; exclude all prior audit snapshots and published sources; chosen before running variants',excluded=sorted(excluded),items=chosen,production_results={r['pair']:sha(PROD/'results'/f'{r["pair"]}.json') for r in chosen}))
 for item in read(selection)['items']:
  path=EXP/(item['pair']+'.json')
  if path.exists():continue
  audit=read(PROD/'audit'/f'{item["pair"]}.json')
  ctx=base.SourceContext(item,shot_times=audit['shots'])
  write(path,dict(pair=item['pair'],item=item,people=ctx.people,rms=[x.tolist() for x in ctx.rms],limit=ctx.limit,sources=ctx.sources,shots=ctx.shots,shots_by_track=ctx.shots_by_track,regions=ctx.regions,source_gaps=ctx.source_gaps))
  print('PREPARED',item['pair'],ctx.limit,ctx.shots,flush=True)
if __name__=='__main__':main()
