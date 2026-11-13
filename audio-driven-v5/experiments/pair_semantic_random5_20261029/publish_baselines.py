import ctypes
assert ctypes.CDLL(None).prctl(41,1,0,0,0)==0
"""Append the three holdout sources to the existing review HTML."""
import sys,subprocess
from pathlib import Path
from types import SimpleNamespace
EXP=Path(__file__).resolve().parent;ROOT=EXP.parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src'),str(EXP),str(ROOT/'experiments/pair_boundary_holdout3_20261029')]
from review_listen_full_source import classify_timeline,compact
from p3_listen.common import sha
import listen_local_recovery as base
from compare import read
import json
PREVIEW=ROOT/'data_previews/seamless_listen_full_source_recovery'
PROD=ROOT/'data_corpora/seamless_listen_local_full';PREV=ROOT/'data_corpora/seamless_listen_v8_audio_final_full'
def write(p,d):
 tmp=p.with_suffix(p.suffix+'.pending');tmp.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
def main():
 old=read(EXP/'review/production_before/manifest.json')['pairs'];new=read(EXP/'review/h_manifest.json')['pairs'];entries=[]
 links=[dict(pair=d['pair'],label=f'随机来源 {i+1} · {d["source_seconds"]:.0f} 秒') for i,d in enumerate(new)]
 for g,l in zip(old,new):
  pair=l['pair'];d=read(EXP/(pair+'.json'));media=read(EXP/(pair+'.media.json'));folder=PREVIEW/pair
  audit=read(PROD/'audit'/f'{pair}.json');prod=read(PROD/'results'/f'{pair}.json');prev=read(PREV/'results'/f'{pair}.json')
  ctx=SimpleNamespace(limit=d['limit'],regions=audit['regions'],source_gaps=audit['source_gaps'],shots=d['shots'],shots_by_track=d['shots_by_track'])
  final=[r for r in prod['clips'] if r['audio_final']['pool']=='keep_candidate']
  attempts=sorted(prod['clips']+prod.get('superseded_attempts',[]),key=lambda r:(r['start'],r['end']))
  before=[(r['start'],r['end']) for r in prev['clips'] if r['audio_final']['pool']=='keep_candidate'];after=[(r['start'],r['end']) for r in final]
  summary=dict(pair=pair,source_seconds=d['limit'],old_kept_clips=len(before),old_kept_seconds=base.seconds(before),final_kept_clips=len(final),new_kept_seconds=base.seconds(after),added_seconds=base.seconds(base.subtract(after,before)),removed_seconds=base.seconds(base.subtract(before,after)),training_admitted=False)
  rows=[]
  for r in attempts:
   v=compact(r);v['crops']=[dict(status='unmeasured')]*2;rows.append(v)
  link=folder/'evidence'
  if not link.exists():link.symlink_to(EXP/'review',target_is_directory=True)
  value=dict(pair=pair,summary=summary,media=pair+'/source.mp4',media_verification=media['media'],panels=media['panels'],old=[compact(r) for r in prev['clips']],attempts=rows,kept_ids=[r['sample_id'] for r in final],timeline=classify_timeline(ctx,prev['clips'],attempts,final,audit['diagnostic']),shots=d['shots_by_track'],scene_detection_applied=False,regions=ctx.regions,source_gaps=ctx.source_gaps,diagnostic=audit['diagnostic'],overview_crops=[],old_unresolved=[],transcripts=[[dict(start=s['start'],end=s['end'],text=s['text']) for s in p['segments']] for p in d['people']],issues=[],review_kind='holdout_rule_comparison',holdout_sample=True,fixed_pairs=l,fixed_pairs_previous=g,fixed_pair_versions=[l,g],fixed_source_links=links,production_result_sha256=sha(PROD/'results'/f'{pair}.json'))
  write(folder/'review.json',value);entries.append(dict(pair=pair,summary=summary,holdout_sample=True))
 index=read(PREVIEW/'index.json');names={e['pair'] for e in entries};index['pairs']=[e for e in index['pairs'] if e['pair'] not in names]+entries;index['random5_sample_seed']=20261031;write(PREVIEW/'index.json',index)
 # Existing publisher preserves additional entries and refreshes shared assets/fonts.
 # Shared assets are refreshed after J overlays are ready.
 print('PUBLISHED 5 random sources',flush=True)
if __name__=='__main__':main()
