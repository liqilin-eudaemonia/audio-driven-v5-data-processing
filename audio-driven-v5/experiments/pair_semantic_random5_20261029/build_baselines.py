"""Render five full sources and exact P/H pair media, all serial."""
import ctypes
assert ctypes.CDLL(None).prctl(41,1,0,0,0)==0
import collections,copy,json,sys
from pathlib import Path
from types import SimpleNamespace
EXP=Path(__file__).resolve().parent;ROOT=EXP.parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src'),str(ROOT/'experiments/pair_boundary_real_audit_20261029')]
import export_fixed_pairs as export
from review_listen_full_source import render_source
read=export.read;write=export.write;PROD=export.PROD;PREVIEW=export.PREVIEW
items=read(EXP/'selection.json')['items'];pviews=[];hviews=[]
for n,item in enumerate(items,8):
 pair=item['pair'];snapshot=read(EXP/(pair+'.json'));norm=read(PROD/'normalized_pairs'/(pair+'.json'))
 ctx=SimpleNamespace(limit=snapshot['limit'],sources=snapshot['sources'],geometry=[dict(rotation_degrees=0)]*2)
 panels,media=render_source(ctx,norm,PREVIEW/pair)
 write(EXP/(pair+'.media.json'),dict(panels=panels,media=media))
 old=read(PROD/'results'/(pair+'.json'));allrows=old['clips']+old['superseded_attempts']
 adapted=dict(old,initial_ids=[r['sample_id'] for r in allrows if not r.get('recovery_parent_id')],kept_seconds=sum(r['end']-r['start'] for r in old['clips'] if r['audio_final']['pool']=='keep_candidate'),pool_counts=dict(collections.Counter(r['audio_final']['pool'] for r in old['clips'])))
 write(EXP/'production_baseline'/pair/'result.json',adapted)
 write(EXP/'production_baseline/contract.json',dict(revision=read(PROD/'contract.json')['revision'],pause_threshold_seconds=3))
 cache={}
 for variant,folder,prefix,views in [('production_baseline','production_before','P',pviews),('h','h','H',hviews)]:
  export.RESULTS=EXP/variant;export.OUT=EXP/'review'/folder;export.MEDIA_DIR='random5_'+folder;export.ID_PREFIX=prefix
  view=export.view(pair,n)
  if prefix=='P':view.update(variant_title='旧全量版：实际最终结果',scope_description='来自原全量结果，未重新切分；旧流程初始候选含已有音质阻断处理。')
  else:view.update(variant_title='H 版：同一随机原始视频重跑',scope_description='沿用历史 H 规则，先切候选再音质筛选。')
  evidence=PREVIEW/pair/'evidence';evidence.mkdir(exist_ok=True)
  link=evidence/export.MEDIA_DIR
  if not link.exists():link.symlink_to(export.OUT,target_is_directory=True)
  for r in view['rows']:
   key=(r['speaker_index'],r['start'],r['end']);prev=cache.get(key)
   if prev:
    for k in ['media','poster','audio_urls','media_check']:r[k]=copy.deepcopy(prev[k])
    r['media_check']['reused_exact_interval']=True
   else:export.export(r);cache[key]=r
  views.append(view)
  write(EXP/'review'/folder/'manifest.json',dict(pairs=views))
 print('BASELINES DONE',pair,flush=True)
write(EXP/'review/h_manifest.json',dict(pairs=hviews))
