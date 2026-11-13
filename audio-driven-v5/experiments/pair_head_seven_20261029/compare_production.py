"""Compare the actual accepted full-corpus output with latest H, serial media export."""
import ctypes
assert ctypes.CDLL(None).prctl(41,1,0,0,0)==0
import collections,copy,csv,hashlib,json,os,sys
from pathlib import Path
EXP=Path(__file__).resolve().parent;ROOT=EXP.parents[1]
OLD=ROOT/'experiments/pair_boundary_real_audit_20261029';HOLD=ROOT/'experiments/pair_boundary_holdout3_20261029'
sys.path[:0]=[str(OLD),str(HOLD)]
import compare,export_fixed_pairs as export
read=compare.read;write=export.write
PROD=ROOT/'data_corpora/seamless_listen_local_full';PREVIEW=export.PREVIEW
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
keep=lambda d:[r for r in d['clips'] if r['audio_final']['pool']=='keep_candidate']
def simple(r):return dict(speaker='AB'[r['speaker_index']],start=r['start'],end=r['end'],text=r['text'],pool=r['audio_final']['pool'],reasons=r['audio_final']['reasons'])
def matched(a,b):
 aa={compare.sig(r):r for r in a};bb={compare.sig(r):r for r in b}
 equal=[dict(start=k[1],before=[simple(aa[k])],after=[simple(bb[k])],unchanged=True) for k in aa.keys()&bb.keys()]
 changed=[dict(start=g['start'],before=[simple(r) for r in g['before']],after=[simple(r) for r in g['after']],unchanged=False) for g in compare.changes(a,b)]
 return sorted(equal+changed,key=lambda g:g['start'])
def label(g):
 a,b=g['before'],g['after']
 if not a:return '新增通过片段'
 if not b:return '旧通过片段未进入新通过集'
 if len(a)>1 or len(b)>1:return f'重新分段：{len(a)}→{len(b)} 段'
 x,y=a[0],b[0]
 if y['start']<=x['start'] and y['end']>=x['end']:return '边界扩展'
 if y['start']>=x['start'] and y['end']<=x['end']:return '边界收缩'
 return '起止调整'
def main():
 baseline=EXP/'review/manifest_before_production.json'
 if not baseline.exists():write(baseline,read(EXP/'review/manifest.json'))
 manifest=read(baseline);cache={};reports=[];oldsets=[];csvrows=[]
 contract=read(PROD/'contract.json');hashes={}
 # Read only the seven known review JSONs, no corpus scan.
 for d in manifest['pairs']:
  view=read(PREVIEW/d['pair']/'review.json')
  for v in view.get('fixed_pair_versions',[]):
   for r in v['rows']:
    if (PREVIEW/r['media']).exists():cache[(d['pair'],*compare.sig(r))]=r
 for i,d in enumerate(manifest['pairs'],1):
  pair=d['pair'];source=read(EXP/(pair+'.json'));path=PROD/'results'/(pair+'.json');hashes[pair]=sha(path)
  old=read(path);new=read(EXP/'results'/pair/'result.json')
  assert old['complete'] and new['complete']
  allrows=old['clips']+old['superseded_attempts']
  adapted=dict(old,initial_ids=[r['sample_id'] for r in allrows if not r.get('recovery_parent_id')],kept_seconds=sum(r['end']-r['start'] for r in keep(old)),pool_counts=dict(collections.Counter(r['audio_final']['pool'] for r in old['clips'])))
  write(EXP/'production_baseline'/pair/'result.json',adapted)
  write(EXP/'production_baseline/contract.json',dict(revision=contract['revision'],segmentation_revision=contract['segmentation_revision'],pause_threshold_seconds=3,source_contract_sha256=sha(PROD/'contract.json')))
  stages=[]
  for stage,a,b in [('最终通过',keep(old),keep(new)),('全部终端片段（含待复核／排除）',old['clips'],new['clips'])]:
   gs=matched(a,b);stages.append(dict(stage=stage,before_count=len(a),after_count=len(b),unchanged=sum(g['unchanged'] for g in gs),groups=gs))
   for j,g in enumerate(gs,1):
    for side in ['before','after']:
     for row in g[side]:csvrows.append(dict(source=i,pair=pair,stage=stage,group=j,unchanged=g['unchanged'],version='旧全量' if side=='before' else '最新版H',**row))
  metrics=compare.evaluate(source,keep(old),keep(new))
  reports.append(dict(pair=pair,source_number=i,source_seconds=source['limit'],baseline_file=str(path.relative_to(ROOT)),baseline_sha256=hashes[pair],baseline_revision=contract['revision'],baseline_segmentation_revision=contract['segmentation_revision'],latest_revision=new['revision'],old_kept_seconds=adapted['kept_seconds'],new_kept_seconds=new['kept_seconds'],kept=metrics,stages=stages,title='旧全量版 → 最新版：逐段起止时间与内容',description='旧版直接取自已完成全量的 seamless_listen_local_full/results，不使用小批试验或同版重跑。时间均相对原视频，以秒计；最终通过与全部终端片段分开比较。',before_label='旧全量：起止／原文／状态',after_label='最新版：起止／原文／状态'))
  export.RESULTS=EXP/'production_baseline';export.OUT=EXP/'review/production_before';export.MEDIA_DIR='production_before';export.ID_PREFIX='P'
  v=export.view(pair,i);v.update(holdout_sample=True,variant_title='旧全量版：2026-10-23 已完成全量的实际结果',scope_description='直接读取旧全量结果；P 为展示编号。此处初始候选已包含旧流程对既有音质阻断证据的处理，不能与实验中先切完整候选的数量直接等同。')
  link=PREVIEW/pair/'evidence/production_before'
  if not link.is_symlink():link.symlink_to(export.OUT,target_is_directory=True)
  for row in v['rows']:
   cached=cache.get((pair,*compare.sig(row)))
   if cached:
    for key in ['media','poster','audio_urls','media_check']:row[key]=copy.deepcopy(cached[key])
    row['media_check']['reused_exact_interval']=True
   else:export.export(row)
  oldsets.append(v)
  before=[r for r in v['rows'] if r['id'] in v['terminal_ids'] and r['pool']=='keep_candidate']
  after=[r for r in d['rows'] if r['id'] in d['terminal_ids'] and r['pool']=='keep_candidate']
  cases=[];lost=metrics['previous_speaker_words_lost'];added=metrics['added_speaker_words']
  for j,g in enumerate(compare.changes(before,after),1):
   removed=[w for w in lost if compare.covered(w,g['before'])];gained=[w for w in added if compare.covered(w,g['after'])]
   note='左右仅列最终通过的实际 pair；未通过的候选单独折叠展示，不计入保留。'
   if removed:note+=' 原通过、现未覆盖的对齐词：'+' '.join(w['text'] for w in removed)+'。是否误排需试听，不能视作明确优化。'
   if gained:note+=' 新增覆盖的对齐词：'+' '.join(w['text'] for w in gained)+'。新增词不自动等于语义完整。'
   if not removed and not gained:note+=' 原先覆盖的可靠对齐词未减少；请结合原文判断留白或句子分组是否合理。'
   if len(g['before'])==len(g['after'])==1:
    a,b=g['before'][0],g['after'][0];note+=f' 起点变化 {b["start"]-a["start"]:+.6f} 秒，终点变化 {b["end"]-a["end"]:+.6f} 秒。'
   proposals=[r for r in d['rows'] if r['initial'] and r['pool']!='keep_candidate' and any(compare.covered(w,[r]) for w in removed)]
   cases.append(dict(id=f'production_diff{j}',title=f'{label(g)} · {g["start"]:.6f}–{g["end"]:.6f} 秒',note=note,before=g['before'],after=g['after'],excluded_proposals=proposals,missing_words=removed,added_words=gained))
  for c in d.get('comparison_cases',[]):
   c['archived']=True;c['before_label']=c.get('before_label',d.get('comparison_before_label','上一试验版'));c['after_label']=c.get('after_label',d.get('comparison_after_label','H 试验版'))
  d['comparison_cases']=cases+d.get('comparison_cases',[])
  d.pop('adjacent_comparison',None)
  d.update(interval_comparison=reports[-1],comparison_title='旧全量版 → 最新版：全部变化的处理前后试听',comparison_before_label='旧全量版：最终通过',comparison_after_label='最新版：最终通过',comparison_description='所有七条统一以已完成全量的实际结果为基线。P 是旧全量片段编号，H 是最新版编号；左右只展示最终通过片段，音质待复核／排除提案另列。',scope_description='当前主对照为旧全量 seamless_listen_local_full → 最新 H，包含此前累计修复。不是相邻 L→H，也不是 H 同版重跑。',variant_title='最新版 H · 对比此前已认可的全量版',holdout_sample=True,seven_validation_note=f'旧全量→最新版：最终通过 {metrics["old_count"]}→{metrics["new_count"]} 段，{metrics["unchanged"]} 段起止完全一致，{metrics["changed_groups"]} 组变化；通过覆盖 {adapted["kept_seconds"]:.6f}→{new["kept_seconds"]:.6f} 秒。完整时间与原文见下方，变化可逐组试听。')
  print('SOURCE',i,pair,'kept',metrics['old_count'],'->',metrics['new_count'],'same',metrics['unchanged'],'groups',metrics['changed_groups'],'lost words',len(lost),flush=True)
 summary=dict(old_count=sum(r['kept']['old_count'] for r in reports),new_count=sum(r['kept']['new_count'] for r in reports),unchanged=sum(r['kept']['unchanged'] for r in reports),changed_groups=sum(r['kept']['changed_groups'] for r in reports),lost_words=sum(len(r['kept']['previous_speaker_words_lost']) for r in reports))
 gate=dict(passed=False,full_run_started=False,reason='旧全量已通过内容在三条来源减少，不能判为仅明确优化',blocking_sources=[r['pair'] for r in reports if r['kept']['previous_speaker_words_lost']])
 write(EXP/'production_comparison.json',dict(pairs=reports,summary=summary,full_run_gate=gate,source_contract_sha256=sha(PROD/'contract.json'),comparison_scope='accepted full-corpus output vs latest H for all seven'))
 with (EXP/'production_intervals.csv').open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=list(csvrows[0]));writer.writeheader();writer.writerows(csvrows)
 manifest['before_versions']+=oldsets;manifest['full_run_gate']=gate;manifest['primary_comparison']='accepted production vs latest H';manifest['production_comparison_summary']=summary
 write(EXP/'review/production_before/manifest.json',dict(pairs=oldsets));write(EXP/'review/manifest.json',manifest)
 assert all(sha(PROD/'results'/(pair+'.json'))==digest for pair,digest in hashes.items())
 write(EXP/'production_validation.json',dict(production_result_hashes_unchanged=hashes,baseline_contract=contract['revision'],baseline_segmentation=contract['segmentation_revision'],summary=summary,full_run_started=False,serial_media_export=True))
 print('COMPLETE',summary,flush=True)
if __name__=='__main__':main()
