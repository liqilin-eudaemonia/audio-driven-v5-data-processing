"""Serial export of the two newly evaluated sources and seven-source overview."""
import ctypes
assert ctypes.CDLL(None).prctl(41,1,0,0,0)==0
import copy,json,os,shutil,sys
from pathlib import Path
EXP=Path(__file__).resolve().parent;ROOT=EXP.parents[1]
OLD=ROOT/'experiments/pair_boundary_real_audit_20261029';HOLD=ROOT/'experiments/pair_boundary_holdout3_20261029';HEAD=ROOT/'experiments/pair_head_attach5_20261029'
sys.path[:0]=[str(OLD),str(HOLD)]
import export_fixed_pairs as export
import compare
read=compare.read;write=export.write

def main():
 report=read(EXP/'comparison.json');fresh_pairs=[r['pair'] for r in report['pairs'][:2]]
 for pair in fresh_pairs:
  evidence=export.PREVIEW/pair/'evidence';evidence.mkdir(exist_ok=True)
  for folder in ['headseven','seven_before']:
   target=evidence/folder
   if not target.exists() and not target.is_symlink():target.symlink_to(EXP/'review'/folder,target_is_directory=True)
 cache={};versions={}
 for variant,prefix,media_dir in [('before','B','seven_before'),('latest','H','headseven')]:
  export.RESULTS=EXP/('baseline_results' if variant=='before' else 'results');export.OUT=EXP/'review'/media_dir;export.ID_PREFIX=prefix;export.MEDIA_DIR=media_dir
  pairs=[export.view(pair,i+6) for i,pair in enumerate(fresh_pairs)]
  for d in pairs:
   d.update(holdout_sample=True,variant_title='历史连续审核版' if variant=='before' else '最终 H 规则 · 七来源核验',review_scope='seven explicitly requested source conversations',scope_description='本条此前尚未应用 H 规则。本轮与原 HTML 的连续审核结果比较，规则本身未改动。',variant_description='初始候选、最终通过片段和音质待复核结果分别展示；原版保留不代表新版也通过。')
   for r in d['rows']:
    key=(d['pair'],*compare.sig(r));prev=cache.get(key);folder=export.OUT/r['id']
    if prev:
     for name,url in [('pair.mp4',prev['media']),('poster.jpg',prev['poster']),('A.wav',prev['audio_urls']['A']),('B.wav',prev['audio_urls']['B'])]:
      source=export.PREVIEW/url;dest=folder/name
      if not dest.exists():
       try:os.link(source,dest)
       except OSError:shutil.copyfile(source,dest)
     r['media_check']=dict(prev['media_check'],reused_exact_interval=True)
    else:export.export(r)
    cache[key]=r
  versions[variant]=pairs
  write(export.OUT/'manifest.json',dict(pairs=pairs,production_modified=False))
 for old,new,rpt in zip(versions['before'],versions['latest'],report['pairs'][:2]):
  before=[r for r in old['rows'] if r['id'] in old['terminal_ids'] and r['pool']=='keep_candidate'];after=[r for r in new['rows'] if r['id'] in new['terminal_ids'] and r['pool']=='keep_candidate']
  new.update(comparison_before_label='原审核版：最终保留（B 编号）',comparison_after_label='H 版：通过与待复核提案分列',comparison_description='左右均播放实际 pair，无额外上下文。右侧同时列出本范围的新通过片段与未通过的提案，后者不计入最终保留；请以每段状态为准。',comparison_cases=[])
  lost=rpt['kept']['previous_speaker_words_lost'];added=rpt['kept']['added_speaker_words']
  for i,g in enumerate(compare.changes(before,after),1):
   missing=[w for w in lost if compare.covered(w,g['before'])]
   gained=[w for w in added if compare.covered(w,g['after'])]
   proposals=[r for r in new['rows'] if r['initial'] and r['pool']!='keep_candidate' and any(compare.covered(w,[r]) for w in missing)]
   right=sorted(g['after']+proposals,key=lambda r:(r['start'],r['end']))
   if missing:
    text=' '.join(w['text'] for w in missing)
    note=f'原先保留的 {len(missing)} 个对齐词未进入新版通过集：{text}。这些词仍在新初始候选里，但其所在候选未整体通过音质规则，局部恢复未全部保回；不能直接认定是噪声或误杀。'
    title=f'需要复核：原保留内容减少 · {g["start"]:.3f}–{g["end"]:.3f} 秒'
   elif gained:
    note='新增对齐词：'+' '.join(w['text'] for w in gained)+'。这里追加的是下一句开头的 He / He’s，片段可能停在未完句首，不能仅凭多保留单词就判断改善。'
    title=f'边界待判断：句尾增加下一句开头 · {g["start"]:.3f} 秒'
   else:
    note='已覆盖 speaker 单词没有变化，仅边界留白变化；仍需结合原有边界提示判断。';title=f'留白调整 · {g["start"]:.3f} 秒'
   new['comparison_cases'].append(dict(id=f'seven_diff{i}',title=title,note=note,before=g['before'],after=right,missing_words=missing,added_words=gained))
  new['seven_validation_note']=f'七来源核验：本条最终通过 {rpt["kept"]["old_count"]}→{rpt["kept"]["new_count"]} 段，通过覆盖 {rpt["old_kept_seconds"]:.3f}→{rpt["new_kept_seconds"]:.3f} 秒；{len(lost)} 个原保留对齐词未进入新通过集。本轮未达到“只有明确优化”的全量启动条件，未启动全量。'
  new['seven_comparison']=rpt
 combined=[]
 old_head={d['pair']:d for d in read(HEAD/'review/head_attach/manifest.json')['pairs']}
 for rpt in report['pairs']:
  d=next((x for x in versions['latest'] if x['pair']==rpt['pair']),None)
  if d is None:
   d=copy.deepcopy(old_head[rpt['pair']]);d['seven_comparison']=rpt
   d['seven_validation_note']=f'七来源核验：本条与上一轮 H 版一致，最终通过 {rpt["kept"]["new_count"]} 段，词覆盖和音质结论未变化。下方保留此前的 L→H 修复对照。'
  combined.append(d)
 # Existing source/evidence paths are preserved; new media get separate links.
 for pair in fresh_pairs:
  evidence=export.PREVIEW/pair/'evidence';evidence.mkdir(exist_ok=True)
  for folder in ['headseven','seven_before']:
   target=evidence/folder
   if not target.exists():target.symlink_to(EXP/'review'/folder,target_is_directory=True)
 write(EXP/'review/manifest.json',dict(pairs=combined,before_versions=versions['before'],full_run_gate=report['full_run_gate'],production_modified=False))
 write(EXP/'review/headseven/manifest.json',dict(pairs=versions['latest'],production_modified=False))
 print('EXPORTED seven-source comparison',flush=True)
if __name__=='__main__':main()
