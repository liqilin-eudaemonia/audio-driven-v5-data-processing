"""Uniform L-to-H interval/content comparison, keeping the historical B audit separate."""
import csv,copy,json,sys
from pathlib import Path
EXP=Path(__file__).resolve().parent;ROOT=EXP.parents[1]
OLD=ROOT/'experiments/pair_boundary_real_audit_20261029';HOLD=ROOT/'experiments/pair_boundary_holdout3_20261029'
sys.path[:0]=[str(OLD),str(HOLD)]
import compare,export_fixed_pairs as export
read=compare.read;write=export.write
manifest=read(EXP/'review/manifest.json')
# Keep the original published comparison available for reproducibility.
backup=EXP/'review/manifest_before_uniform.json'
if not backup.exists():write(backup,manifest)
manifest=read(backup)
reports=[];csvrows=[];new_versions=[]
slim=lambda r:dict(speaker='AB'[r['speaker_index']],start=r['start'],end=r['end'],text=r['text'],pool=r['audio_final']['pool'])
def groups(a,b):
 am={compare.sig(r):r for r in a};bm={compare.sig(r):r for r in b}
 same=[dict(start=k[1],before=[slim(am[k])],after=[slim(bm[k])],unchanged=True) for k in am.keys()&bm.keys()]
 changed=[dict(start=g['start'],before=[slim(r) for r in g['before']],after=[slim(r) for r in g['after']],unchanged=False) for g in compare.changes(a,b)]
 return sorted(same+changed,key=lambda g:g['start'])
for i,d in enumerate(manifest['pairs']):
 pair=d['pair'];folder=EXP/'previous_l' if i<2 else HOLD/'localbc' if i<5 else OLD/'localbc_pairs'
 a=read(folder/pair/'result.json');b=read(EXP/'results'/pair/'result.json')
 stages=[]
 for stage,aa,bb in [('初始切分',compare.roots(a),compare.roots(b)),('最终通过',[r for r in a['clips'] if r['audio_final']['pool']=='keep_candidate'],[r for r in b['clips'] if r['audio_final']['pool']=='keep_candidate'])]:
  gs=groups(aa,bb);same=sum(g['unchanged'] for g in gs)
  stages.append(dict(stage=stage,before_count=len(aa),after_count=len(bb),unchanged=same,groups=gs))
  for j,g in enumerate(gs,1):
   for side in ['before','after']:
    for r in g[side]:csvrows.append(dict(source=i+1,pair=pair,stage=stage,group=j,unchanged=g['unchanged'],version='L' if side=='before' else 'H',**r))
 report=dict(pair=pair,before_revision=a['revision'],after_revision=b['revision'],baseline_file=str((folder/pair/'result.json').relative_to(ROOT)),stages=stages)
 reports.append(report);d['adjacent_comparison']=report
 final=stages[1]
 d['seven_validation_note']=f'统一比较 L→H：最终通过 {final["before_count"]}→{final["after_count"]} 段，其中 {final["unchanged"]} 段起止时间完全一致。初始切分与最终通过的逐段时间、原文见下表。历史 B→H 的减少另列，不归因于这次 L→H 补接。'
 if i<2:
  export.RESULTS=folder;export.OUT=EXP/'review/previous_l';export.ID_PREFIX='L';export.MEDIA_DIR='previous_l'
  v=export.view(pair,i+6);cache={compare.sig(r):r for r in d['rows']}
  for row in v['rows']:
   h=cache[compare.sig(row)]
   assert row['text']==h['text'] and row['pool']==h['pool'] and row['reasons']==h['reasons']
   for key in ['media','poster','audio_urls','media_check']:row[key]=copy.deepcopy(h[key])
   row['media_check']['reused_exact_interval']=True
  evidence=export.PREVIEW/pair/'evidence/previous_l'
  if not evidence.is_symlink():evidence.symlink_to(export.OUT,target_is_directory=True)
  v.update(holdout_sample=True,variant_title='上一版 L：局部反馈＋句末优先，未加句首补接',scope_description='与 H 使用相同源数据、音质规则；本条所有候选与最终通过片段的起止、原文、音质结论均一致。')
  new_versions.append(v)
  for c in d['comparison_cases']:
   c['title']='历史 B→H（非相邻版本）：'+c['title']
   c['before_label']='历史 B：2026-10-23 审核版，最终通过片段'
   c['after_label']='最新 H：最终通过片段＋未通过的原因证据'
  target=next(r for r in v['rows'] if r['initial'] and r['pool']=='keep_candidate')
  after=cache[compare.sig(target)]
  d['comparison_cases'].insert(0,dict(id='adjacent_unchanged',title='相邻版本 L→H：本条所有切分起止、内容与音质结论一致',note='下表列出全部片段，这里播放其中一组相同片段。历史 B→H 的差异在后面单独标注。',before=[target],after=[after]))
  d.update(comparison_before_label='上一版 L：句首补接前',comparison_after_label='最新版 H：句首补接后',comparison_description='先展示相邻 L→H；后面明确标记“历史 B→H”的是跨多次规则变化的历史对照。未通过提案只用于解释原因，不是最终保留结果。',scope_description='本页明确区分相邻版本 L→H 与历史 B→H。B 是 10 月 23 日历史审核结果的展示编号，不是上一版 L。')
manifest['before_versions']+=new_versions
manifest['adjacent_comparison_summary']=reports
write(EXP/'adjacent_comparison.json',dict(pairs=reports,scope='uniform L-to-H on all seven sources',full_run_started=False))
with (EXP/'adjacent_intervals.csv').open('w',newline='') as f:
 writer=csv.DictWriter(f,fieldnames=list(csvrows[0]));writer.writeheader();writer.writerows(csvrows)
write(EXP/'review/manifest.json',manifest)
lines=['# 七来源统一比较：L → H','', 'L 是局部反馈＋句末优先版；H 在 L 上加短回应句首补接，失败则完整回退。均使用相同音质规则。时间单位为源视频秒，保留原始精度。','', '每条分别列初始候选与最终通过；不把未通过提案计作最终保留。完整逐段原文和状态也可下载 [CSV](adjacent_intervals.csv)。','']
for i,r in enumerate(reports,1):
 lines += [f'## {i}. {r["pair"]}','']
 for stage in r['stages']:
  lines += [f'### {stage["stage"]}：{stage["before_count"]}→{stage["after_count"]}，{stage["unchanged"]} 段起止完全一致','','| 起止变化 | L：起止秒、原文、音质状态 | H：起止秒、原文、音质状态 |','|---|---|---|']
  for g in stage['groups']:
   cells=['<br>'.join(f'{x["speaker"]} {x["start"]:.6f}–{x["end"]:.6f} · {x["text"]} · {x["pool"]}' for x in g[side]).replace('|','\\|') or '无' for side in ['before','after']]
   lines.append('| '+('不变' if g['unchanged'] else '变化')+' | '+' | '.join(cells)+' |')
  lines+=['']
(EXP/'ADJACENT_INTERVALS.md').write_text('\n'.join(lines)+'\n')
for r in reports:print(r['pair'],[(s['stage'],s['before_count'],s['after_count'],s['unchanged']) for s in r['stages']])
