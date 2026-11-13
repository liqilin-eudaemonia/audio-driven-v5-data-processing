"""All old-production/H/J changes for the five remaining known sources."""
import ctypes
assert ctypes.CDLL(None).prctl(41,1,0,0,0)==0
import copy,csv,hashlib,json,sys
from pathlib import Path
EXP=Path(__file__).resolve().parent;ROOT=EXP.parents[1];SEVEN=EXP
sys.path[:0]=[str(ROOT/'experiments/pair_boundary_real_audit_20261029'),str(ROOT/'experiments/pair_boundary_holdout3_20261029')]
import compare,export_fixed_pairs as export
read=compare.read;write=export.write
PAIRS=read(EXP/'results/contract.json')['pairs']
keep=lambda d:[r for r in d['rows'] if r['id'] in d['terminal_ids'] and r['pool']=='keep_candidate']
initial=lambda d:[r for r in d['rows'] if r['initial']]
terminal=lambda d:[r for r in d['rows'] if r['id'] in d['terminal_ids']]
small=lambda r:dict(speaker='AB'[r['speaker_index']],start=r['start'],end=r['end'],text=r['text'],pool=r['pool'])
def overlap(rows,a,b):return sorted([r for r in rows if min(r['end'],b)>max(r['start'],a)+1e-6],key=lambda r:(r['start'],r['end']))
def two_groups(a,b):
 aa={compare.sig(r):r for r in a};bb={compare.sig(r):r for r in b}
 out=[dict(start=k[1],unchanged=True,before=[small(aa[k])],after=[small(bb[k])]) for k in aa.keys()&bb.keys()]
 out += [dict(start=g['start'],unchanged=False,before=[small(r) for r in g['before']],after=[small(r) for r in g['after']]) for g in compare.changes(a,b)]
 return sorted(out,key=lambda g:g['start'])
def three_changes(p,h,j):
 maps={side:{compare.sig(r):r for r in rows} for side,rows in [('production',p),('before',h),('after',j)]}
 common=set.intersection(*(set(v) for v in maps.values()))
 entries=sorted([(r['start'],r['end'],side,r) for side,rows in maps.items() for k,r in rows.items() if k not in common],key=lambda x:x[:2])
 groups=[]
 for a,b,side,r in entries:
  if not groups or a>groups[-1]['end']+.001:groups.append(dict(start=a,end=b,production=[],before=[],after=[]))
  group=groups[-1];group['end']=max(group['end'],b);group[side].append(r)
 return groups
def signature(case):return tuple(tuple((r['speaker_index'],r['start'],r['end'],r['pool']) for r in case[s]) for s in ['production','before','after'])
def main():
 hsets={d['pair']:d for d in read(SEVEN/'review/h_manifest.json')['pairs']};psets={d['pair']:d for d in read(SEVEN/'review/production_before/manifest.json')['pairs']}
 export.RESULTS=EXP/'results';export.OUT=EXP/'review/random5_j';export.MEDIA_DIR='random5_j';export.ID_PREFIX='J'
 pairs=[];reports=[];csvrows=[];newmedia=0;reuse=0
 for number,pair in enumerate(PAIRS,8):
  p,h=psets[pair],hsets[pair];j=export.view(pair,number);source=read(EXP/(pair+'.json'));result=read(EXP/'results'/pair/'result.json');hresult=read(SEVEN/'h'/pair/'result.json');prod=read(export.PROD/'results'/(pair+'.json'))
  saved=read(export.PREVIEW/pair/'review.json');cache={compare.sig(r):r for v in saved['fixed_pair_versions'] for r in v['rows']}
  originals={r['sample_id']:r for r in result['clips']+result['superseded_attempts']}
  link=export.PREVIEW/pair/'evidence/random5_j'
  if not link.is_symlink():link.symlink_to(export.OUT,target_is_directory=True)
  for row in j['rows']:
   actual=originals[row['sample_id']]
   for key in ['sentence_end_protection','soft_pause_rebalance','semantic_recovery_hold','head_attachment','head_attachment_fallback']:row[key]=actual.get(key)
   prev=cache.get(compare.sig(row))
   if prev:
    for key in ['media','poster','audio_urls','media_check']:row[key]=copy.deepcopy(prev[key])
    row['media_check']['reused_exact_interval']=True;reuse+=1
   else:export.export(row);newmedia+=1
   write(export.OUT/row['id']/'evidence.json',dict(view=row,full_qc=actual))
  qi=compare.evaluate(source,compare.roots(hresult),compare.roots(result))
  rawkeep=lambda d:[r for r in d['clips'] if r['audio_final']['pool']=='keep_candidate']
  qh=compare.evaluate(source,rawkeep(hresult),rawkeep(result));qp=compare.evaluate(source,rawkeep(prod),rawkeep(result))
  finalcases=[]
  for n,g in enumerate(three_changes(keep(p),keep(h),keep(j)),1):
   no_new=not g['after']
   note='三列均只列最终音质通过的实际片段。完整候选与音质疑点另列，不算自动通过。'
   if no_new:note+=' 本范围当前 J 暂无最终通过片段；请查看下方保留的原候选及待复核说明。'
   if g['before'] and g['after'] and any(r.get('sentence_end_protection') for r in g['after']):note+=' 试修保护已选句末，没有追加下一句开头；是否更完整请听原语音。'
   removed=[w for w in qp['previous_speaker_words_lost'] if compare.covered(w,g['production'])]
   gained=[w for w in qp['added_speaker_words'] if compare.covered(w,g['after'])]
   if removed:note+=' 相比旧全量，J 最终通过集未再覆盖：'+' '.join(w['text'] for w in removed)+'。请区分原候选仍待复核与实际未成段。'
   if gained:note+=' 相比旧全量，J 最终通过集新增覆盖：'+' '.join(w['text'] for w in gained)+'。'
   if {compare.sig(r) for r in g['production']}=={compare.sig(r) for r in g['after']}:note+=' 此范围旧全量与 J 的最终通过起止完全相同。'
   finalcases.append(dict(id=f'random5_final{n}',stage='final',start=g['start'],end=g['end'],title=f'最终通过范围对照 · {g["start"]:.3f}–{g["end"]:.3f} 秒',note=note,production=g['production'],before=g['before'],after=g['after'],production_label='旧全量：最终通过',before_label='修复前 H：最终通过',after_label='当前 J：最终通过'))
  initialcases=[]
  for n,g in enumerate(compare.changes(initial(h),initial(j)),1):
   missing=[w for w in qi['previous_speaker_words_lost'] if compare.covered(w,g['before'])]
   text=' '.join(w['text'] for w in missing)
   protected=bool(g['after']) and all(r.get('sentence_end_protection') for r in g['after'])
   caution=bool(missing) and not protected
   note='左列是旧全量在此范围的最终通过片段，中、右列比较 H→J 初始切分（音质状态仍标出）。起止时间均为原视频秒。'
   if missing:note+=' 原 H 覆盖、现在未进入 J 初始 pair 的词：'+text+'。词仍在源数据与未成段诊断中，没有计作已保留。'
   if protected:note+=' 这里保护原先已选的完整句末，移除的是后续句子的开头碎片；没有把这些词计作已保留。'
   if caution:note+=' 这里可能包含独立短句或有意义的回应，不能一概当作残尾删除，请重点试听。'
   if any(r.get('soft_pause_rebalance') for r in g['after']):note+=' 这里触发了短前缀后的长句切点调整。'
   case=dict(id=f'random5_initial{n}',stage='initial',start=g['start'],end=g['end'],priority=0 if caution else 1,title=('重点复核：短句／回应未再成段' if caution else '句末保护：不追加下一句残尾' if protected else '初始切分变化')+f' · {g["start"]:.3f}–{g["end"]:.3f} 秒',note=note,production=overlap(keep(p),g['start'],g['end']),before=g['before'],after=g['after'],production_label='旧全量：同范围最终通过',before_label='修复前 H：初始候选',after_label='当前 J：初始候选',missing_words=missing)
   # If the same three videos already form a final-kept comparison, keep one
   # case with both facts, rather than forcing the user to watch it twice.
   same=next((c for c in finalcases if signature(c)==signature(case)),None)
   if same and not caution:
    same['note']+=' 本组也是初始切分变化。'+(' 原 H 覆盖、现未成 J pair 的词：'+text+'。' if text else '')
    same['also_initial_change']=True;same['missing_words']=missing
   else:initialcases.append(case)
  holds=[]
  for row in terminal(j):
   if not row.get('semantic_recovery_hold'):continue
   holds.append(dict(id=f'random5_hold_{row["id"]}',stage='hold',start=row['start'],end=row['end'],title=f'整段待人工检查 · {row["start"]:.3f}–{row["end"]:.3f} 秒',note='右列是原候选，音质疑点未放行；本次没有用句首／句末不能确认完整的恢复子段替代它。中列为 H 的终端结果，可能包含音质通过的局部片段。此组不代表 J 自动通过覆盖增加。',production=overlap(keep(p),row['start'],row['end']),before=overlap(terminal(h),row['start'],row['end']),after=[row],production_label='旧全量：同范围最终通过',before_label='修复前 H：终端结果，按状态区分',after_label='当前 J：整段候选，未自动通过'))
  # Keep every unchanged-boundary candidate with a QC issue accessible as well.
  shown={r['id'] for c in initialcases+finalcases+holds for r in c['after']}
  remaining=[r for r in terminal(j) if r['pool']!='keep_candidate' and r['id'] not in shown]
  for row in remaining:
   holds.append(dict(id=f'random5_review_{row["id"]}',stage='hold',start=row['start'],end=row['end'],title=f'仍未自动通过 · {row["start"]:.3f}–{row["end"]:.3f} 秒',note='此候选的音质复核／排除结论仍保留；这里提供整段试听，不代表边界一定发生了变化。',production=overlap(keep(p),row['start'],row['end']),before=overlap(terminal(h),row['start'],row['end']),after=[row],production_label='旧全量：同范围最终通过',before_label='修复前 H：终端结果，按状态区分',after_label='当前 J：未自动通过候选'))
  cases=sorted(initialcases,key=lambda c:(c['priority'],c['start']))+finalcases+holds
  stages=[]
  for label,selector in [('初始切分',initial),('最终通过',keep)]:
   aa,bb=selector(h),selector(j);gs=two_groups(aa,bb)
   stages.append(dict(stage=label,before_count=len(aa),after_count=len(bb),unchanged=sum(g['unchanged'] for g in gs),groups=gs))
  for version,view in [('旧全量',p),('修复前H',h),('当前J',j)]:
   for stage,selector in [('初始候选',initial),('最终通过',keep),('全部终端结果',terminal)]:
    for row in selector(view):csvrows.append(dict(pair=pair,version=version,stage=stage,id=row['id'],**small(row)))
  report=dict(pair=pair,source_seconds=j['source_seconds'],initial_h=len(initial(h)),initial_j=len(initial(j)),initial_unchanged=qi['unchanged'],initial_changed_groups=qi['changed_groups'],old_full_kept=len(keep(p)),h_kept=len(keep(h)),j_kept=len(keep(j)),j_withheld=len(terminal(j))-len(keep(j)),production_unchanged=qp['unchanged'],h_unchanged=qh['unchanged'],initial_metrics=qi,vs_production=qp,vs_H=qh,qc_counts=result['counts'],semantic_parent_holds=[dict(id=r['id'],start=r['start'],end=r['end'],text=r['text'],pool=r['pool']) for r in terminal(j) if r.get('semantic_recovery_hold')],entry_case=cases[0]['id'] if cases else None,comparison_case_count=len(cases),old_seconds=p['kept_seconds'],h_seconds=h['kept_seconds'],j_seconds=j['kept_seconds'])
  reports.append(report)
  j.update(default_filter='all',holdout_sample=True,variant_title='J 规则：新增随机五条检查',variant_description='沿用已检查的同一份 J 代码；原音质规则不变。',scope_description='本条来自运行前固定的五条新随机来源，seed=20261031。旧全量 P、修复前 H、当前 J 并列；初始候选与最终通过分开比较，待复核候选没有自动放行。',seven_validation_note=f'初始切分 H→J：{len(initial(h))}→{len(initial(j))} 段，{qi["unchanged"]} 段起止完全一致。最终通过：旧全量 {len(keep(p))}、H {len(keep(h))}、J {len(keep(j))} 段；J 另有 {len(terminal(j))-len(keep(j))} 段未自动通过。所有具体起止与原文在表中，未成段词明确标注。',comparison_title='旧全量 / 修复前 H / 当前 J：所有变化与待复核候选',comparison_description='先列需要重点听的初始切分变化，再列三版本最终通过范围的全部差异，最后列完整待复核／排除候选。每组的阶段和状态均单独标明，播放实际 pair，无额外上下文。',comparison_cases=cases,interval_comparison=dict(title='H→J 全部起止时间与原文（含不变片段）',description='分初始切分与最终音质通过两层；旧全量作为三列视频的固定基线。完整三版本逐段时间另存 CSV。',before_label='修复前 H',after_label='当前 J',stages=stages),trial_unresolved=result['diagnostic']['unresolved'])
  pairs.append(j)
  print(pair,'initial',report['initial_h'],'->',report['initial_j'],'same',qi['unchanged'],'kept P/H/J',report['old_full_kept'],report['h_kept'],report['j_kept'],'cases',len(cases),flush=True)
 write(EXP/'review/manifest.json',dict(pairs=pairs,stats=reports,production_modified=False,full_run_started=False,media_new=newmedia,media_reused=reuse))
 write(EXP/'comparison.json',dict(pairs=reports,production_modified=False,full_run_started=False))
 with (EXP/'intervals.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(csvrows[0]));w.writeheader();w.writerows(csvrows)
 print('COMPLETE media new',newmedia,'reused',reuse,flush=True)
if __name__=='__main__':main()
