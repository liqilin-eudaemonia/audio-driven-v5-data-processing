"""Export only two trial sources; exact media reuse and explicit three-version cases."""
import ctypes
assert ctypes.CDLL(None).prctl(41,1,0,0,0)==0
import copy,csv,json,sys
from pathlib import Path
EXP=Path(__file__).resolve().parent;ROOT=EXP.parents[1];SEVEN=ROOT/'experiments/pair_head_seven_20261029'
sys.path[:0]=[str(ROOT/'experiments/pair_boundary_real_audit_20261029'),str(ROOT/'experiments/pair_boundary_holdout3_20261029')]
import export_fixed_pairs as export
import compare
read=compare.read;write=export.write
PAIRS=['naturalistic_V02_S3610_I00000088','naturalistic_V02_S3579_I00000175']
keep=lambda view:[r for r in view['rows'] if r['id'] in view['terminal_ids'] and r['pool']=='keep_candidate']
terminal=lambda view:[r for r in view['rows'] if r['id'] in view['terminal_ids']]
initial=lambda view:[r for r in view['rows'] if r['initial']]
small=lambda r:dict(speaker='AB'[r['speaker_index']],start=r['start'],end=r['end'],text=r['text'],pool=r['pool'])
def groups(a,b):
 aa={compare.sig(r):r for r in a};bb={compare.sig(r):r for r in b}
 rows=[dict(start=k[1],unchanged=True,before=[small(aa[k])],after=[small(bb[k])]) for k in aa.keys()&bb.keys()]
 rows += [dict(start=g['start'],unchanged=False,before=[small(r) for r in g['before']],after=[small(r) for r in g['after']]) for g in compare.changes(a,b)]
 return sorted(rows,key=lambda x:x['start'])
def main():
 oldmanifest=read(SEVEN/'review/manifest.json');hsets={d['pair']:d for d in oldmanifest['pairs']}
 prodsets={d['pair']:d for d in read(SEVEN/'review/production_before/manifest.json')['pairs']}
 export.RESULTS=EXP/'results';export.OUT=EXP/'review/semantic_trial';export.MEDIA_DIR='semantic_trial';export.ID_PREFIX='J'
 out=[];stats=[];csvrows=[];fresh=0;reused=0
 for number,pair in enumerate(PAIRS,1):
  h=hsets[pair];prod=prodsets[pair];view=export.view(pair,number);result=read(EXP/'results'/pair/'result.json')
  originals={r['sample_id']:r for r in result['clips']+result['superseded_attempts']}
  published=read(export.PREVIEW/pair/'review.json');cache={compare.sig(r):r for v in published['fixed_pair_versions'] for r in v['rows']}
  evidence=export.PREVIEW/pair/'evidence/semantic_trial'
  if not evidence.is_symlink():evidence.symlink_to(export.OUT,target_is_directory=True)
  for r in view['rows']:
   actual=originals[r['sample_id']]
   for key in ['sentence_end_protection','soft_pause_rebalance','semantic_recovery_hold']:r[key]=actual.get(key)
   prev=cache.get(compare.sig(r))
   if prev:
    for key in ['media','poster','audio_urls','media_check']:r[key]=copy.deepcopy(prev[key])
    r['media_check']['reused_exact_interval']=True;reused+=1
   else:export.export(r);fresh+=1
   write(export.OUT/r['id']/'evidence.json',dict(view=r,full_qc=actual))
  cases=[]
  def one(rows,start):return next(r for r in rows if abs(r['start']-start)<1e-6)
  def overlap(rows,a,b):return [r for r in rows if min(r['end'],b)>max(r['start'],a)+1e-6]
  def add(key,title,a,b,production,before,after,note,stage='最终结果'):
   cases.append(dict(id=key,title=title,production=production,before=before,after=after,
    production_label='旧全量：最终通过',before_label='修复前 H：'+stage,after_label='试修 J：'+stage,note=note))
  if number==1:
   add('trial_long_sentence','先看：Mufasa movie theater 已恢复到同一段',25.16,35.498,
    [one(keep(prod),25.16)],overlap(initial(h),25.16,35.498),[one(initial(view),25.16)],
    '此组比较初始切分。试修恢复 25.160–35.498 秒完整长句，且原音质规则通过。前面的短语气词 Man（21.495–21.815）未成独立 pair，已明确记录；5 秒强制停顿阈值仍保留。','初始候选')
   for key,start in [('trial_he_is',140.652),('trial_he',148.43)]:
    add(key,'句末保护：不再把下一句开头接到上一句末尾',start,start+8,
     [one(keep(prod),start)],[one(keep(h),start)],[one(keep(view),start)],
     '三列均为最终音质通过片段。试修恢复到旧全量句末，保留完整结束语，不再追加孤立的 He’s / He。')
   add('trial_what','句末保护：去掉末尾未完的 What',50.715,58.098,
    overlap(keep(prod),50.715,58.098),[one(initial(h),50.715)],[one(initial(view),50.715)],
    '试修停在 Mufasa. 句末；What 仍记录为未成段尾词。旧全量此段还包含前面的 OK，三列实际起点不同，请按时间比较。','初始候选')
  else:
   for key,title,start,stop in [('trial_san_diego','已有修复保留：San Diego 与回应',5.743,11.7),('trial_during','已有修复保留：During 从句与主句同段',149.903,161.853),('trial_job','已有修复保留：末尾求职完整句',174.058,186.78)]:
    add(key,title,start,stop,overlap(keep(prod),start,stop),overlap(keep(h),start,stop),overlap(keep(view),start,stop),
     'H 与 J 的这组最终通过片段起止、原文保持一致；并列旧全量供检查已有修复。During 的末尾留白为零等原有边界提示仍保留。')
   add('trial_days','仍待听：完整的 That’s like maybe two or three days.',118.066,121.528,
    overlap(keep(prod),118.066,121.528),[one(initial(h),118.066)],[one(initial(view),118.066)],
    '本次没有改动此初始候选，仍保留 118.066–121.528 秒完整话语。音质模型的双轨相似疑点没有自动放行，请人工判断。','完整候选，待音质复核')
  for r in terminal(view):
   if not r.get('semantic_recovery_hold'):continue
   before=overlap(terminal(h),r['start'],r['end'])
   add(f'trial_hold_{r["id"]}','待人工复核：保留整段候选，不用不完整子段替代',r['start'],r['end'],
    overlap(keep(prod),r['start'],r['end']),before,[r],
    '右侧是待音质复核或规则排除的原候选，不计入自动通过。恢复提案未能同时满足句首、句末证据，因此本次保留整段供试听；未放宽音质规则。左、中列可能包含局部通过片段，不能把右侧时长当作新增通过时长。')
  stages=[]
  for label,select in [('初始切分',initial),('最终通过',keep)]:
   a,b=select(h),select(view);gs=groups(a,b)
   stages.append(dict(stage=label,before_count=len(a),after_count=len(b),unchanged=sum(g['unchanged'] for g in gs),groups=gs))
  for version,v in [('旧全量',prod),('修复前H',h),('试修J',view)]:
   for stage,select in [('初始候选',initial),('最终通过',keep),('全部终端结果',terminal)]:
    for r in select(v):csvrows.append(dict(pair=pair,version=version,stage=stage,id=r['id'],**small(r)))
  report=dict(pair=pair,old_full_kept=len(keep(prod)),h_kept=len(keep(h)),j_kept=len(keep(view)),initial_h=len(initial(h)),initial_j=len(initial(view)),initial_unchanged=stages[0]['unchanged'],old_seconds=prod['kept_seconds'],h_seconds=h['kept_seconds'],j_seconds=view['kept_seconds'],semantic_parent_holds=[dict(id=r['id'],start=r['start'],end=r['end'],text=r['text'],pool=r['pool']) for r in terminal(view) if r.get('semantic_recovery_hold')],qc_counts=result['counts'])
  stats.append(report)
  view.update(default_filter='all',holdout_sample=True,variant_title='两条试修 J：保护句末＋长句切点',variant_description='先切候选；保持原音质规则。完整候选与自动通过结果分开列出。',scope_description='仅试修 170 秒和 190 秒两条，按旧全量、修复前 H、试修 J 三列核对。每段播放实际 pair，无额外上下文；初始候选不保证每句语义完整，仍需试听。',
   seven_validation_note=f'本条 {view["initial_count"]} 个初始候选；局部恢复后 {len(keep(view))} 段自动通过，{len(terminal(view))-len(keep(view))} 段整段候选未自动通过。保留整段待复核不代表自动通过增加；边界相同的历史片段复用媒体。',
   comparison_title='旧全量 / 修复前 H / 试修 J：重点与全部恢复待复核片段',comparison_description='先试听重点，再用下方候选列表检查全部切分。每列标明比较阶段和实际音质状态，待复核候选未自动纳入通过集。',comparison_cases=cases,
   interval_comparison=dict(title='本次 H→J：全部起止时间与原文',description='先列初始切分，再列最终音质通过。旧全量参照放在三列视频中，完整三版本逐段表另有 CSV。',before_label='修复前 H',after_label='试修 J',stages=stages),trial_unresolved=result['diagnostic']['unresolved'])
  out.append(view)
 write(EXP/'review/manifest.json',dict(pairs=out,stats=stats,production_modified=False,full_run_started=False,media_new=fresh,media_reused=reused))
 write(EXP/'comparison.json',dict(pairs=stats,production_modified=False))
 with (EXP/'intervals.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(csvrows[0]));w.writeheader();w.writerows(csvrows)
 print('COMPLETE',json.dumps(stats,ensure_ascii=False),'media',fresh,reused,flush=True)
if __name__=='__main__':main()
