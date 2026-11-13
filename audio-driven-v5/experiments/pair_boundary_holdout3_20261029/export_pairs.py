"""Export both full variants, with all changed components and no cherry picking."""
import ctypes
assert ctypes.CDLL(None,use_errno=True).prctl(41,1,0,0,0)==0
import concurrent.futures,os,shutil,sys,time
from pathlib import Path
EXP=Path(__file__).resolve().parent;ROOT=EXP.parents[1]
sys.path[:0]=[str(EXP),str(ROOT/'experiments/pair_boundary_real_audit_20261029')]
import export_fixed_pairs as export
import compare
from p3_listen.common import sha
read=compare.read;write=export.write
PAIRS=[r['pair'] for r in read(EXP/'selection.json')['items']]

def export_ready(row):
 deadline=time.monotonic()+1800
 while not (EXP/(row['pair']+'.media.json')).exists():
  if time.monotonic()>deadline:raise RuntimeError('Source media render not complete: '+row['pair'])
  time.sleep(5)
 export.export(row)

def main():
 versions={};cache={}
 for variant,prefix in [('gap5','G'),('localbc','L')]:
  export.RESULTS=EXP/variant;export.OUT=EXP/'review'/variant;export.MEDIA_DIR=variant;export.ID_PREFIX=prefix
  pairs=[export.view(p,i+3) for i,p in enumerate(PAIRS)]
  for d in pairs:
   d.update(holdout_sample=True,review_scope='three randomly selected new sources; fixed seed 20261030',scope_description='本轮为预先随机抽取的三条新来源。G 为上一版 5 秒规则，L 为当前局部反馈分类＋句末优先规则；完整 pair 无额外上下文。下方保留历史生产时间线。',variant_title='5 秒＋局部 backchannel／句末优先' if variant=='localbc' else '上一版 5 秒规则',variant_description='随机样本外验证：展示全部差异，包括潜在退步；音质和时长规则不变。')
   result=read(EXP/variant/d['pair']/'result.json');d['local_feedback_audit']=result['local_feedback_audit']
   for r in d['rows']:
    key=(d['pair'],r['speaker_index'],r['start'],r['end'])
    folder=export.OUT/r['id']
    if key in cache:
     for name in ['pair.mp4','A.wav','B.wav']:
      dst=folder/name
      if not dst.exists():
       try:os.link(cache[key]/name,dst)
       except OSError:shutil.copyfile(cache[key]/name,dst)
    r['holdout_sample']=True
  rows=[r for d in pairs for r in d['rows']]
  with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(export_ready,rows))
  for d in pairs:
   for r in d['rows']:cache[(d['pair'],r['speaker_index'],r['start'],r['end'])]=export.OUT/r['id']
  versions[variant]=pairs
  write(export.OUT/'manifest.json',dict(pairs=pairs,exact_pair_playback=True,production_modified=False))
 comparison=read(EXP/'comparison.json')
 for old,new,report in zip(versions['gap5'],versions['localbc'],comparison['pairs']):
  before=[r for r in old['rows'] if r['initial']];after=[r for r in new['rows'] if r['initial']]
  new['change_summary']=report['initial'];new['comparison_cases']=[]
  groups=compare.changes(before,after)
  if not groups:
   r=before[len(before)//2];a=next(x for x in after if compare.sig(x)==compare.sig(r))
   new['comparison_cases'].append(dict(id='unchanged',title='本条全部初始 pair 边界相同：中间一段对照',note=f'本条 {len(before)} 段初始候选的新旧说话角色、起止时间全部一致。下方只展示同一段的左右副本供试听，其余候选可在列表中查看。',before=[r],after=[a]))
  for i,g in enumerate(groups,1):
   notes={1:('句末优先：2 段变为 3 段','旧版把 “Like,” 留在前段末尾；新版把它与后面的句子放在一起，并在真正句末切分。词覆盖保留，但中间一段尾部留白为 0，语义更整齐不等于声学边界已确认。'),2:('尾部延长 31 毫秒，文字不变','同一片段只调整尾部留白，speaker 词覆盖无变化。仍保留原有不完整句及尾部余量提示。'),3:('发现退步：句首 Okay. 丢失','重点复核：B 的 Okay. 在源时间 213.343–213.963 秒。后面的 Well 在 213.983 秒开始，仅隔 20 毫秒。新版把 Okay 独立归为反馈后，未纳入前一段，也未纳入随后由 B 主讲的片段。旧版保留，新版遗漏；这是新规则的衔接缺口，不应算作改进。'),4:('补回句首 it’s，仍是句子片段','新版从 359.759 秒开始，比旧版提前 341 毫秒，补回 359.819–359.959 秒的 it’s。原始词对齐和不完整句提示仍保留；新起点进入对方 VAD 活动区（359.680–359.950 秒），仍需试听两人重叠声音，不能仅凭补回单词认定更好。'),5:('起点提前 38 毫秒，文字不变','同一段文字和结束时间不变，仅起点留白变化；仍有原有边界提示。')}
   title,note=notes[i]
   new['comparison_cases'].append(dict(id=f'change{i}',title=title,note=note,before=g['before'],after=g['after'],source_start=g['start'],source_end=g['end']))
  # Put the regression first while keeping the original chronological IDs.
  new['comparison_cases'].sort(key=lambda c:0 if c['id']=='change3' else 1)
 for variant,pairs in versions.items():
  write(EXP/'review'/variant/'manifest.json',dict(pairs=pairs,exact_pair_playback=True,production_modified=False,screening_rules_changed=False))
 selection=read(EXP/'selection.json')
 assert all(sha(export.PROD/'results'/(p+'.json'))==h for p,h in selection['production_results'].items())
 print('EXPORTED',sum(len(d['rows']) for ds in versions.values() for d in ds),flush=True)
if __name__=='__main__':main()
