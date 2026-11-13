"""Publish exact candidates; reuse immutable, previously decoded media by interval."""
import ctypes
assert ctypes.CDLL(None).prctl(41,1,0,0,0)==0
import concurrent.futures,os,shutil,sys
from pathlib import Path
EXP=Path(__file__).resolve().parent;ROOT=EXP.parents[1]
OLD=ROOT/'experiments/pair_boundary_real_audit_20261029';HOLD=ROOT/'experiments/pair_boundary_holdout3_20261029'
sys.path[:0]=[str(OLD),str(HOLD)]
import export_fixed_pairs as export
import compare
from p3_listen.common import sha
read=compare.read;write=export.write

def main():
 old_sets=read(OLD/'review/localbc_pairs/manifest.json')['pairs']+read(HOLD/'review/localbc/manifest.json')['pairs']
 cache={}
 for path in [OLD/'review/fixed_pairs/manifest.json',OLD/'review/gap5_pairs/manifest.json',OLD/'review/localbc_pairs/manifest.json',HOLD/'review/gap5/manifest.json',HOLD/'review/localbc/manifest.json']:
  for d in read(path)['pairs']:
   for r in d['rows']:cache[(d['pair'],*compare.sig(r))]=r
 export.RESULTS=EXP/'results';export.OUT=EXP/'review/head_attach';export.MEDIA_DIR='head_attach';export.ID_PREFIX='H'
 pairs=[export.view(d['pair'],i+1) for i,d in enumerate(old_sets)]
 reports=[];fresh=[]
 for i,(old,new) in enumerate(zip(old_sets,pairs)):
  pair=new['pair'];result=read(EXP/'results'/pair/'result.json');old_result=read((OLD/'localbc_pairs' if i<2 else HOLD/'localbc')/pair/'result.json');source=read(EXP/(pair+'.json'))
  initial=compare.evaluate(source,compare.roots(old_result),compare.roots(result))
  kept=lambda d:[r for r in d['clips'] if r['audio_final']['pool']=='keep_candidate']
  report=dict(pair=pair,initial=initial,kept=compare.evaluate(source,kept(old_result),kept(result)),old_kept_seconds=old_result['kept_seconds'],new_kept_seconds=result['kept_seconds'],head_audit=result['diagnostic'].get('head_attachment',[]),quality_fallback=result['diagnostic'].get('head_quality_fallback',[]),qc_counts=result['counts'])
  reports.append(report)
  new.update(holdout_sample=old.get('holdout_sample',False),review_scope='five-source uncovered-feedback head-attachment trial',scope_description='本轮复跑现有五条来源。只尝试补接未被覆盖、且在 0.3 秒内接着由同一人正式发言的短回应；补接未通过音质规则时恢复原完整候选。',variant_title='5 秒＋局部反馈／句末优先＋句首补接',variant_description='补接通过音质检查才保留；终点不移动，失败则恢复原 pair。',comparison_before_label='补接前（L 编号）',comparison_after_label='补接试验（H 编号，注意规则结论）',comparison_description='左侧为上一版 L，右侧为本次 H。播放范围均为实际候选，无额外上下文。失败扩展与最终回退片段分别展示。',comparison_cases=[],change_summary=initial)
  actual={r['sample_id']:r for r in result['clips']+result['superseded_attempts']}
  for r in new['rows']:
   r['head_attachment']=actual[r['sample_id']].get('head_attachment')
   r['head_attachment_fallback']=actual[r['sample_id']].get('head_attachment_fallback')
   prev=cache.get((pair,*compare.sig(r)))
   if prev:
    target=export.OUT/r['id']
    paths={'pair.mp4':prev['media'],'poster.jpg':prev['poster'],'A.wav':prev['audio_urls']['A'],'B.wav':prev['audio_urls']['B']}
    for name,url in paths.items():
     src=export.PREVIEW/url;dst=target/name
     if not dst.exists():
      try:os.link(src,dst)
      except OSError:shutil.copyfile(src,dst)
    r['media_check']=dict(prev['media_check'],reused_exact_interval=True,source_media=prev['media'])
   else:fresh.append(r)
  old_by_sample={r['sample_id']:r for r in old['rows']}
  for r in new['rows']:
   a=r.get('head_attachment')
   if not a:continue
   before=old_by_sample[a['previous_sample_id']]
   children=[x for x in new['rows'] if x['parent_id']==r['id']]
   word=a['feedback_text'];key={'Okay.':'okay_head','Yeah.':'yeah_head','Right.':'right_head'}[word]
   passed=r['pool']=='keep_candidate'
   note=(f'补回 {word}，与随后同一人正式发言相隔 {a["gap_seconds"]:.3f} 秒。片段终点不变，原音质规则通过。' if passed else
     'Right. 的扩展候选触发环境声模型疑点，未自动采用；下方另列回退后的完整原片段，起止时间与 L 版相同且规则通过，原来保留的内容没有因此丢失。模型疑点不等于人工确认存在噪声。')
   new['comparison_cases'].append(dict(id=key,title=f'{word} 已补回并通过规则' if passed else 'Right. 补接未通过：恢复原完整 pair',note=note,before=[before],after=[r]+children))
  if not new['comparison_cases']:
   targets=([r for r in new['rows'] if r['initial'] and ('San Diego?' in r['text'] or r['text'].startswith('During the time'))] if i==0 else [r for r in new['rows'] if r['initial'] and r['start']<=335.787 and r['end']>=336.355] if i==1 else [])
   if not targets:targets=[next(r for r in new['rows'] if r['id']==new['initial_ids'][len(new['initial_ids'])//2])]
   for j,r in enumerate(targets):
    before=old_by_sample[r['sample_id']]
    new['comparison_cases'].append(dict(id=f'unchanged{j+1}',title='本条初始边界全部不变：既有片段对照',note=f'本条 {new["initial_count"]} 段初始候选边界均与 L 版一致。之前的语义和声学提示仍然保留；During 整句的末尾留白仍为 0。' if i==0 else f'本条 {new["initial_count"]} 段初始候选边界均与 L 版一致。',before=[before],after=[r]))
  new['comparison_cases'].sort(key=lambda c:0 if c['id']=='okay_head' else 1 if c['id']=='yeah_head' else 2)
 with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:list(pool.map(export.export,fresh))
 for folder in [OLD/'review',HOLD/'review']:
  link=folder/'head_attach'
  if not link.exists():link.symlink_to(export.OUT,target_is_directory=True)
 write(export.OUT/'manifest.json',dict(pairs=pairs,production_modified=False,screening_rules_changed=False,media_new=len(fresh),media_reused=sum(len(d['rows']) for d in pairs)-len(fresh)))
 write(EXP/'comparison.json',dict(pairs=reports,production_modified=False))
 assert all(not r['kept']['previous_speaker_words_lost'] for r in reports)
 contract=read(EXP/'results/contract.json')
 assert all(sha(ROOT/p)==h for p,h in contract['frozen_code_sha256'].items())
 assert all(sha(export.PROD/'results'/f'{p}.json')==h for p,h in contract['production_results'].items())
 print('EXPORTED',sum(len(d['rows']) for d in pairs),'new media',len(fresh),flush=True)
if __name__=='__main__':main()
