"""Append two audited source conversations to the existing continuous reviewer."""
import concurrent.futures
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[2]
EXP=Path(__file__).resolve().parent
OUT=ROOT/'data_previews/seamless_listen_full_source_recovery'
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from review_listen_full_source import render_source, classify_timeline, compact
from listen_local_recovery import seconds, subtract
from p3_listen.common import sha

def read(p):return json.loads(p.read_text())
def write(p,d):
    tmp=p.with_suffix(p.suffix+'.pending')
    tmp.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)

def prepare(number):
    evidence=read(EXP/'review'/f'case{number}_evidence.json')
    pair=evidence['pair'];d=read(EXP/(pair+'.json'));folder=OUT/pair;folder.mkdir(exist_ok=True)
    prodroot=ROOT/'data_corpora/seamless_listen_local_full'
    prod=read(prodroot/'results'/(pair+'.json'));audit=read(prodroot/'audit'/(pair+'.json'))
    prev=read(ROOT/'data_corpora/seamless_listen_v8_audio_final_full/results'/(pair+'.json'))
    context=SimpleNamespace(limit=d['limit'],sources=d['sources'],geometry=[dict(rotation_degrees=0)]*2,
                            shots=[],shots_by_track=[[],[]],regions=audit['regions'],source_gaps=audit['source_gaps'])
    panels,media=render_source(context,read(prodroot/'normalized_pairs'/(pair+'.json')),folder)
    final=[r for r in prod['clips'] if r['audio_final']['pool']=='keep_candidate']
    attempts=sorted(prod['clips']+prod['superseded_attempts'],key=lambda r:(r['start'],r['end']))
    rows=[]
    for row in attempts:
        view=compact(row);crops=[dict(status='unmeasured'),dict(status='unmeasured')]
        for role in ('speaker','listener'):
            crops[row[role+'_index']]=row.get('crop_proposals',{}).get(role,dict(status='unmeasured'))
        view['crops']=crops;rows.append(view)
    before=[(r['start'],r['end']) for r in prev['clips'] if r['audio_final']['pool']=='keep_candidate']
    after=[(r['start'],r['end']) for r in final]
    summary=dict(pair=pair,source_seconds=d['limit'],old_kept_clips=len(before),old_kept_seconds=seconds(before),
                 final_kept_clips=len(final),new_kept_seconds=seconds(after),
                 added_seconds=seconds(subtract(after,before)),removed_seconds=seconds(subtract(before,after)),
                 qc_attempts=len(attempts),training_admitted=False)
    finding=evidence['finding'];current=finding['current'][-1]
    alt=finding['alternative'][-2:] if number==1 else finding['alternative']
    if number==1:
        title='问题 1：贪心切分留下短尾，整句未进入任何生产片段'
        desc='生产片段在 184.661 秒结束。后一句 “So I’m looking for a new job right now.”（184.921–186.642 秒，共 9 个词）单独切分触发不足 2 秒，随后逐词跳过；代码不回退前一个切点。保持质量排除区间与声学函数不变，将前一段改为 174.058–176.179 和 176.179–186.780 秒，可完整保留这句话。'
        intervals=[dict(start=184.921,end=186.642,label='整句遗漏：So I’m looking for a new job right now.')]
        focus=183.0;reason='complete_unit_below_2s'
    else:
        title='问题 2：角色区间尾部上限挡住完整句尾与 listener 回应'
        desc='生产片段为 332.905–335.707 秒，停在 “…my car to heat”。上游角色区间只延至 336.228 秒，放不下 336.355 秒才结束的 “Oh, yeah.”；完整句尝试失败后，算法退到更早的词边界，speaker 的 “up.” 和 listener 的回应都遗漏。仅把结束上限延至 336.655 秒，同一声学函数得到 332.905–336.655 秒完整片段，边界警告为空。'
        intervals=[dict(start=335.787,end=335.928,label='speaker 句尾遗漏：up.'),
                   dict(start=336.115,end=336.355,label='listener 回应遗漏：Oh, yeah.')]
        focus=333.0;reason='no_room_for_complete_tail_before_next_voice'
    link=folder/'evidence'
    if not link.exists():link.symlink_to(EXP/'review',target_is_directory=True)
    comparisons=[]
    for c in evidence['clips']:
        comparisons.append(dict(c,media=f'{pair}/evidence/{c["file"]}'))
    issue=dict(id=f'case{number}',title=title,description=desc,start=current['start'],end=alt[-1]['end'],
               focus=focus,speaker_index=finding['speaker_index'],intervals=intervals,
               original=current,alternatives=alt,error=reason,comparisons=comparisons,
               context_media=f'{pair}/evidence/case{number}_context.mp4',
               evidence_url=f'{pair}/evidence/case{number}_evidence.json',
               raw_audio=[f'{pair}/evidence/case{number}_raw_{track}.wav' for track in 'AB'])
    closeups=EXP/'review/closeups.json'
    comparison=EXP/'review/before_after.json'
    if comparison.exists():
        issue['before_after']=read(comparison)[f'case{number}']
        for side in ('before','after'):
            row=issue['before_after'][side]
            row['media']=f'{pair}/evidence/{row["file"]}'
            row['poster']=f'{pair}/evidence/{row["poster"]}'
    if closeups.exists():
        short=read(closeups)[f'case{number}']
        issue['closeup']=dict(short,media=f'{pair}/evidence/{short["file"]}',
                             poster=f'{pair}/evidence/{short["poster"]}',
                             instruction='看右边：第 4.66 秒出现“当前 pair 已结束”后，继续听最后一句。' if number==1 else
                             '先看右边说话者的句尾，再看左边的短回应；第 4.71 秒是当前 pair 的结束点。')
    value=dict(pair=pair,summary=summary,media=pair+'/source.mp4',media_verification=media,panels=panels,
               old=[compact(r) for r in prev['clips']],attempts=rows,kept_ids=[r['sample_id'] for r in final],
               timeline=classify_timeline(context,prev['clips'],attempts,final,audit['diagnostic']),
               shots=[[],[]],scene_detection_applied=False,regions=context.regions,source_gaps=context.source_gaps,
               diagnostic=audit['diagnostic'],overview_crops=[],old_unresolved=[],
               transcripts=[[dict(start=s['start'],end=s['end'],text=s['text']) for s in p['segments']] for p in d['people']],
               issues=[issue],review_kind='production_boundary_reproduction',
               production_result_sha256=sha(prodroot/'results'/(pair+'.json')),
               alternative_status='诊断替代切法，未重新做音质准入；未替换生产数据或训练。')
    manual=EXP/'review/manual_review/manifest.json'
    if manual.exists():
        review_set=read(manual)
        value['manual_review']=[r for r in review_set['rows'] if r['pair']==pair]
        value['manual_review_revision']=review_set['revision']
    fixed=EXP/'review/fixed_pairs/manifest.json'
    if fixed.exists():
        value['fixed_pairs']=next(r for r in read(fixed)['pairs'] if r['pair']==pair)
    gap5=EXP/'review/gap5_pairs/manifest.json'
    if gap5.exists():
        value['fixed_pairs_previous']=value.get('fixed_pairs')
        value['fixed_pairs']=next(r for r in read(gap5)['pairs'] if r['pair']==pair)
    versions=[value.get('fixed_pairs'),value.get('fixed_pairs_previous')]
    localbc=EXP/'review/localbc_pairs/manifest.json'
    if localbc.exists():
        value['fixed_pairs_previous']=value.get('fixed_pairs')
        value['fixed_pairs']=next(r for r in read(localbc)['pairs'] if r['pair']==pair)
        versions.insert(0,value['fixed_pairs'])
    value['fixed_pair_versions']=[v for v in versions if v]
    write(folder/'review.json',value)
    print(json.dumps(dict(pair=pair,issues=1,seconds=d['limit']),ensure_ascii=False),flush=True)
    return dict(pair=pair,summary=summary,issue_title=title)

if __name__=='__main__':
    backup=EXP/'before_review_publish';backup.mkdir(exist_ok=True)
    for name in ['index.json','index.html','review.woff']:
        if not (backup/name).exists():shutil.copyfile(OUT/name,backup/name)
    new=[prepare(number) for number in [1,2]]
    index=read(OUT/'index.json');names={r['pair'] for r in new}
    index['pairs']=[r for r in index['pairs'] if r['pair'] not in names]+new
    index['boundary_reproduction_added']='2026-10-29'
    write(OUT/'index.json',index)
    from listen_review_head_attach import overlay
    overlay()
    html=(ROOT/'scripts/listen_full_source_review.html').read_text()
    banner='<main><div style="padding:12px 20px;background:#203650;color:#fff"><a href="/listen_size_review/" style="color:#a6dcff">打开最新全量抽样与人脸尺寸审核（含储备池，共 130 条）</a></div>'
    html=html.replace('<main>',banner,1)
    shutil.copyfile(ROOT/'scripts/listen_full_source_manual_review.js',OUT/'manual_review.js')
    shutil.copyfile(ROOT/'scripts/listen_full_source_fixed_pairs.js',OUT/'fixed_pairs.js')
    from fontTools import subset
    from fontTools.ttLib import TTFont
    font=TTFont('/tmp/v8_browser_libs/extracted/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',fontNumber=2)
    sub=subset.Subsetter()
    sub.populate(text=html+(OUT/'manual_review.js').read_text()+(OUT/'fixed_pairs.js').read_text()+(OUT/'index.json').read_text()+''.join(p.read_text() for p in OUT.glob('*/review.json')))
    sub.subset(font);font.flavor='woff';font.save(OUT/'review.woff.pending')
    (OUT/'review.woff.pending').replace(OUT/'review.woff')
    (OUT/'index.html.pending').write_text(html);(OUT/'index.html.pending').replace(OUT/'index.html')
    print('data published',flush=True)
