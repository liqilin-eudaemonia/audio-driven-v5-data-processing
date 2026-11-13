"""Export exact new candidates and before/after comparisons in the old HTML."""
import collections
import concurrent.futures
import os
import shutil
import export_fixed_pairs as export
from p3_listen.common import sha

RESULTS=export.EXP/'localbc_pairs'
OUT=export.EXP/'review/localbc_pairs'

def main():
    export.RESULTS=RESULTS;export.OUT=OUT;export.MEDIA_DIR='localbc_pairs';export.ID_PREFIX='L'
    pairs=[export.view(p,i+1) for i,p in enumerate(export.PAIRS)]
    prior=export.read(export.EXP/'review/gap5_pairs/manifest.json')
    old_pairs={d['pair']:d for d in prior['pairs']}
    for d in pairs:
        old=old_pairs[d['pair']];r=export.read(RESULTS/d['pair']/'result.json')
        d['variant_title']='5 秒＋局部 backchannel／句末优先'
        d['variant_description']='局部发声独立分类、真正句末优先；音质规则、安全余量和 2～11.95 秒约束不变。'
        d['local_feedback_audit']=r['local_feedback_audit']
        sig=lambda x:(x['speaker_index'],x['start'],x['end'])
        old_roots={sig(x):x for x in old['rows'] if x['initial']}
        new_roots={sig(x):x for x in d['rows'] if x['initial']}
        d['change_summary']=dict(old_count=len(old_roots),new_count=len(new_roots),
            unchanged=len(old_roots.keys()&new_roots.keys()),
            previous_kept_seconds=old['kept_seconds'],current_kept_seconds=d['kept_seconds'])
        d['comparison_cases']=[]
        if d['source_seconds']<200:
            definitions=[('local_feedback','独立 yeah 作为 backchannel 保留', ['G1-01'],5,12,
                '旧版在 San 后结束；新片段保留 San Diego、对方独立 yeah，以及说话者随后的一声 Okay。只改变段内反馈分类，未取消其他实质发言的阻断。'),
                ('sentence_priority','从句与后面的主句合为一个候选',['G1-16','G1-17'],149,163,
                '新片段为 11.95 秒，保留从 During 到 marry her 的整句。但句尾额外留白为 0，仍有尾部余量不足／未找到安静切口的提示，请重点试听结尾。')]
        else:
            definitions=[('sentence_priority_2a','句末优先：第一组相邻切点变化',['G2-09','G2-10'],89,98,
                '同一条句末排序规则带来的变化；未针对频繁打断的 G2-27 增加特殊规则。'),
                ('sentence_priority_2b','句末优先：第二组相邻切点变化',['G2-14','G2-15'],130,144,
                '优先尝试真正句末，重新分配相邻候选边界；请对照两侧试听。')]
        for key,title,ids,a,b,note in definitions:
            before=[x for x in old['rows'] if x['id'] in ids]
            after=[x for x in d['rows'] if x['initial'] and a<=x['start'] and x['end']<=b]
            assert before and after
            d['comparison_cases'].append(dict(id=key,title=title,note=note,before=before,after=after))
        # Exact old A/V intervals can share their immutable media; QC evidence
        # and candidate metadata remain new independent files.
        cache={sig(x):x for x in old['rows']}
        for row in d['rows']:
            prev=cache.get(sig(row))
            if not prev:continue
            source=export.PREVIEW/prev['media'];target=OUT/row['id']
            for name,src in [('pair.mp4',source),('A.wav',export.PREVIEW/prev['audio_urls']['A']),('B.wav',export.PREVIEW/prev['audio_urls']['B'])]:
                dst=target/name
                if not dst.exists():
                    try:os.link(src,dst)
                    except OSError:shutil.copyfile(src,dst)
    rows=[r for d in pairs for r in d['rows']]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(export.export,rows))
    contract=export.read(RESULTS/'contract.json')
    assert all(sha(export.PROD/'results'/(p+'.json'))==h for p,h in contract['production_results'].items())
    export.write(OUT/'manifest.json',dict(pairs=pairs,total_initial=sum(d['initial_count'] for d in pairs),
        total_withheld_initial=sum(r['initial'] and r['pool']!='keep_candidate' for r in rows),
        exact_pair_playback=True,production_modified=False,screening_rules_changed=False))
    print('COMPLETE',len(rows),flush=True)

if __name__=='__main__':main()
