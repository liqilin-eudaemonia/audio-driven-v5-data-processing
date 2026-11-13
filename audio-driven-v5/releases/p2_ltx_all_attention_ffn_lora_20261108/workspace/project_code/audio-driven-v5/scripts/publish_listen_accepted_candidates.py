"""Publish an accepted candidate view over frozen audio-passed manifest parts."""
import datetime
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'data_corpora/seamless_listen_local_full'
OUT=BASE/'accepted_audio_candidates'
FEEDBACK=ROOT/'data_reviews/listener_onset_acceptance_20261024.json'


def read(path):return json.loads(path.read_text())
def write(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    pending=path.with_suffix(path.suffix+'.pending')
    pending.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n');pending.replace(path)


def add(a,b):
    if isinstance(a,dict):
        assert a.keys()==b.keys()
        return {k:add(a[k],b[k]) for k in a}
    if isinstance(a,list):
        assert len(a)==len(b)
        return [add(x,y) for x,y in zip(a,b)]
    assert type(a) in (int,float) and type(b) in (int,float)
    return a+b


def main():
    acceptance=read(FEEDBACK)
    assert acceptance['decision']=='include_audio_passed_onset_in_visual_candidates'
    summary=read(BASE/'screening_summary.json');original=read(BASE/'manifests/index.json')
    assert summary['final'] and summary['pending_pairs']==0 and original['export']['counts_verified']
    pools=['train/primary/keep_candidate','train/onset_reserve/keep_candidate']
    parts=[]
    for key in pools:
        filename=key.replace('/','_')+'.jsonl';path=BASE/'manifests'/filename
        assert path.is_file() and original['counts'][key]==summary['counts'][key]
        parts.append(dict(path='../manifests/'+filename,source_pool=key,
            clips=summary['counts'][key],hours=summary['hours'][key],bytes=path.stat().st_size,
            sha256=original['manifest_sha256'][key.replace('/','_')],
            hash_provenance='Recorded by the verified final export; source files not rewritten.'))
    counts=sum(p['clips'] for p in parts);hours=sum(p['hours'] for p in parts)
    index=dict(revision='audio-passed-onset-included-20261024',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        split='train',clips=counts,hours=hours,parts=parts,
        acceptance_record=str(FEEDBACK.relative_to(ROOT)),acceptance_sha256=hashlib.sha256(FEEDBACK.read_bytes()).hexdigest(),
        effective_candidate_pool='audio_passed_pending_visual',
        onset_activity_alone_withholds=False,source_admission_pool_is_provenance=True,
        audio_review_admitted=False,audio_reject_admitted=False,human_holds_admitted=False,
        visual_filter_applied=False,training_admitted=False,
        note='Read both parts exactly once. Existing segmentation/audio decisions and source onset flags remain unchanged. Dev/test stay separate.')
    for filename in ['face_size_distribution.json','crop_size_distribution.json']:
        raw=read(BASE/filename);merged=add(*(raw['pools'][p] for p in pools))
        assert merged['clips']==counts and abs(merged['seconds']/3600-hours)<1e-7
        for role in merged['roles'].values():
            field='side_seconds' if 'side_seconds' in role else 'median_short_side_seconds'
            assert abs(sum(role[field])+role['unmeasured_seconds']-merged['seconds'])<1e-5
        write(OUT/filename,dict(raw,pools={'train/audio_passed_pending_visual':merged},
            effective_policy_revision=index['revision'],source_pools=pools))
    write(OUT/'index.json',index)
    (OUT/'README.md').write_text(f'''# 已接受的音质候选：主池 + 起始活动池

共 **{counts:,} 段 / {hours:.6f} 小时**，等待人脸/裁框筛选。

入口 `index.json` 是由两份原始 JSONL 组成的统一索引；按 parts 的顺序读取，每份只读一次，路径相对本目录。没有复制或改写大清单。两份源清单分别为主池和音质通过的起始活动池；原 admission_pool 与 boundary_activity 保留作来源/活动标记，不能据此再次排除 onset_reserve。下游视觉筛选应使用此索引，而不只读取旧 primary 文件。

用户认可储备池抽样的总体可保留性，取消“起始活动单独阻止进入视觉候选”的条件；这不等于为全池建立逐条人工正例。原音质复核、拒绝、人工保留待定项均未放行。切分区间与音频不变；dev/test 独立，尚未执行视觉筛选、最终裁剪或训练准入。

两份尺寸分布 JSON 将两池的计数与时长直方图相加，未重新测量，也未实施尺寸阈值。原快照保留。
''')
    print(json.dumps(dict(index=str(OUT/'index.json'),clips=counts,hours=hours),ensure_ascii=False))


if __name__=='__main__':main()
