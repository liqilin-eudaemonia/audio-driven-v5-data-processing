#!/usr/bin/env python3
"""Apply explicit user OR/failure decisions to this measured batch, without deleting media."""
import json,hashlib,datetime,importlib.util
from pathlib import Path
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1]
WEB=ROOT/'data_previews/visual_quality_holdout120_20261030';VERSION=OUT/'policy_v2'


def read(p):return json.loads(Path(p).read_text())
def save(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    VERSION.mkdir(exist_ok=True)
    snapshot=VERSION/'previous_web_data.json'
    if not snapshot.exists():snapshot.write_bytes((WEB/'data.json').read_bytes())
    old=read(snapshot);selection=read(OUT/'selection.json')
    by_id={r['sample_id']:r for r in selection['rows']}
    # These are the two previously presented failures explicitly rejected by the user.
    # Do not invent a global detection-coverage threshold: old accepted clips include brief misses.
    failure_ids={r['id'] for r in old['results'] if r['focus']['status']=='measurement_incomplete'}
    sideways_ids={r['id'] for r in old['results'] if r['ordinal']==52}
    policy={'schema':'visual-quality-batch-policy-v2','recorded_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'user_instruction':'低清晰度持续时长长、累计占比多都可以去掉；横转画面、人脸检测失败的都应舍弃。',
            'scope':'these 120 measured pairs only; no full-corpus launch or source deletion',
            'clarity_rule':{'operator':'OR','longest_low_s_at_least':2.,'cumulative_low_fraction_at_least':.25,'underlying_measurement':'unchanged .45/.70 relative thresholds and 5 fps sampling'},
            'face_failure_rule':{'explicitly_reviewed_failure_ids':sorted(failure_ids),'interpretation':'the two incomplete-measurement cases presented in this batch; brief low-head/hand-occlusion misses remain allowed; not a newly calibrated corpus-wide 80% gate'},
            'sideways_ids':sorted(sideways_ids),'previous_protocol_hash':old['protocol']['hash'],
            'source_snapshot_sha256':sha(snapshot),'runner_sha256':sha(__file__),'full_corpus_applied':False}
    decisions=[];excluded=[];kept=[]
    for r in old['results']:
        q=r['focus'];reasons=[];labels=[]
        if q['longest_low_s']>=2-1e-9:reasons.append('continuous_low_clarity_at_least_2s');labels.append('连续低清晰度达到 2 秒')
        if q['low_fraction']>=.25-1e-9:reasons.append('cumulative_low_clarity_at_least_25pct');labels.append('累计低清晰度达到 25%')
        if r['id'] in failure_ids:reasons.append('user_rejected_clip_level_face_measurement_failure');labels.append('片段级人脸检测／测量失败')
        if r['id'] in sideways_ids:reasons.append('user_rejected_sideways_video');labels.append('横转画面')
        decision={'sample_id':r['id'],'ordinal':r['ordinal'],'decision':'exclude' if reasons else 'retain',
                  'reasons':reasons,'reason_labels':labels,'longest_low_s':q['longest_low_s'],
                  'low_fraction':q['low_fraction'],'measured_fraction':q['measured_fraction'],
                  'scope':'batch quality selection, not final training admission','full_corpus_applied':False}
        r['quality_decision_v2']=decision;decisions.append(decision)
        if reasons:
            excluded.append({'decision':decision,'row':by_id[r['id']]})
            r['review_reasons']=['按用户新规则排除：'+'、'.join(labels)]+[x for x in r['review_reasons'] if not x.startswith('接近')]
        else:kept.append(by_id[r['id']])
    # Newly excluded short clip N039 was not in the previous rendered selection.
    missing=[r for r in old['results'] if r['quality_decision_v2']['decision']=='exclude' and not r['review_media']]
    if missing:
        spec=importlib.util.spec_from_file_location('holdout_review_renderer',OUT/'build_review.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        for r in missing:r['review_media']=module.render(r)
    ranked=[r for r in old['results'] if r['quality_decision_v2']['decision']=='exclude']
    ranks={r['id'] for r in ranked}
    old['review_order']=[r['id'] for r in ranked]+[sid for sid in old['review_order'] if sid not in ranks]
    old['policy_v2']=policy
    old['summary']['policy_v2']={'excluded':len(excluded),'retained':len(kept),'excluded_ordinals':[r['decision']['ordinal'] for r in excluded],
                                  'clarity_excluded':sum(any(x.startswith(('continuous_','cumulative_')) for x in r['decision']['reasons']) for r in excluded),
                                  'face_failure_excluded':len(failure_ids),'sideways_excluded':len(sideways_ids),
                                  'counts_overlap':True,'full_corpus_applied':False}
    old['summary']['rendered_reviews']=sum(bool(r['review_media']) for r in old['results'])
    assert len(kept)+len(excluded)==120
    assert {r['sample_id'] for r in kept}.isdisjoint({r['row']['sample_id'] for r in excluded})
    assert {r['sample_id'] for r in kept}|{r['row']['sample_id'] for r in excluded}==set(by_id)
    for item in selection['jobs']:assert sha(item['path'])==item['sha256']
    for r in old['results']:assert sha(r['source']['source_result'])==r['source']['source_result_sha256']
    save(VERSION/'policy.json',policy);save(VERSION/'decisions.json',decisions);save(VERSION/'summary.json',old['summary']['policy_v2'])
    (VERSION/'retained_rows.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in kept))
    (VERSION/'excluded_rows.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in excluded))
    save(VERSION/'review_data.json',old);save(WEB/'data.json',old)
    save(VERSION/'partition_check.json',{'status':'passed','input':120,'retained':len(kept),'excluded':len(excluded),'disjoint_complete_partition':True,'source_and_job_hashes_unchanged':True,'all_excluded_have_review_media':all(r['review_media'] for r in ranked)})
    print(json.dumps(old['summary']['policy_v2'],ensure_ascii=False))


if __name__=='__main__':main()
