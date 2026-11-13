"""Bounded independent QC2 planning, screening and publishing. No training."""
import argparse,json,random,time,shutil
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import preview_listen_v6 as preview
from p3_listen.common import read,write,sha,digest
from screen_listen_qc_sample import metadata,annotations,read_wave,MODEL
from listen_audio_quality import EventModel,assess
from listen_audio_quality_v2 import POLICY,retriage,row_decision
ROOT=preview.ROOT;OUT=ROOT/'data_previews/seamless_listen_qc2_review'
EXP=ROOT/'experiments/p3_listen_qc2_review';SEED=2026102207


def freeze():
    EXP.mkdir(parents=True,exist_ok=True)
    contract=dict(seed=SEED,policy=POLICY,created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        model=read(MODEL/'provenance.json'),code_sha256={f:sha(ROOT/'scripts'/f) for f in
        ['listen_audio_quality_v2.py','listen_audio_quality.py','run_listen_qc2_review.py','preview_listen_v6.py','build_seamless_rule_turns.py']})
    write(EXP/'frozen_contract.json',contract)


def plan():
    freeze();preview.OUT=OUT
    catalog=read(preview.OLD/'catalog.json');lookup={r['pair']:r for r in catalog['train']+catalog['dev']}
    examined=set();groups=set()
    for dirname in ['seamless_listen_v6_review','seamless_listen_qc_review']:
        p=ROOT/'data_previews'/dirname
        examined.update(f.stem for f in (p/'audit').glob('*.json'))
        if (p/'samples.json').exists():
            for r in read(p/'samples.json')['samples']:examined.add(r['pair']);groups.add(r['group_id'])
        if (p/'random_selection.json').exists():
            for r in read(p/'random_selection.json')['candidates']:examined.add(r['pair']);groups.add(r['group_id'])
    groups.update(lookup[p]['group_id'] for p in examined if p in lookup)
    pool=list(read(preview.OLD/'selected.json')['pairs']);random.Random(SEED).shuffle(pool)
    selected=[];seen=set(groups)
    for item in pool:
        rec=lookup[item['pair']]
        if rec['pair'] in examined or rec['group_id'] in seen:continue
        norm=preview.OLD/'normalized_pairs'/(rec['pair']+'.json')
        if not norm.exists() or not read(norm)['summary']['ok']:continue
        old=read(item['path'])
        if not any(c['crop_boxes']['listener']['size']>=512 and not c['crop_boxes']['listener']['wide_crop'] for c in old.get('clips',[])):continue
        selected.append(rec);seen.add(rec['group_id'])
        if len(selected)>=48:break
    assert not ({r['group_id'] for r in selected}&groups)
    write(OUT/'random_selection.json',dict(seed=SEED,candidates=selected,excluded_previous_pairs=sorted(examined),excluded_previous_groups=sorted(groups),scope='Already normalized train/dev with eligible original crop; disjoint groups from ALL previously inspected pairs.'))
    print('Selected independent groups',len(selected),flush=True)
    with ThreadPoolExecutor(max_workers=6) as pool:
        audits=list(pool.map(partial(preview.inspect_pair,defer_audio_uncertainty=True),selected))
    rows=[]
    for result in audits:
        clips=result['clips'];random.Random(SEED+int(digest(result['pair'])[:8],16)).shuffle(clips)
        rows.extend(clips[:4])
    write(OUT/'planned_candidates.json',dict(seed=SEED,pairs=len(selected),rows=rows))
    print('Audio candidates',len(rows),flush=True)


def regression():
    # Cached scores are unchanged; no human label overrides to automatic decisions here.
    rows=[]
    previous=read(ROOT/'data_previews/seamless_listen_qc_review/samples.json')['samples']
    for r in previous:
        if not r.get('human_review'):continue
        c=retriage(r['audio_qc']['listener'])
        rows.append(dict(sample_id=r['sample_id'],human_label=r['human_review']['decision'],automatic_status=c['status'],check=c))
    known=read(ROOT/'experiments/p3_listen_qc_review/known_noise_regression.json')['rows']
    for r in known:
        c=retriage(r['automatic_result']);rows.append(dict(sample_id=r['sample_id'],human_label='reject_noise',automatic_status=c['status'],check=c))
    write(EXP/'known_examples_regression.json',dict(rows=rows,note='Development examples only; cached AST scores, no manual overrides. Remaining review/false passes are reported, not tuned away.'))
    print('Known-example statuses',[(r['human_label'],r['automatic_status']) for r in rows],flush=True)


def screen():
    contract=read(EXP/'frozen_contract.json')
    for f,h in contract['code_sha256'].items():
        assert sha(ROOT/'scripts'/f)==h, 'Frozen code changed: '+f
    regression();model=EventModel(MODEL,batch_size=16)
    planned=read(OUT/'planned_candidates.json');results=[]
    for i,row in enumerate(planned['rows']):
        norm=read(preview.OLD/'normalized_pairs'/(row['pair']+'.json'));checks={}
        for role in ('listener','speaker'):
            source=row['sources'][role];wave=read_wave(norm['tracks'][row[role+'_index']]['path'],row['start'],row['duration_s'])
            vad,words=annotations(metadata(source['archive'],source['member']),row['start'],row['end'])
            checks[role]=retriage(assess(wave,48000,vad,words,model,role=role))
        result=dict(row,sample_id=row['sample_id'].replace('_v6_','_qc2_'),audio_qc=checks,screen_decision=row_decision(checks,row['listener_screen']))
        result['listener_human_evidence']=any(e['decision']['human_score']>=POLICY['human_score'] for e in checks['listener']['events'])
        results.append(result);write(OUT/'audio_checks'/(result['sample_id']+'.json'),result)
        print(json.dumps(dict(index=i+1,total=len(planned['rows']),sample=result['sample_id'],decision=result['screen_decision'],listener_events=len(checks['listener']['events']))),flush=True)
    counts=dict(Counter(r['screen_decision'] for r in results))
    write(OUT/'screened_candidates.json',dict(seed=SEED,pairs=planned['pairs'],policy=POLICY,rows=results,counts=counts))
    rng=random.Random(SEED+19);rng.shuffle(results);selected=[];used=set()
    # Stratified listening review: vocal-evidence keep, ambiguous, suspected noise, quiet controls.
    strata=[('keep_voice',lambda r:r['screen_decision']=='keep' and r['listener_human_evidence'],6),
            ('review',lambda r:r['screen_decision']=='review',6),
            ('suspect_noise',lambda r:r['screen_decision']=='suspect_noise',2),
            ('keep_quiet',lambda r:r['screen_decision']=='keep' and not r['listener_human_evidence'],2)]
    for name,predicate,target in strata:
        n=0
        for r in results:
            if not predicate(r) or r['pair'] in used:continue
            selected.append(dict(r,review_stratum=name));used.add(r['pair']);n+=1
            if n>=target:break
    # If a stratum is empty, fill from remaining active examples, without changing thresholds.
    for r in results:
        if len(selected)>=16:break
        if r['pair'] not in used and r['audio_qc']['listener']['events']:
            selected.append(dict(r,review_stratum='active_backfill'));used.add(r['pair'])
    write(OUT/'publication_selection.json',dict(seed=SEED,rows=selected,note='Stratified seeded review; emphasis on listener activity, maximum one clip per conversation. Not an unbiased yield estimate.'))
    print('DECISIONS',counts,'PUBLISH',dict(Counter(r['screen_decision'] for r in selected)),flush=True)


def publish():
    preview.OUT=OUT;selection=read(OUT/'publication_selection.json');assert 1<=len(selection['rows'])<=16
    with ThreadPoolExecutor(max_workers=4) as pool:samples=list(pool.map(preview.publish,selection['rows']))
    screened=read(OUT/'screened_candidates.json')
    write(OUT/'samples.json',dict(seed=SEED,inspected_pairs=screened['pairs'],counts=screened['counts'],samples=samples,policy=POLICY,status='independent_review_pending_no_bulk_or_training',note=selection['note']))
    shutil.copyfile(ROOT/'scripts/listen_qc2_review.html',OUT/'index.html')
    link=ROOT/'data_previews/seamless_face_listening_v1/listen_qc2'
    if not link.exists():link.symlink_to(OUT,target_is_directory=True)
    print('Published http://127.0.0.1:8875/listen_qc2/',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['plan','screen','publish']);a=p.parse_args()
    {'plan':plan,'screen':screen,'publish':publish}[a.stage]()
