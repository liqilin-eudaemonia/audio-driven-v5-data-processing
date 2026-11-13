"""Plan, render and re-screen 24 fixed V7/V8 boundary comparisons, six strata."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

import plan_listen_v8 as planner
import publish_listen_v7_sample as exporter

ROOT=planner.ROOT
OLD=ROOT/'data_previews/seamless_listen_v7_sample'
OUT=ROOT/'data_previews/seamless_listen_v8_boundaries'
EXP=ROOT/'experiments/p3_listen_v8_boundary_validation'
SEED=202610221330
CODE=['listen_turns_v8.py','listen_policy_v8.py','plan_listen_v8.py']
CPU_WORKERS=max(1,int(os.environ.get('LISTEN_V8_CPU_WORKERS','32')))


def fingerprint():return {f:exporter.sha(ROOT/'scripts'/f) for f in CODE}


def select():
    data=json.loads((OLD/'samples_categories.json').read_text());rng=random.Random(SEED);rows=[]
    special=['improvised_V03_S1431_I00000373_v7_p014_00',
             'naturalistic_V02_S3177_I00000132_v7_p008_00',
             'naturalistic_V01_S1424_I00000127_v7_p002_01']
    for category in ['keep_primary','keep_onset_reserve','review_primary','review_onset_reserve','exclude_primary','exclude_onset_reserve']:
        pool=[r for r in data['samples'] if r['review_category']==category];rng.shuffle(pool)
        chosen=[r for sid in special for r in pool if r['sample_id']==sid] if category=='keep_primary' else []
        chosen += [r for r in pool if r['sample_id'] not in {x['sample_id'] for x in chosen}][:4-len(chosen)]
        rows.extend(chosen)
    assert len(rows)==24 and len({r['group_id'] for r in rows})==24
    exporter.write(EXP/'selection.json',dict(seed=SEED,rows=rows,known_problem_ids=special,
        note='Three known boundary failures plus 21 controls sampled from the existing six-category page; not a corpus accuracy estimate.'))
    return rows


def plan():
    OUT.mkdir(parents=True,exist_ok=True);EXP.mkdir(parents=True,exist_ok=True)
    selected=select();catalog=json.loads((exporter.CORPUS/'catalog.json').read_text())
    lookup={r['pair']:r for r in catalog['rows']};code=fingerprint()
    def one(old):
        path=EXP/'audit'/(old['pair']+'.json')
        audit=json.loads(path.read_text()) if path.exists() else None
        if not audit or audit.get('code_sha256')!=code:
            audit=planner.inspect_pair(lookup[old['pair']]);audit['code_sha256']=code;exporter.write(path,audit)
        candidates=[]
        for row in audit['clips']:
            if row['speaker_id']!=old['speaker_id']:continue
            overlap=max(0,min(row['speech_end'],old['speech_end'])-max(row['speech_start'],old['speech_start']))
            if overlap<.5:continue
            covers=row['speech_start']<=old['speech_start']+1e-6 and row['speech_end']>=old['speech_end']-1e-6
            candidates.append((covers,overlap,-abs(row['speech_start']-old['speech_start']),row))
        new=dict(max(candidates,key=lambda x:x[:3])[3]) if candidates else None
        if new:
            new['normalized_audio']=old['normalized_audio']
            new['preview']=f"media/{new['sample_id']}/"
            new['listener_screen']=dict(accepted=True,deferred_reasons=
                ['listener_short_speech_unconfirmed'] if new['listener_compatibility']['uncertain_short_speech'] else [])
        unresolved=[r for r in audit['unresolved'] if r.get('speaker_index')==old['speaker_index']
                    and r['end']>old['start'] and r['start']<old['end']]
        if new is None and not unresolved:
            unresolved=[dict(start=old['start'],end=old['end'],speaker_index=old['speaker_index'],
                reason='no_complete_candidate_for_original_role',text=old['text'],
                new_floors=[f for f in audit['floors'] if f['end']>old['start'] and f['start']<old['end']])]
        # Normalized continuous context is for inspection only, never a training candidate.
        start=max(0.,min(old['start'],new['start'] if new else old['start'])-2)
        end=min(audit['source_seconds'],max(old['end'],new['end'] if new else old['end'])+2)
        context=dict(old,sample_id=old['sample_id']+'_context',start=start,end=end,duration_s=end-start,
                     preview=f"media/{old['sample_id']}_context/",inspection_context_only=True)
        result=dict(old=old,new=new,context=context,unresolved=unresolved,
                    case_id=old['sample_id'],old_category=old['review_category'],
                    status='new_candidate' if new else 'unresolved_no_forced_cut')
        print(json.dumps(dict(planned=old['sample_id'],new=bool(new),unresolved=len(unresolved))),flush=True)
        return result
    with ThreadPoolExecutor(max_workers=min(CPU_WORKERS,len(selected))) as pool:cases=list(pool.map(one,selected))
    exporter.write(EXP/'planned.json',dict(cases=cases,code_sha256=code,policy=planner.POLICY))


def render():
    data=json.loads((EXP/'planned.json').read_text());assert data['code_sha256']==fingerprint()
    exporter.OUT=OUT
    def one(job):
        case,key=job;row=case[key]
        folder=OUT/row['preview'];marker=folder/'export.json'
        old=json.loads(marker.read_text()) if marker.exists() else None
        if old and all(old[k]==row[k] for k in ('start','end','listener_crop','normalized_audio','sources')):
            assert all(exporter.sha(folder/(m+'.mp4'))==c['sha256'] for m,c in old['media_checks'].items())
            case[key]=dict(row,media_checks=old['media_checks'],audio_checks=old['audio_checks'],media_verified=True)
        else:case[key]=exporter.export(row)
    cases=data['cases']
    jobs=[(case,key) for case in cases for key in ('new','context') if case[key] is not None]
    with ThreadPoolExecutor(max_workers=min(CPU_WORKERS,max(1,len(jobs)))) as pool:
        list(pool.map(one,jobs))
    for case in cases:
        link=OUT/'media'/('old_'+case['case_id']);link.parent.mkdir(parents=True,exist_ok=True)
        if not link.exists():link.symlink_to(OLD/case['old']['preview'],target_is_directory=True)
        case['old']=dict(case['old'],preview=f'media/old_{case["case_id"]}/')
    exporter.write(EXP/'rendered.json',dict(cases=cases,code_sha256=fingerprint()))


def qc_worker(rank):
    import torch
    torch.set_num_threads(2)
    import run_listen_v7_bulk as bulk
    bulk.verify_contract()
    rows=[c['new'] for c in json.loads((EXP/'rendered.json').read_text())['cases'] if c['new']][rank::8]
    if not rows:return
    ast=bulk.EventModel(bulk.MODEL,device='cuda:0',batch_size=16)
    panns=bulk.PannsModel(device='cuda:0',batch_size=16)
    rejects=bulk.read(bulk.OUT/'human_rejections.json')['rows'];positives=bulk.read(bulk.OUT/'human_retained.json')['rows']
    for row in rows:
        tracks=[None,None]
        for role in ('speaker','listener'):tracks[row[role+'_index']]=row['normalized_audio'][role]
        result=bulk.scan(row,dict(tracks=tracks),ast,panns,rejects,positives)
        result['audio_qc_status']=result['effective_decision']
        result['v8_combined_status']=('exclude' if result['effective_decision']=='exclude' or row['source_manual_rejections']
            else 'review' if result['effective_decision']=='review' or row['boundary_review_flags'] else 'candidate')
        exporter.write(EXP/'qc'/(row['sample_id']+'.json'),result)
        print(json.dumps(dict(rank=rank,sample=row['sample_id'],audio_qc=result['audio_qc_status'])),flush=True)


def qc():
    (EXP/'logs').mkdir(parents=True,exist_ok=True);jobs=[]
    for rank in range(8):
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(rank),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2')
        log=(EXP/'logs'/f'qc_{rank}.log').open('w')
        p=subprocess.Popen([sys.executable,__file__,'qc-worker','--rank',str(rank)],env=env,stdout=log,stderr=subprocess.STDOUT)
        log.close();jobs.append(p)
    codes=[p.wait() for p in jobs]
    if any(codes):raise RuntimeError('QC worker failures: '+str(codes))


def publish():
    data=json.loads((EXP/'rendered.json').read_text());assert data['code_sha256']==fingerprint()
    cases=data['cases']
    for c in cases:
        if c['new']:
            q=json.loads((EXP/'qc'/(c['new']['sample_id']+'.json')).read_text())
            assert (q['start'],q['end'])==(c['new']['start'],c['new']['end'])
            c['new'].update(audio_qc_status=q['audio_qc_status'],v8_combined_status=q['v8_combined_status'],
                qc_reasons={role:v['reasons'] for role,v in q['qc3'].items()})
    stats=dict(cases=len(cases),new_candidates=sum(bool(c['new']) for c in cases),
        unresolved=sum(c['new'] is None for c in cases),
        over_10s=sum(bool(c['new'] and c['new']['duration_s']>10) for c in cases),
        boundary_status=dict(Counter(c['new']['boundary_status'] for c in cases if c['new'])),
        combined_status=dict(Counter(c['new']['v8_combined_status'] for c in cases if c['new'])))
    exporter.write(OUT/'samples.json',dict(cases=cases,summary=stats,policy=planner.POLICY,code_sha256=fingerprint(),
        scope='24 matched boundary comparisons; all 6 old QC categories. Audio QC rerun on new intervals. No full-corpus replacement or training.'))
    template=(ROOT/'scripts/listen_v8_boundaries.html').read_text()
    from fontTools import subset
    from fontTools.ttLib import TTFont
    font=TTFont('/tmp/seamless_browser_libs/fonts/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',fontNumber=2)
    sub=subset.Subsetter();sub.populate(text=template);sub.subset(font);font.flavor='woff';font.save(OUT/'review.woff')
    (OUT/'index.html').write_text(template)
    webroot=ROOT/'data_previews/seamless_face_listening_v1';link=webroot/'listen_v8'
    if not link.exists():link.symlink_to(OUT,target_is_directory=True)
    # Keep the previous six-category review address available, and add a clear link.
    old_page=OLD/'index.html';text=old_page.read_text()
    banner='<p><a href="/listen_v8/" style="color:#a4ddff">新版：V8 完整句子切分 · 24 组新旧对照</a></p>'
    if banner not in text:old_page.write_text(text.replace('</header>',banner+'</header>'))
    exporter.write(EXP/'validation_summary.json',stats)
    print(json.dumps(dict(published='http://127.0.0.1:8875/listen_v8/',**stats)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['all','plan','render','qc','qc-worker','publish']);p.add_argument('--rank',type=int,default=0);a=p.parse_args()
    if a.stage=='all':plan();render();qc();publish()
    elif a.stage=='qc-worker':qc_worker(a.rank)
    else:globals()[a.stage]()
