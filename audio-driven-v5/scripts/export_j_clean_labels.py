#!/usr/bin/env python3
"""Publish and independently audit a source-bound cleanup of the frozen J labels."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import time
import orjson

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'experiments/listen_j_labels_manifest_20261105'
WORK=ROOT/'experiments/listen_j_clean_regen_20261105'
OUT=ROOT/'experiments/listen_j_labels_clean_20261105'
OLD_LABEL_SHA='b7dc75f51c9f408ff779ad9b7c4f5e42c9eee3b3a2bfdc34eda501269547ddda'
OLD_COVER_SHA='9de926f75044dc6da0b8487a3f5c94883946ca2475a7c3623f86ccab71210089'
CHANGED=['caption','caption_file','caption_file_sha256','caption_identity_version','pipeline_id','prompt_version',
         'generation_checks_passed','review_flags','remaining_gaps','finish_reason','completed_at']


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        while b:=f.read(8*1024*1024):h.update(b)
    return h.hexdigest()


def save(p,d):
    p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_bytes(orjson.dumps(d,option=orjson.OPT_INDENT_2)+b'\n');tmp.replace(p)


def read(p):return orjson.loads(Path(p).read_bytes())


def key(row):
    s=row['source']['source']
    return hashlib.sha256((s['archive']+'\0'+s['member']).encode()).hexdigest()[:24]


def context():
    decisions=read(WORK/'orientation_source_decisions_final.json')['records']
    assert len({x['source_key'] for x in decisions})==len(decisions)
    sideways={x['source_key']:x for x in decisions if x['decision']=='exclude_sideways_source'}
    hard={r['sample_id']:r for r in map(orjson.loads,(WORK/'regenerate_candidates.jsonl').read_bytes().splitlines())}
    repaired={p.stem:(p,read(p)) for p in (WORK/'regenerated').glob('*.json')}
    rejected={p.stem:(p,read(p)) for p in (WORK/'regeneration_rejected').glob('*.json')}
    assert set(repaired).isdisjoint(rejected) and set(repaired)|set(rejected)==set(hard)
    assert len(hard)==34 and len(repaired)==32 and len(rejected)==2
    return decisions,sideways,hard,repaired,rejected


def check_repair(row,p,d):
    assert d['sample_id']==row['sample_id'] and d['source']==row['source']
    assert d['crop_provenance']==row['crop_provenance'] and d['audio_recipe']==row['audio_recipe']
    assert d['parent_caption_file']==row['caption_file'] and d['parent_caption_sha256']==row['caption_file_sha256']
    assert d['hard_generation_passed'] and d['finish_reason']=='stop' and d['caption'].strip()
    parent=Path(row['caption_file']);assert sha(parent)==row['caption_file_sha256']
    oldcap=read(parent)
    for field in ['video_frames_sha256','audio_encoder_sha256']:
        assert d['input_evidence'][field]==oldcap['input_evidence'][field]==d['cache_identity'][field]
    assert not any(a['hard_failures'] for a in d['attempts'][-1:])
    assert d['cache_identity']['prompt_sha256']==hashlib.sha256(d['prompt'].encode()).hexdigest()
    return sha(p)


def export():
    started=time.monotonic();OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'EXPORT_COMPLETE.json').exists(),'Already published; do not overwrite'
    decisions,sideways,hard,repaired,rejected=context()
    names=['labels','coverage','excluded','exceptions','regeneration_replacements']
    files={n:(OUT/(n+'.jsonl.tmp')).open('wb') for n in names}
    hashes={n:hashlib.sha256() for n in names};counts=Counter();splits=Counter();flags=Counter();identities=Counter()
    sourcehash=hashlib.sha256();coverhash=hashlib.sha256();seen=set();seenhard=set();sourcecounts=Counter()
    def put(n,r,raw=None):
        raw=raw if raw is not None else orjson.dumps(r)+b'\n'
        files[n].write(raw);hashes[n].update(raw)
    try:
        with (OLD/'labels.jsonl').open('rb') as lf,(OLD/'coverage.jsonl').open('rb') as cf:
            for craw in cf:
                coverhash.update(craw);c=orjson.loads(craw);sid=c['sample_id'];counts['targets']+=1
                assert sid not in seen;seen.add(sid)
                if c['status']!='caption':
                    assert c['status']=='held';counts['held']+=1;put('coverage',c,craw);put('exceptions',c,craw);continue
                raw=next(lf);sourcehash.update(raw);row=orjson.loads(raw);assert row['sample_id']==sid
                counts['original_labels']+=1;sk=key(row);why=[]
                if sid in hard:seenhard.add(sid)
                if sk in sideways:why.append('source_scene_rotated_90_degrees')
                if sid in rejected:why.append('hard_generation_failure_after_two_attempts')
                if why:
                    ex={k:row[k] for k in ['sample_id','pair','split','group_id','listener_id','duration_s','caption_file','caption_file_sha256']}
                    ex.update(status='excluded',reasons=why,source_key=sk,source_media=row['source']['source'])
                    if sk in sideways:
                        s=sideways[sk];ex.update(orientation_review_id=s['review_id'],orientation_frame=s['frame_file'],orientation_frame_sha256=s['frame_sha256'])
                        counts['orientation_excluded']+=1;sourcecounts[sk]+=1
                    if sid in rejected:
                        ex['regeneration_diagnostic']=str(rejected[sid][0]);counts['regeneration_failed_excluded']+=1
                    if len(why)>1:counts['exclusion_reason_overlap']+=1
                    counts['excluded_unique']+=1;put('excluded',ex)
                    c.update(status='excluded',reasons=why,source_key=sk);put('coverage',c);continue
                if sid in repaired:
                    p,d=repaired[sid];h=check_repair(row,p,d)
                    rep=dict(sample_id=sid,parent_caption_file=row['caption_file'],parent_caption_sha256=row['caption_file_sha256'],
                             caption_file=str(p),caption_file_sha256=h,caption_identity_version=d['caption_identity_version'])
                    put('regeneration_replacements',rep)
                    for f in CHANGED:
                        row[f]=str(p) if f=='caption_file' else h if f=='caption_file_sha256' else d[f]
                    row['regeneration_provenance']={'parent_caption_file':rep['parent_caption_file'],
                        'parent_caption_sha256':rep['parent_caption_sha256'],'hard_generation_passed':True,
                        'input_video_frames_sha256':d['input_evidence']['video_frames_sha256'],
                        'input_audio_encoder_sha256':d['input_evidence']['audio_encoder_sha256']}
                    raw=orjson.dumps(row)+b'\n';counts['regenerated_replaced']+=1
                    c.update(caption_file=str(p),caption_file_sha256=h,generation_checks_passed=d['generation_checks_passed'],disposition='regenerated')
                else:
                    counts['unchanged']+=1;c['disposition']='original_preserved'
                assert row['finish_reason']=='stop' and row['training_admitted'] is False
                counts['labels']+=1;splits[row['split']]+=1;flags.update(row['review_flags']);identities[row['pipeline_id']]+=1
                counts['with_review_flags' if row['review_flags'] else 'without_review_flags']+=1
                counts['generation_checks_passed' if row['generation_checks_passed'] else 'generation_checks_flagged']+=1
                put('labels',row,raw);put('coverage',c)
                if counts['targets']%200000==0:print(json.dumps(dict(stage='export',counts=dict(counts))),flush=True)
            assert lf.read()==b''
        assert counts['targets']==1107323 and counts['original_labels']==1107322 and counts['held']==1
        assert seenhard==set(hard) and counts['labels']+counts['excluded_unique']+counts['held']==counts['targets']
        assert counts['regenerated_replaced']==len(repaired)
        assert sourcehash.hexdigest()==OLD_LABEL_SHA and coverhash.hexdigest()==OLD_COVER_SHA
        for f in files.values():f.close()
        for n in names:(OUT/(n+'.jsonl.tmp')).replace(OUT/(n+'.jsonl'))
        result=dict(schema='j-clean-label-manifest-v1-20261105',counts=dict(counts),splits=dict(splits),review_flags=dict(flags),
            pipelines=dict(identities),orientation=dict(reviewed_sources=len(decisions),excluded_sources=len(sideways),
                kept_upright_sources=len(decisions)-len(sideways),actual_excluded_source_clips=dict(sourcecounts)),
            source_manifest=str(OLD/'labels.jsonl'),source_manifest_sha256=sourcehash.hexdigest(),
            source_coverage_sha256=coverhash.hexdigest(),orientation_decisions=str(WORK/'orientation_source_decisions_final.json'),
            orientation_decisions_sha256=sha(WORK/'orientation_source_decisions_final.json'),
            files={n+'.jsonl':dict(sha256=hashes[n].hexdigest(),bytes=(OUT/(n+'.jsonl')).stat().st_size) for n in names},
            policy=dict(preserve_pass_and_review=True,source_scene_rotation_only=True,normal_head_tilt_preserved=True,
                failed_regeneration_excluded=True,original_media_and_split_preserved=True,
                per_row_caption_identity='Original captions plus independently identified repair versions; read each row identity',
                behavior_statistics='not_recomputed_for_this_caption_version',training_selection='not_reselected'),elapsed_s=time.monotonic()-started)
        save(OUT/'EXPORT_SUMMARY.json',result);print(json.dumps(result,ensure_ascii=False),flush=True)
    finally:
        for f in files.values():
            if not f.closed:f.close()


def audit():
    started=time.monotonic();summary=read(OUT/'EXPORT_SUMMARY.json')
    decisions,sideways,hard,repaired,rejected=context();counts=Counter();seen=set();kept_upright=Counter();excluded_sources=Counter()
    special={r['sample_id'] for r in map(orjson.loads,(WORK/'short_repetition_retained.jsonl').read_bytes().splitlines())}
    special.update(['naturalistic_V03_S0984_I00000046_local_63463d08404f','naturalistic_V03_S2113_I00000261_local_f3d48eca5de1',
        'improvised_V00_S1525_I00000652_local_d0b4246485c1','naturalistic_V02_S3295_I00000385_local_7a412befd305','improvised_V00_S1630_I00001032_local_944640741a7b'])
    special_status={};upright={x['source_key'] for x in decisions if x['decision']=='keep_upright_source'}
    groups={};oldsha=hashlib.sha256();oldcsha=hashlib.sha256()
    with (OLD/'coverage.jsonl').open('rb') as oldc,(OLD/'labels.jsonl').open('rb') as oldl,\
         (OUT/'coverage.jsonl').open('rb') as newc,(OUT/'labels.jsonl').open('rb') as newl,\
         (OUT/'excluded.jsonl').open('rb') as exc:
        for ocraw in oldc:
            oldcsha.update(ocraw);oc=orjson.loads(ocraw);c=orjson.loads(next(newc));sid=oc['sample_id']
            assert c['sample_id']==sid and sid not in seen;seen.add(sid);counts['targets']+=1
            for f in ['pair','split','duration_s']:assert c[f]==oc[f]
            if oc['status']=='held':
                assert c==oc;counts['held']+=1;continue
            oraw=next(oldl);oldsha.update(oraw);o=orjson.loads(oraw);assert o['sample_id']==sid
            sk=key(o);expected_ex=sk in sideways or sid in rejected
            if expected_ex:
                ex=orjson.loads(next(exc));assert c['status']=='excluded' and ex['sample_id']==sid
                assert ex['source_key']==sk and ex['caption_file_sha256']==o['caption_file_sha256']
                expected_reasons=[]
                if sk in sideways:expected_reasons.append('source_scene_rotated_90_degrees');excluded_sources[sk]+=1
                if sid in rejected:expected_reasons.append('hard_generation_failure_after_two_attempts')
                assert ex['reasons']==c['reasons']==expected_reasons;counts['excluded']+=1
                if sid in special:special_status[sid]='excluded'
                continue
            raw=next(newl);r=orjson.loads(raw);assert r['sample_id']==sid and c['status']=='caption'
            if sid in repaired:
                p,d=repaired[sid];h=check_repair(o,p,d);counts['repairs']+=1
                assert set(r)==set(o)|{'regeneration_provenance'}
                for f in o:
                    if f not in CHANGED:assert r[f]==o[f],(sid,f)
                for f in CHANGED:assert r[f]==(str(p) if f=='caption_file' else h if f=='caption_file_sha256' else d[f]),(sid,f)
                counts['repair_media_hash_pairs_verified']+=1
                caption=r['caption'].lower();sentences=re.split(r'[.!?]+\s*',caption)
                frequent=Counter(s.strip() for s in sentences if len(s.strip())>=20)
                assert max(frequent.values(),default=0)<8
                words=re.findall(r'\b\w+\b',caption);ngrams=Counter(tuple(words[i:i+16]) for i in range(len(words)-15))
                assert max(ngrams.values(),default=0)<8
            else:assert raw==oraw;counts['byte_identical_original_rows']+=1
            assert r['finish_reason']=='stop' and r['caption'].strip() and r['training_admitted'] is False
            assert sk not in sideways and sid not in rejected
            for f in ['caption_file','caption_file_sha256','generation_checks_passed']:assert c[f]==r[f]
            g=r['group_id'];assert groups.setdefault(g,r['split'])==r['split']
            if sk in upright:kept_upright[sk]+=1
            if sid in special:special_status[sid]='regenerated' if sid in repaired else 'original_preserved'
            counts['labels']+=1
            if counts['targets']%200000==0:print(json.dumps(dict(stage='audit',counts=dict(counts))),flush=True)
        assert oldl.read()==newc.read()==newl.read()==exc.read()==b''
    assert oldsha.hexdigest()==OLD_LABEL_SHA and oldcsha.hexdigest()==OLD_COVER_SHA
    assert counts['labels']==summary['counts']['labels'] and counts['excluded']==summary['counts']['excluded_unique']
    assert counts['repairs']==32 and len(kept_upright)==len(upright) and set(excluded_sources)==set(sideways)
    assert len(special_status)==len(special)
    for sid in special:
        assert special_status[sid]==('excluded' if 'S0984' in sid else 'regenerated' if sid in repaired else 'original_preserved')
    assert (OUT/'exceptions.jsonl').read_bytes()==(OLD/'exceptions.jsonl').read_bytes()
    for x in decisions:assert sha(x['frame_file'])==x['frame_sha256']
    checked={}
    for name,meta in summary['files'].items():
        digest=sha(OUT/name);assert digest==meta['sha256'] and (OUT/name).stat().st_size==meta['bytes'];checked[name]=digest
    assert sha(WORK/'orientation_source_decisions_final.json')==summary['orientation_decisions_sha256']
    result=dict(status='passed',counts=dict(counts),duplicate_ids=0,omitted_target_ids=0,changed_original_rows=0,
        retained_sideways_sources=0,retained_failed_generations=0,group_cross_split=0,
        kept_upright_reviewed_sources=len(kept_upright),kept_upright_reviewed_clips=sum(kept_upright.values()),
        special_sample_status=special_status,verified_output_sha256=checked,elapsed_s=time.monotonic()-started)
    save(OUT/'CLEAN_AUDIT.json',result)
    save(OUT/'EXPORT_COMPLETE.json',dict(status='complete',summary_sha256=sha(OUT/'EXPORT_SUMMARY.json'),audit_sha256=sha(OUT/'CLEAN_AUDIT.json'),completed_at=time.time()))
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['export','audit']);a=ap.parse_args()
    (export if a.command=='export' else audit)()
