#!/usr/bin/env python3
"""Audit source preservation, media-cache identity, isolation, and open reader."""
import argparse
from collections import Counter,defaultdict
import hashlib
import itertools
import json
from pathlib import Path
import random
import multiprocessing as mp
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from p2_listen.contracts import read,sha,digest,atomic_json
from p2_listen.config import load,activate
from listen_benchmark.backchannel_face_stats import loads,CATEGORIES
from listen_benchmark.integration_pilot import duplicate
from listen_benchmark.open_subset_reader import load_open_training_sample

_AUDIT_ONE=None
def _audit_item(job):
    return _AUDIT_ONE(job)


def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--workers',type=int,default=32);a=p.parse_args();root=a.root.resolve()
    if not (root/'export_complete.json').exists():raise ValueError('Open export incomplete')
    cfg=load(root/'media_cache_config.json');activate(cfg)
    from listen_audio_postprocess.overlay import sample_contract
    from p2_listen.features import identity
    from listen_benchmark.audit_identity_snapshot import IdentitySnapshot
    snapshot=IdentitySnapshot(cfg)
    observations=read(root/'manifests/observations.json');targets=read(root/'manifests/targets.json');rows=read(root/'selection.json')
    refs=[loads(l) for l in (root/'dataset.jsonl').open('rb')];ids=[r['sample_id'] for r in rows]
    if len(ids)!=len(set(ids)) or any(ids!=[r['sample_id'] for r in records] for records in [observations,targets,refs]):raise ValueError('ID/order mismatch')
    if ids!=(root/'sample_ids.txt').read_text().splitlines():raise ValueError('ID text mismatch')
    spans=defaultdict(list);counts=Counter()
    for r in rows:
        floor=192 if r['selection_reason'].startswith('supplement:') and r['selection_reason']!='supplement:head_shake' else 160
        if r['split']!='train' or r['label_issue_flag'] or r['face_valid_fraction']<.98 or r['face_short_side_p10']<floor:
            raise ValueError('Open row fails approved quality threshold')
        if not r['open_event_count'] or not (r['open_extra_visual_count'] or r['canonical_mask']&123):raise ValueError('Missing non-smile/open evidence')
        spans[r['pair']].append((r['source_start_s'],r['source_end_s']))
        counts.update(c for c in CATEGORIES if r[c+'_count'])
    for values in spans.values():
        values.sort()
        if len(values)>24 or any(b[0]<a[1]-1e-6 for a,b in zip(values,values[1:])):raise ValueError('Duplicate/overlapping source support')
    if set(counts)!=set(CATEGORIES):raise ValueError('Existing seven classes not all represented')
    from listen_benchmark.train_subset import audit_catalog,audit_eval_manifests
    catalog=ROOT/'data_corpora/seamless_listen_local_full/catalog.json';pairs,_=audit_catalog(read(catalog))
    split_check=audit_eval_manifests(catalog.parent/'manifests',pairs,rows)
    def one(job):
        obs,target,row,ref,raw=job;sid=row['sample_id'];label=loads(raw)
        if label['sample_id']!=sid or hashlib.sha256(raw).hexdigest()!=ref['open_label_sha256']:raise ValueError('Open label/order mismatch')
        if label['new_semantic_classification'] or label['category_restriction'] is not None:raise ValueError('Unexpected classifier or seven-class filter')
        if obs['official_split']!='train' or obs['experiment_split']!='train':raise ValueError('Wrong media split')
        _,_,_,recipe,canonical,overlay_identity=sample_contract(cfg,obs,target)
        if sha(root/'labels/canonical'/f'{sid}.json')!=ref['canonical_sha256'] or sha(root/'audio_postprocess/recipes'/f'{sid}.json')!=ref['recipe_sha256']:raise ValueError('Reference or recipe changed')
        for part,info in ref['cache'].items():
            if part=='events':raise ValueError('Seven-class events must not be active open targets')
            marker=read(Path(info['path'])/'complete.json')
            expected=overlay_identity if part=='audio_postprocess' else identity(cfg,part,obs,target if part=='vae' else None)
            if marker['identity']!=digest(expected) or marker['identity']!=info['identity'] or marker['payload_sha256']!=info['payload_sha256']:raise ValueError('Stale media cache')
        source=label['source_evidence']
        with Path(source['file']).open('rb') as f:f.seek(source['offset']);body=f.read(source['size'])
        if hashlib.sha256(body).hexdigest()!=source['sha256']:raise ValueError('Original evidence changed')
        evidence=loads(body);original={e['event_id']:e for e in evidence['original_events'] if e['modality']=='visual'}
        decisions={d['open_event_id']:d for d in evidence['decisions']};seen=set();visual=[e for e in label['events'] if e['modality']=='visual']
        for event in visual:
            members=[original[i] for i in event['source_observation_ids']]
            if not members or seen&set(event['source_observation_ids']):raise ValueError('Missing/repeated evidence assignment')
            seen.update(event['source_observation_ids'])
            if (event['start'],event['end']) not in {(e['start'],e['end']) for e in members}:raise ValueError('Synthetic/union boundary')
            strings={s['description'] for e in members for s in e.get('source_descriptions',[]) } | {e['behavior'] for e in members}
            if event['description'] not in strings or not set(event['original_descriptions'])<=strings:raise ValueError('Original wording rewritten')
            if event['duplicate_observations_removed']!=len(members)-1:raise ValueError('Dedup count mismatch')
            if any(not duplicate(x,y,decisions) for x,y in itertools.combinations(members,2)):raise ValueError('Changed old dedup semantics')
        if seen!=set(original):raise ValueError('Dropped original visual observation')
        record_ref=label['source_postprocess_record'];record=read(record_ref['file'])
        if sha(record_ref['file'])!=record_ref['sha256']:raise ValueError('Processed audio record changed')
        expected_audio=sorted(record['audio_events'],key=lambda e:(e['start'],e['end'],e['event_id']))
        if [e for e in label['events'] if e['modality']=='audio']!=expected_audio:raise ValueError('Processed audio modified/restored')
        return len(original),len(visual),len(expected_audio)
    total=Counter();index=0
    global _AUDIT_ONE
    _AUDIT_ONE=one
    with (root/'integrated_labels.jsonl').open('rb') as stream,mp.get_context('fork').Pool(a.workers) as workers:
        while index<len(ids):
            size=min(512,len(ids)-index);raws=[stream.readline() for _ in range(size)]
            if not all(raws):raise ValueError('Open labels truncated')
            jobs=zip(observations[index:index+size],targets[index:index+size],rows[index:index+size],refs[index:index+size],raws)
            for original,visual,audio in workers.imap_unordered(_audit_item,jobs,chunksize=4):
                total['original_visual_observations']+=original;total['integrated_visual_events']+=visual;total['audio_events']+=audio
            index+=size;atomic_json(root/'audit_progress.json',dict(audited=index,total=len(ids)))
            if index%8192==0:print(json.dumps(dict(audited=index,total=len(ids))),flush=True)
        if stream.read(1):raise ValueError('Unexpected extra open label')
    picked=set(random.Random(20261027).sample(range(len(ids)),min(32,len(ids))))
    for c in CATEGORIES:picked.add(next(i for i,r in enumerate(rows) if r[c+'_count']))
    extras=[i for i,r in enumerate(rows) if not r['canonical_mask'] and r['open_extra_visual_count']]
    if extras:picked.add(extras[0])
    for i in sorted(picked):
        *_,opened=load_open_training_sample(root,observations[i],targets[i],'cpu')
        if not isinstance(opened,dict) or opened['sample_id']!=ids[i]:raise ValueError('Reader dropped open labels')
    snapshot_check=snapshot.verify_and_restore()
    validation=dict(status='passed',samples=len(ids),duration_s=sum(o['requested_duration_s'] for o in observations),
        original_visual_evidence_fully_preserved=True,descriptions_rewritten=False,new_semantic_classification=False,
        dedup_matches_previous_algorithm=True,filtered_audio_events_unchanged=True,all_cache_contracts_verified=True,
        all_source_evidence_hashes_verified=True,within_pair_overlap=0,cross_split_identity_overlap=0,
        actual_open_training_reads=len(picked),canonical_reference_class_samples=dict(counts),totals=dict(total),
        labels_are_weak=True,stock_seven_class_event_trainer_compatible=False,training_started=False,immutable_identity_snapshot=snapshot_check)
    atomic_json(root/'validation.json',validation)
    names=['media_cache_config.json','selection_policy.json','summary.json','selection.json','split_audit.json','samples.csv','sample_ids.txt',
        'dataset.jsonl','integrated_labels.jsonl','open_label_index.sqlite','manifests/observations.json','manifests/targets.json',
        'dataset_contract.json','validation.json','PREPARE_COMPLETE.json']
    atomic_json(root/'COMPLETE.json',dict(status='passed',samples=len(ids),hours=validation['duration_s']/3600,
        label_mode='all_original_descriptions_with_existing_dedup',files={n:sha(root/n) for n in names},training_started=False))
    atomic_json(root/'audit_progress.json',dict(audited=len(ids),total=len(ids),validated=True))
    atomic_json(root/'progress.json',dict(phase='complete',samples=len(ids),total=len(ids),validated=True))
    print(json.dumps(validation),flush=True)


if __name__=='__main__':main()
