#!/usr/bin/env python3
"""Select a count-limited, seven-class test panel from already processed candidates."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from p2_listen.contracts import atomic_json, sha, digest, read
from listen_benchmark.backchannel_face_stats import CATEGORIES
from listen_benchmark.postprocessed_subset import event_statistics
from listen_benchmark.train_subset import Selection, score, summarize, audit_catalog, identities


def current_statistics(value):
    row = dict(value['statistics']); label = value['integrated']; projection = []
    for e in label['events']:
        for cat in e.get('categories', []):
            if cat not in CATEGORIES:
                continue
            for i, span in enumerate(e.get('intervals', [dict(start=e['start'], end=e['end'])])):
                projection.append(dict(event_id=f"{e['event_id']}:{cat}:{i}", source=dict(event_id=e['event_id']),
                                       type=cat, start=span['start'], end=span['end']))
    measured = event_statistics(dict(events=projection), row['duration_s'])
    for field in ['event_count','non_smile_event_count','category_count'] + [c+'_count' for c in CATEGORIES]:
        if measured[field] != row[field]:
            raise ValueError(('Stale postprocessed event statistics', value['sample_id'], field))
    row.update(measured)
    row['selection_score'], row['capped_event_count'], row['capped_density_per_10s'] = score(row)
    return row


def select(rows, count):
    if count < len(CATEGORIES):
        raise ValueError('Panel must have room to reserve seven-class coverage')
    eligible = [dict(r) for r in rows if r['split']=='test' and not r['label_issue_flag']
                and r['non_smile_event_count'] >= 1 and r['face_valid_fraction'] >= .98
                and r['face_short_side_p10'] >= 160]
    for r in eligible:
        r['selection_score'], r['capped_event_count'], r['capped_density_per_10s'] = score(r)
    ranked = sorted(eligible, key=lambda r:(-r['selection_score'], -r['face_short_side_p10'], r['sample_id']))
    result = Selection(8); core_count = math.floor(.9*count)
    for r in ranked:
        if len(result.rows) >= core_count:
            break
        if r['face_short_side_p10'] >= 224:
            result.add(r, 'core')
    if len(result.rows) != core_count:
        raise ValueError('Insufficient independent test core')
    for cat in sorted(CATEGORIES, key=lambda c:(sum(r[c+'_count']>0 for r in eligible), c)):
        if result.counts[cat]:
            continue
        floor = 160 if cat=='head_shake' else 192
        found = next((r for r in ranked if r[cat+'_count'] and r['face_short_side_p10'] >= floor and result.reason(r) is None), None)
        if found is None or len(result.rows) >= count:
            raise ValueError('Cannot cover class without changing the fixed selection policy: '+cat)
        result.add(found, 'supplement:'+cat)
    for r in ranked:
        if len(result.rows) >= count:
            break
        if r['face_short_side_p10'] >= 224:
            result.add(r, 'core_fill')
    if len(result.rows) != count or any(not result.counts[c] for c in CATEGORIES):
        raise ValueError('Incomplete panel')
    policy = dict(samples=count, core_samples=core_count, eligible_candidates=len(eligible),
        eligible_class_samples={c:sum(r[c+'_count']>0 for r in eligible) for c in CATEGORIES},
        score='0.5*min(face_p10/384,1)+0.3*class_count/7+0.2*min(capped_density/12,1)',
        capped_density='10*sum(min(class_count,3), except smile cap=1)/max(duration_s,5)',
        core_face_p10=224, supplement_face_p10=192, head_shake_face_p10=160, face_valid_fraction=.98,
        max_per_pair=8, source_intervals_nonoverlapping=True, minimum_samples_per_class=1,
        classes_are_weak=True, generation_outputs_used_for_selection=False)
    return result.rows, policy


def main():
    p=argparse.ArgumentParser();p.add_argument('--candidates',type=Path,required=True)
    p.add_argument('--train',type=Path,required=True);p.add_argument('--catalog',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--count',type=int,default=48)
    p.add_argument('--workers',type=int,default=8);a=p.parse_args()
    source=a.candidates.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    values={};source_hashes={}
    for path in sorted((source/'postprocessed').glob('*.json')):
        value=read(path);sid=value['sample_id']
        if sid in values:
            raise ValueError('Duplicate candidate ID')
        values[sid]=value;source_hashes[str(path)]=sha(path)
    if set(values)!={r['sample_id'] for r in read(source/'annotation_shortlist.json')}:
        raise ValueError('Candidate coverage mismatch')
    manifest=dict(candidates=str(source),train=str(a.train.resolve()),count=a.count,
        candidate_hashes=source_hashes,train_selection_sha256=sha(a.train/'selection.json'),
        catalog_sha256=sha(a.catalog),script_sha256=sha(__file__))
    if (out/'provenance.json').exists() and read(out/'provenance.json')!=manifest:
        raise ValueError('Frozen selection inputs changed; use a new output')
    atomic_json(out/'provenance.json',manifest)
    if (out/'COMPLETE.json').exists():
        print('Already complete:',out);return
    rows,policy=select([current_statistics(v) for v in values.values()],a.count)
    pairs,catalog_counts=audit_catalog(read(a.catalog))
    train_rows=read(a.train/'selection.json')
    train_keys=set().union(*(identities(pairs[r['pair']]) for r in train_rows))
    verifier=Selection(8)
    for r in rows:
        if r['split']!='test' or pairs[r['pair']]['split']!='test' or train_keys & identities(pairs[r['pair']]):
            raise ValueError('Train/test identity leakage')
        if not verifier.add(r,'audit'):
            raise ValueError('Duplicate, pair cap, or overlapping source interval')
    selected_ids={r['sample_id'] for r in rows}
    old_ids=set((source/'sample_ids.txt').read_text().splitlines())
    policy.update(annotation_candidates=len(values),previous_test_overlap=len(selected_ids&old_ids),
                  previous_test_samples=len(old_ids))
    atomic_json(out/'selection.json',rows);atomic_json(out/'selection_policy.json',policy)
    atomic_json(out/'config.json',dict(project=str(ROOT),raw_run=str(source/'raw_run'),fusion_run=str(source/'fusion')))
    if not (out/'source_index').exists():
        (out/'source_index').symlink_to(source/'source_index',target_is_directory=True)
    source_index=read(source/'source_index/COMPLETE.json')
    if sha(source/'source_index/offsets.sqlite')!=source_index['db_sha256']:
        raise ValueError('Source index changed')
    annotation_audio={}
    for file in sorted((source/'raw_run/raw/audio/results').glob('*.jsonl')):
        with file.open() as f:
            for line in f:
                receipt=json.loads(line)
                if receipt.get('record_kind')=='result' and receipt.get('view')=='full' and receipt['sample_id'] in selected_ids and receipt.get('status')=='schema_pass':
                    annotation_audio[receipt['sample_id']]=receipt['input_evidence']['audio']['wav_sha256']
    if set(annotation_audio)!=selected_ids:
        raise ValueError('Missing full-clip listener input receipt')
    from listen_benchmark.feature_pipeline import export_media,source_row
    from listen_audio_postprocess.recipes import make_recipe,read_listener_audio
    from p2_listen.dataset import verify_media
    def export(r):
        sid=r['sample_id'];v=values[sid];c=v['integrated']
        source_row(out,c['source'])
        old_media=source/'media'/sid;new_media=out/'media'/sid
        if (old_media/'COMPLETE.json').exists() and not new_media.exists():
            new_media.parent.mkdir(parents=True,exist_ok=True);new_media.symlink_to(old_media,target_is_directory=True)
        media=export_media(out,c);obs=dict(media['observation']);target=media['target']
        if obs['official_split']!='test' or obs['experiment_split']!='test':
            raise ValueError('Test split lost')
        obs['grouping_ids']=sorted(identities(pairs[r['pair']]))
        verify_media(obs,target)
        for name,h in media['meta']['sha256'].items():
            if sha(new_media/name)!=h:
                raise ValueError('Media hash mismatch')
        if target['media_hashes']['target_audio']!=annotation_audio[sid]:
            raise ValueError('Listener audio differs from annotation input')
        recipe=make_recipe(sid,target['target_audio'],target['media_hashes']['target_audio'],v['official_listener'],v['extra_mute_intervals'],v['provenance'])
        atomic_json(out/f'audio_postprocess/recipes/{sid}.json',recipe)
        wave,rate,meta=read_listener_audio(recipe)
        import numpy as np
        if not np.isfinite(wave).all() or abs(len(wave)/rate-obs['requested_duration_s'])>1/rate+1e-6:
            raise ValueError('Invalid virtual waveform')
        return obs,target,dict(sample_id=sid,recipe_sha256=digest(recipe),media=media['meta']['sha256'],
            source_line_sha256=r['source_line_sha256'],postprocessed_source_sha256=source_hashes[str(source/'postprocessed'/f'{sid}.json')])
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        exported=list(pool.map(export,rows))
    atomic_json(out/'manifests/observations.json',[x[0] for x in exported])
    atomic_json(out/'manifests/targets.json',[x[1] for x in exported])
    atomic_json(out/'media_references.json',[x[2] for x in exported])
    with (out/'integrated_labels.jsonl').open('w') as f:
        for r in rows:
            f.write(json.dumps(values[r['sample_id']]['integrated'],ensure_ascii=False)+'\n')
    with (out/'samples.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (out/'sample_ids.txt').write_text(''.join(r['sample_id']+'\n' for r in rows))
    summary=summarize(rows);summary.update(split='test',labels_are_weak=True,formal_evaluation_reference=False)
    atomic_json(out/'summary.json',summary)
    validation=dict(status='passed',samples=a.count,all_file_hashes_verified=True,
        all_listener_waveforms_match_annotation_input=True,all_virtual_audio_reads_passed=True,
        source_overlap=0,train_test_identity_overlap=0,full_catalog_split_identity_isolation=True,
        classes=dict(verifier.counts),weak_labels_not_human_ground_truth=True,training_started=False)
    atomic_json(out/'validation.json',validation)
    atomic_json(out/'dataset_contract.json',dict(split='test',samples=a.count,training_admitted=False,
        human_ground_truth=False,formal_metric_reference_admitted=False,seven_class_coverage_verified=True,
        audio_policy='Original listener clip plus virtual silence recipe; no processed WAV',feature_caches_created=False))
    names=['config.json','summary.json','selection.json','selection_policy.json','samples.csv','sample_ids.txt',
        'integrated_labels.jsonl','manifests/observations.json','manifests/targets.json','media_references.json',
        'dataset_contract.json','validation.json','provenance.json']
    atomic_json(out/'COMPLETE.json',dict(status='passed',samples=a.count,files={n:sha(out/n) for n in names},training_started=False))
    print(json.dumps(dict(summary=summary,policy=policy),ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
