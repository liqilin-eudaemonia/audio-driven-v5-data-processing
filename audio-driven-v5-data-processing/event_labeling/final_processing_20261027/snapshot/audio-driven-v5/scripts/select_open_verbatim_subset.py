#!/usr/bin/env python3
"""Select and export an independently ranked open-description training subset."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import copy
import csv
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from p2_listen.contracts import read,sha,atomic_json
from listen_benchmark.backchannel_face_stats import loads,CATEGORIES
from listen_benchmark.train_subset import Selection,audit_catalog,audit_eval_manifests,summarize
from listen_benchmark.postprocessed_subset import export_one,event_statistics


def choose(rows,hours=300):
    eligible=[]
    for r in rows:
        if not r['open_event_count'] or (not (r['canonical_mask']&123) and not r['open_extra_visual_count']):continue
        r=dict(r)
        r['selection_score']=.5*min(r['face_short_side_p10']/384,1)+.3*min(r['open_ranking_description_count']/7,1)+.2*min(r['open_capped_density']/12,1)
        eligible.append(r)
    ranked=sorted(eligible,key=lambda r:(-r['selection_score'],-r['face_short_side_p10'],r['sample_id']))
    result=Selection(24);target=hours*3600
    for r in ranked:
        if result.seconds>=.9*target:break
        result.add(r,'core')
    if result.seconds<.9*target:raise ValueError('Insufficient open core; no silent threshold relaxation')
    for r in sorted((r for r in ranked if r['head_shake_count']),key=lambda r:(-r['face_short_side_p10'],-r['selection_score'],r['sample_id'])):
        if result.reason(r) is None:
            if result.seconds+r['duration_s']>target:raise ValueError('Rare supplement exceeds target')
            result.add(r,'supplement:head_shake')
    for c in sorted((c for c in CATEGORIES if c!='head_shake'),key=lambda c:sum(r[c+'_count']>0 for r in ranked)):
        for r in ranked:
            if result.counts[c]>=2000:break
            if r[c+'_count'] and r['face_short_side_p10']>=192 and result.reason(r) is None:
                if result.seconds+r['duration_s']>target:raise ValueError('Rare supplement exceeds target')
                result.add(r,'supplement:'+c)
        if result.counts[c]<2000:raise ValueError('Seven-class reference coverage quota unavailable')
    for r in ranked:
        if result.seconds>=target:break
        result.add(r,'core_fill')
    if result.seconds<target or any(not result.counts[c] for c in CATEGORIES):raise ValueError('Incomplete open subset')
    return result.rows,dict(target_hours=hours,eligible_candidates=len(eligible),core_fraction=.9,max_per_pair=24,
        floors=dict(core_face_p10=160,supplement_face_p10=192,head_shake_face_p10=160,valid_fraction=.98),
        score='0.5*min(face_p10/384,1)+0.3*min(capped_original_description_variety/7,1)+0.2*min(capped_open_density/12,1)',
        diversity_is_original_wording_not_semantic_classes=True,
        text_cap='Normalize with the previous literal() only for ranking; cap each non-smile wording at 3 and existing smile-only annotations collectively at 1',
        canonical_reference_minimum=2000,head_shake_supplement='all eligible nonduplicate candidates',
        original_descriptions_unchanged=True,new_semantic_classification=False,seven_class_filter=False,
        every_selected_clip_retains_all_original_visual_observations=True,
        all_global_semantic_types_claimed=False)


def main():
    p=argparse.ArgumentParser()
    for name in ['full','stats','report','catalog','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--hours',type=float,default=300);p.add_argument('--workers',type=int,default=48);a=p.parse_args()
    out=a.output.resolve();full=a.full.resolve();view=full/'view';prepared=read(out/'PREPARE_COMPLETE.json')
    if (out/'COMPLETE.json').exists():raise ValueError('Completed dataset is immutable')
    details={}
    for f in sorted((out/'prepared').glob('*.stats.jsonl')):
        marker=read(f.with_name(f.name.replace('.stats.jsonl','.done.json')))
        if marker.get('statistics_version')!=2 or sha(f)!=marker['files'][f.name]:raise ValueError('Stale open ranking data')
        with f.open('rb') as stream:
            for line in stream:
                r=loads(line);details[r['sample_id']]=r
    if len(details)!=prepared['samples']:raise ValueError('Prepared scope mismatch')
    db=sqlite3.connect(f'file:{a.stats}/stats.sqlite?mode=ro',uri=True);db.row_factory=sqlite3.Row
    db.execute('ATTACH DATABASE ? AS post',(f'file:{a.report}/sample_index.sqlite?mode=ro',))
    pool=[]
    for item in db.execute("SELECT s.*,p.canonical_mask FROM samples s JOIN post.samples p USING(sample_id) WHERE s.split='train' AND s.label_issue_flag=0 AND s.face_valid_fraction>=.98 AND s.face_short_side_p10>=160 AND p.training_admitted=1"):
        r=dict(item);r.update(details.pop(r['sample_id']))
        # Presence comes from current canonical labels, never the pre-silence count.
        for i,c in enumerate(CATEGORIES):r[c+'_count']=int(bool(r['canonical_mask']&(1<<i)))
        pool.append(r)
    db.close()
    if details:raise ValueError('Unmatched prepared candidates')
    rows,policy=choose(pool,a.hours);del pool
    pairs,split_counts=audit_catalog(read(a.catalog))
    for r in rows:
        if pairs[r['pair']]['split']!='train' or pairs[r['pair']]['group_id']!=r['group_id']:raise ValueError('Catalog mismatch')
    split=dict(catalog_sha256=sha(a.catalog),retained_catalog_pairs=split_counts,identity_intersections=0,
        eval_manifests=audit_eval_manifests(a.catalog.parent/'manifests',pairs,rows))
    atomic_json(out/'split_audit.json',split);atomic_json(out/'selection_policy.json',policy)
    atomic_json(out/'selection_initial.json',rows)
    print(json.dumps(dict(phase='selected_open',samples=len(rows),hours=sum(r['duration_s'] for r in rows)/3600)),flush=True)
    for name in ['manifests','labels/canonical','audio_postprocess/recipes','cache']:(out/name).mkdir(exist_ok=True,parents=True)
    cfg=copy.deepcopy(read(view/'config.json'));base=Path(cfg['data']['audio_postprocess']['base_output']);cfg['output']=str(out)
    atomic_json(out/'media_cache_config.json',cfg)
    (out/'audio_postprocess/policy_manifest.json').write_bytes((view/'audio_postprocess/policy_manifest.json').read_bytes())
    for name in ['input','text','audio_postprocess']:
        dest=out/'cache'/name
        if not dest.exists():dest.symlink_to(view/'cache'/name,target_is_directory=True)
        elif dest.resolve()!=(view/'cache'/name).resolve():raise ValueError('Unexpected cache link')
    scope=sqlite3.connect(f'file:{full}/scope.sqlite?mode=ro',uri=True);scope.row_factory=sqlite3.Row
    wanted={r['sample_id'] for r in rows};media={}
    for s in scope.execute('SELECT sid,obs,target FROM samples'):
        if s['sid'] in wanted:media[s['sid']]=(loads(s['obs']),loads(s['target']))
    scope.close()
    if set(media)!=wanted:raise ValueError('Missing scope records')
    def export(r):
        r=dict(r);sid=r['sample_id'];o,t=media[sid];canonical=view/'labels/canonical'/f'{sid}.json';label=read(canonical)
        r['current_canonical_sha256']=sha(canonical);r.update(event_statistics(label,r['duration_s']))
        ref=export_one((r,o,t,full,out,base));ref['seven_class_reference_cache']=ref['cache'].pop('events')
        with Path(r['open_label_file']).open('rb') as f:f.seek(r['open_label_offset']);raw=f.read(r['open_label_size'])
        if hashlib.sha256(raw).hexdigest()!=r['open_label_sha256']:raise ValueError('Prepared open labels changed')
        opened=loads(raw)
        if opened['sample_id']!=sid or opened['new_semantic_classification']:raise ValueError('Wrong open labels')
        record=read(full/'records'/f'{sid}.json')
        if [e for e in opened['events'] if e['modality']=='audio']!=sorted(record['audio_events'],key=lambda e:(e['start'],e['end'],e['event_id'])):
            raise ValueError('Filtered audio event changed')
        ref['open_label_sha256']=r['open_label_sha256'];ref['source_evidence']=opened['source_evidence']
        return r,ref,raw
    index=sqlite3.connect(out/'open_label_index.sqlite');index.execute('CREATE TABLE IF NOT EXISTS labels(sample_id TEXT PRIMARY KEY,offset INT,size INT,sha256 TEXT)');index.execute('DELETE FROM labels');index.commit()
    final=[]
    with (out/'dataset.jsonl').open('w') as refs,(out/'integrated_labels.jsonl').open('wb') as labels,ThreadPoolExecutor(a.workers) as workers:
        for begin in range(0,len(rows),1024):
            batch=[]
            for r,ref,raw in workers.map(export,rows[begin:begin+1024]):
                offset=labels.tell();labels.write(raw);batch.append((r['sample_id'],offset,len(raw),hashlib.sha256(raw).hexdigest()))
                refs.write(json.dumps(ref,ensure_ascii=False)+'\n');final.append(r)
            index.executemany('INSERT INTO labels VALUES(?,?,?,?)',batch);index.commit()
            atomic_json(out/'progress.json',dict(phase='export_open_dataset',samples=len(final),total=len(rows)))
    index.close();rows=final
    atomic_json(out/'selection.json',rows)
    atomic_json(out/'manifests/observations.json',[media[r['sample_id']][0] for r in rows])
    atomic_json(out/'manifests/targets.json',[media[r['sample_id']][1] for r in rows])
    with (out/'samples.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (out/'sample_ids.txt').write_text(''.join(r['sample_id']+'\n' for r in rows))
    summary=summarize(rows);summary['seven_class_reference_events']=summary.pop('events')
    summary['open_events']=sum(r['open_event_count'] for r in rows)
    summary['open_extra_visual_events']=sum(r['open_extra_visual_count'] for r in rows)
    summary['open_duplicate_observations_removed']=sum(r['open_duplicate_observations_removed'] for r in rows)
    summary['below_224_face_p10']=summary.pop('below_core_face_floor');summary['below_core_face_floor']=sum(r['face_short_side_p10']<160 for r in rows)
    summary['new_semantic_classification']=False;atomic_json(out/'summary.json',summary)
    atomic_json(out/'dataset_contract.json',dict(schema='listen-open-all-verbatim-subset-v1',samples=len(rows),duration_s=sum(r['duration_s'] for r in rows),
        split='train',label_mode='all_original_descriptions_with_existing_dedup',label_file='integrated_labels.jsonl',label_index='open_label_index.sqlite',
        reader='listen_benchmark.open_subset_reader.load_open_training_sample',media_config='media_cache_config.json',
        fixed_seven_event_loader_supported=False,seven_class_labels_are_cache_identity_references_only=True,
        new_semantic_classification=False,source_labels_rewritten=False,processed_wav_created=False,training_started=False,
        full_postprocess=str(full),source_prepare_sha256=sha(out/'PREPARE_COMPLETE.json')))
    atomic_json(out/'export_complete.json',dict(samples=len(rows),hours=summary['hours'],training_started=False))
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
