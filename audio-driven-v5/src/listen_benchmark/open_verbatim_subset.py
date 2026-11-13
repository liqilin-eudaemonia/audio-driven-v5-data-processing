"""Open-source-description subset: existing visual dedup, unchanged filtered audio.

No new semantic classifier, taxonomy, rewritten description, or synthetic boundary.
"""
from collections import Counter,defaultdict
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sqlite3

from p2_listen.contracts import atomic_json,sha
from .backchannel_face_stats import loads,union_seconds
from .integration_pilot import clusters,key,make_event
from .existing_result_fusion import restore_source_wording
from .response_review import literal


def integrate_open(base,evidence,record):
    if not base['sample_id']==evidence['sample_id']==record['sample_id']:
        raise ValueError('Open fusion source ID mismatch')
    original=evidence['original_events'];visual=[e for e in original if e['modality']=='visual']
    decisions={d['open_event_id']:d for d in evidence['decisions']}
    buckets=defaultdict(list)
    # Existing duplicate() requires identical keys, so independent key buckets
    # are equivalent to the old complete-link algorithm and avoid quadratic work.
    for e in visual:buckets[key(e,decisions)].append(e)
    grouped=[]
    for bucket in buckets.values():grouped.extend(clusters(bucket,decisions))
    grouped.sort(key=lambda g:min((e['start'],e['end'],e['event_id']) for e in g))
    events=[]
    for i,group in enumerate(grouped):
        e=make_event(group,i);e['event_id']=f'open-visual-{i:04d}'
        restore_source_wording(e,original)
        e['original_mapping_statuses']=sorted({x['mapping_status'] for x in group})
        e['source_boundary_methods']={x['event_id']:decisions.get(x['event_id'],{}).get('origin',{}).get('boundary_method') for x in group}
        events.append(e)
    if {oid for e in events for oid in e['source_observation_ids']}!={e['event_id'] for e in visual}:
        raise ValueError('Lost original visual observation')
    audio=deepcopy(record['audio_events']);events+=audio
    if len({e['event_id'] for e in events})!=len(events):raise ValueError('Open event ID collision')
    events.sort(key=lambda e:(e['start'],e['end'],e['event_id']))
    if any(not 0<=e['start']<e['end']<=base['duration_s']+.0011 for e in events):raise ValueError('Invalid original boundary')
    groups=[]
    for e in audio:
        peers=[v['event_id'] for v in events if v['modality']=='visual' and min(v['end'],e['end'])>max(v['start'],e['start'])]
        groups.append(dict(group_id=f'group-{len(groups):04d}',member_event_ids=[e['event_id']]+peers,
                           relation='possible_temporal_cooccurrence',continuous_activity_claim=False))
    members={i for g in groups for i in g['member_event_ids']}
    for e in events:
        if e['event_id'] not in members:groups.append(dict(group_id=f'group-{len(groups):04d}',member_event_ids=[e['event_id']],relation='single_event_reference',continuous_activity_claim=False))
    return dict(schema='listen-open-all-verbatim-v1',sample_id=base['sample_id'],duration_s=base['duration_s'],
        source=base['source'],events=events,response_groups=groups,content_policy='verbatim_sources_no_semantic_replacement',
        original_visual_observation_count=len(visual),original_audio_observation_count=sum(e['modality']=='audio' for e in original),
        raw_observation_ids=[e['event_id'] for e in original],new_semantic_classification=False,
        category_restriction=None,training_admitted=False,human_ground_truth=False,
        negative_coverage={'visual':[],'audio':[]},unlabeled_time_policy='unknown',
        audio_postprocess=dict(policy='listener-audio-overlay-v1',audio_events_unchanged=True,
            record_provenance=record['provenance'],excluded_audio_not_restored=True))


def open_statistics(label):
    events=label['events'];duration=label['duration_s'];descriptions=Counter();v=[];a=[]
    extra=0;removed=0;smile_only=0;other_descriptions=Counter()
    for e in events:
        identity=(e['modality'],literal(e['description']))
        descriptions[identity]+=1;removed+=e.get('duplicate_observations_removed',0)
        if set(e.get('categories',[]))=={'smile'}:smile_only+=1
        else:other_descriptions[identity]+=1
        if e['modality']=='visual' and any(s!='mapped_candidate' for s in e.get('original_mapping_statuses',[])):extra+=1
        spans=[(x['start'],x['end']) for x in e.get('intervals',[dict(start=e['start'],end=e['end'])])]
        (v if e['modality']=='visual' else a).extend(spans)
    # Original text variety is only a capped ranking proxy, never a semantic class count.
    capped=min(smile_only,1)+sum(min(n,3) for n in other_descriptions.values())
    return dict(sample_id=label['sample_id'],open_event_count=len(events),open_visual_count=sum(e['modality']=='visual' for e in events),
        open_audio_count=sum(e['modality']=='audio' for e in events),open_extra_visual_count=extra,
        open_original_description_count=len(descriptions),open_capped_density=10*capped/max(duration,5),
        open_ranking_description_count=len(other_descriptions)+int(smile_only>0),
        open_coverage_seconds=union_seconds(v+a),open_visual_seconds=union_seconds(v),open_audio_seconds=union_seconds(a),
        open_duplicate_observations_removed=removed)


def prepare_shard(job):
    shard,ids,fusion,full,out=job;fusion=Path(fusion);full=Path(full);out=Path(out);wanted=set(ids)
    marker=out/'prepared'/f'{shard}.done.json'
    if marker.exists():
        m=loads(marker.read_bytes())
        for name,h in m['files'].items():
            if sha(out/'prepared'/name)!=h:raise ValueError('Prepared open shard changed')
        if m.get('statistics_version')!=2:
            labels_path=out/'prepared'/f'{shard}.labels.jsonl';stats_path=out/'prepared'/f'{shard}.stats.jsonl'
            with labels_path.open('rb') as f,stats_path.open('w') as stats_out:
                offset=0
                for raw in f:
                    value=open_statistics(loads(raw));value.update(open_label_file=str(labels_path),open_label_offset=offset,open_label_size=len(raw),open_label_sha256=hashlib.sha256(raw).hexdigest())
                    stats_out.write(json.dumps(value,ensure_ascii=False)+'\n');offset+=len(raw)
            m['files'][stats_path.name]=sha(stats_path);m['statistics_version']=2;atomic_json(marker,m)
        return m
    original_path=fusion/'labels'/f'{shard}.jsonl';evidence_path=fusion/'evidence'/f'{shard}.jsonl'
    fm=loads((fusion/'state/fusion'/f'{shard}.json').read_bytes())
    for p in (original_path,evidence_path):
        if sha(p)!=fm['files'][f'{p.parent.name}/{p.name}']['sha256']:raise ValueError('Original fused source changed')
    originals={};line_hashes={}
    with original_path.open('rb') as f:
        for line in f:
            r=loads(line)
            if r['sample_id'] in wanted:originals[r['sample_id']]=r;line_hashes[r['sample_id']]=hashlib.sha256(line).hexdigest()
    labels_path=out/'prepared'/f'{shard}.labels.jsonl';stats_path=out/'prepared'/f'{shard}.stats.jsonl';seen=set()
    with evidence_path.open('rb') as f, labels_path.open('wb') as labels_out,stats_path.open('w') as stats_out:
        offset=0
        for n,line in enumerate(f,1):
            ev=loads(line);sid=ev['sample_id'];start=offset;offset+=len(line)
            if sid not in wanted:continue
            base=originals[sid];record_path=full/'records'/f'{sid}.json';record=loads(record_path.read_bytes())
            if record['provenance']['source_integrated_sha256']!=line_hashes[sid]:raise ValueError('Postprocessed audio source mismatch')
            label=integrate_open(base,ev,record)
            label['source_evidence']=dict(file=str(evidence_path),line_1based=n,offset=start,size=len(line),sha256=hashlib.sha256(line).hexdigest())
            label['source_postprocess_record']=dict(file=str(record_path),sha256=sha(record_path))
            raw=json.dumps(label,ensure_ascii=False,separators=(',',':')).encode()+b'\n';loc=labels_out.tell();labels_out.write(raw)
            stats=open_statistics(label);stats.update(open_label_file=str(labels_path),open_label_offset=loc,open_label_size=len(raw),open_label_sha256=hashlib.sha256(raw).hexdigest())
            stats_out.write(json.dumps(stats,ensure_ascii=False)+'\n');seen.add(sid)
    if seen!=wanted:raise ValueError('Missing candidate source annotations')
    m=dict(samples=len(seen),statistics_version=2,files={p.name:sha(p) for p in (labels_path,stats_path)})
    atomic_json(marker,m);return m


def prepare(fusion,full,stats,report,out,workers=32):
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=True);(out/'prepared').mkdir(exist_ok=True)
    db=sqlite3.connect(f'file:{stats}/stats.sqlite?mode=ro',uri=True)
    db.execute('ATTACH DATABASE ? AS post',(f'file:{report}/sample_index.sqlite?mode=ro',))
    by_shard=defaultdict(list)
    for sid,label_file in db.execute("SELECT s.sample_id,s.label_file FROM samples s JOIN post.samples p USING(sample_id) WHERE s.split='train' AND s.label_issue_flag=0 AND s.face_valid_fraction>=.98 AND s.face_short_side_p10>=160 AND p.training_admitted=1"):
        by_shard[Path(label_file).stem].append(sid)
    db.close()
    args=[(s,ids,str(Path(fusion).resolve()),str(Path(full).resolve()),str(out)) for s,ids in sorted(by_shard.items())]
    total=sum(map(len,by_shard.values()));done=0
    with ProcessPoolExecutor(workers) as pool:
        for result in pool.map(prepare_shard,args):
            done+=result['samples'];atomic_json(out/'progress.json',dict(phase='prepare_original_open_labels',samples=done,total=total))
    atomic_json(out/'PREPARE_COMPLETE.json',dict(samples=done,shards=len(args),new_semantic_classification=False,
        source_full=str(Path(full).resolve()),source_fusion=str(Path(fusion).resolve()),code_sha256=sha(__file__)))
