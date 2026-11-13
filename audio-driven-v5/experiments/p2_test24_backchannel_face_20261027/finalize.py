"""Postprocess independent test candidates and export a disjoint 24-clip panel."""
import copy,csv,hashlib,json,sys,tarfile,time
from collections import defaultdict,Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];RUN=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from p2_listen.contracts import atomic_json,sha,digest
from listen_benchmark.full_fusion import read
from listen_benchmark.backchannel_face_stats import sample_stats,CATEGORIES
from listen_benchmark.postprocessed_subset import event_statistics
from listen_benchmark.train_subset import Selection,score,summarize,audit_catalog,identities
from listen_audio_postprocess.bulk import official
from listen_benchmark.official_audio_gate import make_gate
from listen_benchmark.official_event_alignment import align_events
from listen_benchmark.chinese_asr_filter import filter_labels
from listen_audio_postprocess.recipes import make_recipe


def postprocess_candidates():
    shortlist={r['sample_id']:r for r in read(RUN/'annotation_shortlist.json')}
    grouped=defaultdict(list)
    for path in sorted((RUN/'fusion/labels').glob('*.jsonl')):
        marker=read(RUN/f'fusion/state/fusion/{path.stem}.json')
        if sha(path)!=marker['files'][f'labels/{path.name}']['sha256']:raise ValueError('Fused labels changed')
        for line in path.open('rb'):
            c=json.loads(line);s=c['source'];grouped[s['source']['archive']].append((c,shortlist[c['sample_id']],hashlib.sha256(line).hexdigest()))
    if sum(map(len,grouped.values()))!=len(shortlist):raise ValueError('Not all test candidates annotated')
    def archive_work(item):
        archive,values=item;members={str(Path(c['source']['source']['member']).with_suffix('.json')) for c,_,_ in values};metadata={}
        with tarfile.open(archive,'r:') as tar:
            for m in tar:
                if m.name in members:
                    body=tar.extractfile(m).read();metadata[m.name]=(json.loads(body),hashlib.sha256(body).hexdigest())
                    if len(metadata)==len(members):break
        if set(metadata)!=members:raise ValueError('Missing official test listener metadata')
        out=[]
        for c,s,source_sha in values:
            sid=c['sample_id'];dest=RUN/f'postprocessed/{sid}.json'
            if dest.exists():out.append(read(dest));continue
            source=c['source'];member=str(Path(source['source']['member']).with_suffix('.json'));raw,checksum=metadata[member]
            rate=source['normalized_audio']['sample_rate'];start=round(source['source_start_s']*rate);end=round(source['source_end_s']*rate)
            duration=(end-start)/rate
            off=official(raw,start/rate,duration);_,gate=make_gate(off,end-start,rate)
            aligned=align_events([e for e in c['events'] if e['modality']=='audio'],off,gate)
            result,_,_=filter_labels(aligned,gate)
            final=copy.deepcopy(c);final['events']=[e for e in c['events'] if e['modality']!='audio']+result['events']
            final['audio_postprocess']=dict(policy='listener-audio-overlay-v1',text_rewritten=False,visual_events_unchanged=True,
                original_integrated_sha256=source_sha,excluded_audio_events=result['excluded_audio_events'],language_filter=result['language_filter'])
            stats=sample_stats(final,s['source'])
            projection=[]
            for e in final['events']:
                for cat in e.get('categories',[]):
                    if cat not in CATEGORIES:continue
                    for i,x in enumerate(e.get('intervals',[dict(start=e['start'],end=e['end'])])):
                        projection.append(dict(event_id=e['event_id']+f':{cat}:{i}',source=dict(event_id=e['event_id']),type=cat,start=x['start'],end=x['end']))
            stats.update(event_statistics(dict(events=projection),duration))
            stats['duration_s']=duration
            stats['selection_score'],stats['capped_event_count'],stats['capped_density_per_10s']=score(stats)
            value=dict(sample_id=sid,integrated=final,statistics=stats,official_listener=off,
                extra_mute_intervals=[dict(start=a/rate,end=b/rate) for a,b in result['language_filter']['mute_intervals_samples']],
                provenance=dict(official_archive=archive,official_member=member,official_sha256=checksum,source_integrated_sha256=source_sha))
            atomic_json(dest,value);out.append(value)
        return out
    result=[]
    with ThreadPoolExecutor(max_workers=12) as pool:
        for values in pool.map(archive_work,grouped.items()):result.extend(values)
    return result,shortlist


def select(rows,n=24):
    rows=[r for r in rows if not r['label_issue_flag'] and r['non_smile_event_count']>=1 and r['face_valid_fraction']>=.98 and r['face_short_side_p10']>=160]
    ranked=sorted(rows,key=lambda r:(-r['selection_score'],-r['face_short_side_p10'],r['sample_id']))
    result=Selection(8)
    for r in ranked:
        if len(result.rows)>=21:break
        if r['face_short_side_p10']>=224:result.add(r,'core')
    if len(result.rows)!=21:raise ValueError('Insufficient test core')
    for c in sorted(CATEGORIES,key=lambda c:sum(r[c+'_count']>0 for r in rows)):
        if result.counts[c]:continue
        floor=160 if c=='head_shake' else 192
        found=next((r for r in ranked if r[c+'_count'] and r['face_short_side_p10']>=floor and result.reason(r) is None),None)
        if found is None:raise ValueError('Need more independent test annotations for '+c)
        if len(result.rows)>=n:raise ValueError('Coverage needs more reserved slots')
        result.add(found,'supplement:'+c)
    for r in ranked:
        if len(result.rows)>=n:break
        if r['face_short_side_p10']>=224:result.add(r,'core_fill')
    if len(result.rows)!=n or any(not result.counts[c] for c in CATEGORIES):raise ValueError('Incomplete test panel')
    return result.rows


def main():
    while not all((RUN/f'fusion/state/fusion/{i:04d}.json').exists() for i in range(16)):
        if (RUN/'STOP').exists():return
        time.sleep(5)
    values,shortlist=postprocess_candidates();byid={r['sample_id']:r for r in values}
    rows=select([r['statistics'] for r in values]);atomic_json(RUN/'selection.json',rows)
    pairs,_=audit_catalog(read(ROOT/'data_corpora/seamless_listen_local_full/catalog.json'))
    train=read(ROOT/'experiments/p2_train50h_postprocessed_reselected_20261027/selection.json')
    train_keys=set().union(*(identities(pairs[r['pair']]) for r in train))
    if any(r['split']!='test' or train_keys & identities(pairs[r['pair']]) for r in rows):raise ValueError('Train/test leakage')
    summary=summarize(rows);summary['split']='test';summary['labels_are_weak']=True;summary['formal_evaluation_reference']=False
    atomic_json(RUN/'summary.json',summary)
    # Export the existing 512px media representation, preserving the actual test split.
    from listen_benchmark.feature_pipeline import build_source_index,export_media
    (RUN/'source_index').mkdir(exist_ok=True)
    atomic_json(RUN/'config.json',dict(project=str(ROOT),raw_run=str(RUN/'raw_run'),fusion_run=str(RUN/'fusion')))
    build_source_index(RUN)
    def export(row):
        sid=row['sample_id'];v=byid[sid];c=v['integrated'];media=export_media(RUN,c);o,t=media['observation'],media['target']
        if o['official_split']!='test' or o['experiment_split']!='test':raise ValueError('Test split lost')
        o['grouping_ids']=sorted(identities(pairs[row['pair']]))
        recipe=make_recipe(sid,t['target_audio'],t['media_hashes']['target_audio'],v['official_listener'],v['extra_mute_intervals'],v['provenance'])
        atomic_json(RUN/f'audio_postprocess/recipes/{sid}.json',recipe)
        return o,t,dict(sample_id=sid,recipe_sha256=digest(recipe),media=media['meta']['sha256'],source_line_sha256=row['source_line_sha256'])
    with ThreadPoolExecutor(max_workers=8) as pool:exported=list(pool.map(export,rows))
    atomic_json(RUN/'manifests/observations.json',[x[0] for x in exported]);atomic_json(RUN/'manifests/targets.json',[x[1] for x in exported]);atomic_json(RUN/'media_references.json',[x[2] for x in exported])
    with (RUN/'integrated_labels.jsonl').open('w') as f:
        for r in rows:f.write(json.dumps(byid[r['sample_id']]['integrated'],ensure_ascii=False)+'\n')
    with (RUN/'samples.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (RUN/'sample_ids.txt').write_text(''.join(r['sample_id']+'\n' for r in rows))
    # Test outputs are weak selection annotations, never promoted to human references.
    atomic_json(RUN/'dataset_contract.json',dict(split='test',samples=24,train_identity_overlap=0,seven_class_coverage_verified=True,
        labels_are_weak=True,human_ground_truth=False,training_admitted=False,formal_metric_reference_admitted=False,
        audio_policy='Original clip plus virtual-silence recipe; no processed WAV',feature_caches_created=False))
    from listen_audio_postprocess.recipes import read_listener_audio
    for o,t,_ in exported:
        recipe=read(RUN/f'audio_postprocess/recipes/{o["sample_id"]}.json');wave,rate,meta=read_listener_audio(recipe)
        if abs(len(wave)/rate-o['requested_duration_s'])>1/rate+1e-6:raise ValueError('Virtual audio duration mismatch')
    names=['summary.json','selection.json','samples.csv','sample_ids.txt','integrated_labels.jsonl','manifests/observations.json','manifests/targets.json','media_references.json','dataset_contract.json']
    atomic_json(RUN/'COMPLETE.json',dict(status='passed',samples=24,files={n:sha(RUN/n) for n in names},training_started=False))
    atomic_json(RUN/'status.json',dict(status='complete',samples=24,seven_class_coverage_verified=True,train_identity_overlap=0,weak_annotations=True))
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
