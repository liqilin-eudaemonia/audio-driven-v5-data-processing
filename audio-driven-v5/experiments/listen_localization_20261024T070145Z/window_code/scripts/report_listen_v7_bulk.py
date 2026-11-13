"""Incremental V7 hours, onset reserves and 2/3-second threshold comparison."""
import argparse
from collections import Counter
import fcntl
import json
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[1]/'data_corpora/seamless_listen_v7_full'


def duration_bin(value):
    return next(label for ceiling,label in [(3,'2-3s'),(4,'3-4s'),(6,'4-6s'),(8,'6-8s'),(10.000001,'8-10s')] if value<ceiling)


def export_manifests(expected):
    """Compact source-backed manifests; natural intervals are not training windows."""
    folder=ROOT/'manifests';folder.mkdir(exist_ok=True)
    handles={};counts=Counter();hours=Counter()
    try:
        for pair in sorted(expected):
            path=ROOT/'results'/(pair+'.json')
            if not path.exists():continue
            result=json.loads(path.read_text())
            if not result.get('complete'):continue
            for row in result['clips']:
                name='_'.join([row['split'],row['effective_decision'],row['admission_pool']])
                if name not in handles:handles[name]=(folder/(name+'.jsonl.pending')).open('w')
                compact={k:v for k,v in row.items() if k not in ('qc3','normalized_audio','human_decisions')}
                compact['normalized_audio']={role:{k:track[k] for k in ('path','sha256','gain_db')}
                                             for role,track in row['normalized_audio'].items()}
                compact['quality_evidence_file']=str(path)
                handles[name].write(json.dumps(compact,ensure_ascii=False)+'\n')
                counts[name]+=1;hours[name]+=row['unique_seconds']/3600
    finally:
        for handle in handles.values():handle.close()
    for name in handles:(folder/(name+'.jsonl.pending')).replace(folder/(name+'.jsonl'))
    (folder/'index.json').write_text(json.dumps(dict(counts=dict(counts),unique_hours=dict(hours),
        training_ready=False,note='Source-backed intervals. No 8k+1 window truncation; model adapter pending.'),indent=2)+'\n')


def report(watch=False):
    catalog=json.loads((ROOT/'catalog.json').read_text());expected={r['pair'] for r in catalog['rows']}
    counts=Counter();hours=Counter();buckets=Counter();bucket_hours=Counter();reasons=Counter();reason_hours=Counter()
    thresholds=Counter();voices=Counter();source_hours=Counter();seen=set();source_rejected=0
    with (ROOT/'report.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            for path in (ROOT/'results').glob('*.json'):
                if path.stem in seen:continue
                if path.stem not in expected:raise ValueError('Unexpected result '+path.stem)
                result=json.loads(path.read_text())
                if not result.get('complete'):continue
                seen.add(path.stem);source_rejected+=int(not result['source_ok'])
                if result.get('source_seconds') is not None:source_hours[result['split']]+=result['source_seconds']/3600
                for row in result['clips']:
                    key='/'.join([row['split'],row['effective_decision'],row['admission_pool']])
                    h=row['unique_seconds']/3600
                    counts[key]+=1;hours[key]+=h;voices[key]+=int(row['listener_human_evidence'])
                    bucket=key+'/'+duration_bin(row['duration_s']);buckets[bucket]+=1;bucket_hours[bucket]+=h
                    if row['effective_decision']=='keep':
                        for minimum in (2,3):
                            if row['duration_s']<minimum-1e-7:continue
                            thresholds[f"{row['split']}/min{minimum}s/including_onset_reserve"]+=h
                            if row['admission_pool']=='primary':thresholds[f"{row['split']}/min{minimum}s/primary"]+=h
                for r in result['rejected']:
                    values=r.get('rejection',r.get('reason','unknown'))
                    for reason in values if isinstance(values,list) else [values]:
                        reasons[reason]+=1
                        if 'start' in r and 'end' in r:reason_hours[reason]+=max(0,r['end']-r['start'])/3600
            errors={p.stem for p in (ROOT/'errors').glob('*.json')}-seen
            nodes=[json.loads((ROOT/f'node_{n}.json').read_text()) if (ROOT/f'node_{n}.json').exists() else {'state':'not_started'} for n in range(3)]
            extra_groups=[json.loads(p.read_text()) for p in
                          sorted((ROOT/'scheduling_v2/extra_groups').glob('group_*_node_*.json'))]
            finished=(all(n['state']=='finished' for n in nodes)
                      and all(g['state']=='finished' for g in extra_groups)
                      and not (ROOT/'scheduling_v2/FINALIZATION_HOLD').exists())
            value=dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),total_pairs=len(expected),
                complete_pairs=len(seen),error_pairs=len(errors),pending_pairs=len(expected-seen-errors),
                source_rejected_pairs=source_rejected,counts=dict(counts),unique_hours=dict(hours),
                duration_bin_counts=dict(buckets),duration_bin_hours=dict(bucket_hours),
                keep_threshold_comparison_hours=dict(thresholds),model_vocal_counts=dict(voices),
                successfully_loaded_source_hours=dict(source_hours),rejection_reasons=dict(reasons),
                rejection_known_interval_hours=dict(reason_hours),workers_finished=finished,
                node_states=nodes,extra_worker_groups=extra_groups,training_ready=False,
                note='Primary excludes listener activity in first 0.5s. Onset reserve is NOT admitted to primary. '
                     '2s/3s comparison filters one canonical nonoverlapping segmentation. Review and exclude never counted as keep. '
                     'Natural intervals only; model frame adapter and rendered-video validation remain separate.')
            target=ROOT/'screening_summary.json';tmp=target.with_suffix('.pending')
            tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');tmp.replace(target)
            print(json.dumps({k:value[k] for k in ('utc','complete_pairs','error_pairs','pending_pairs','keep_threshold_comparison_hours')}),flush=True)
            if finished:export_manifests(expected)
            if not watch or finished:break
            time.sleep(60)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true');report(p.parse_args().watch)
