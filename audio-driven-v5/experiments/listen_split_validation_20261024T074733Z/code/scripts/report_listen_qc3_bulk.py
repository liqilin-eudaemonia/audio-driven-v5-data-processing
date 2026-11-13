"""Incremental reporting only; does not alter frozen screening or admission."""
import argparse
from collections import Counter
import fcntl
import json
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[1]/'data_corpora/seamless_listen_qc3_full'


def main(watch=False):
    catalog=json.loads((ROOT/'catalog.json').read_text())
    expected={r['pair'] for r in catalog['rows']}
    counts=Counter();hours=Counter();media_hours=Counter();voices=Counter();reasons=Counter()
    seen=set();source_rejected=0
    with (ROOT/'report.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            for path in (ROOT/'results').glob('*.json'):
                if path.stem in seen:
                    continue
                if path.stem not in expected:
                    raise ValueError('Unexpected source result: '+path.stem)
                result=json.loads(path.read_text())
                if not result.get('complete'):
                    continue
                seen.add(path.stem);source_rejected+=int(not result['source_ok'])
                for row in result['clips']:
                    key=row['split']+'/'+row['effective_decision']
                    counts[key]+=1;hours[key]+=row['unique_seconds']/3600
                    media_hours[key]+=row['duration_s']/3600
                    voices[key]+=int(row['listener_human_evidence'])
                for r in result['rejected']:
                    # Natural units carry both a boundary kind (reason) and the
                    # actual exclusion explanation (rejection); count the latter.
                    values=r.get('rejection',r.get('reason','unknown'))
                    for reason in values if isinstance(values,list) else [values]:
                        reasons[reason]+=1
            errors={p.stem for p in (ROOT/'errors').glob('*.json')}-seen
            nodes=[]
            for n in range(3):
                p=ROOT/f'node_{n}.json'
                nodes.append(json.loads(p.read_text()) if p.exists() else {'state':'not_started'})
            finished=all(n['state']=='finished' for n in nodes)
            value=dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                total_pairs=len(expected),complete_pairs=len(seen),error_pairs=len(errors),
                pending_pairs=len(expected-seen-errors),source_rejected_pairs=source_rejected,
                counts=dict(counts),unique_hours=dict(hours),media_hours=dict(media_hours),
                with_model_vocal_evidence_counts=dict(voices),
                geometry_boundary_rejection_reasons=dict(reasons),
                catalog_excluded_pairs=catalog['excluded_pairs'],
                workers_finished=finished,training_ready=False,
                note='keep is a candidate, review quarantined. Speaker-span unique hours exclude '
                     'frame-alignment silence; media hours reported separately. Test/mixed groups excluded upstream.')
            path=ROOT/'screening_summary.json';tmp=path.with_suffix('.pending')
            tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)
            print(json.dumps(value,ensure_ascii=False),flush=True)
            if not watch or finished:
                break
            time.sleep(60)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true')
    main(p.parse_args().watch)
