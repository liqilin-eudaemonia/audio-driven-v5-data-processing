"""Write separate, non-admitted manifests for the measured recovery pools."""
import collections,concurrent.futures,json
import evaluate_listen_v8_recovery as e

def main():
    e.baseline.install();e.baseline.full.verify()
    dest=e.OUT/'manifests';dest.mkdir(exist_ok=True);handles={};counts=collections.Counter();hours=collections.Counter()
    paths=sorted((e.BASE/'results').glob('*.json'))
    try:
        with concurrent.futures.ThreadPoolExecutor(4) as pool:
            for result in pool.map(e.read,paths):
                for row in result['clips']:
                    a=e.recovery.analysis(row)
                    kind=('boundary_candidate_pending_review' if a['boundary_recovery_candidate'] else
                          'identity_only_pending_review' if a['identity_only_review_potential'] else None)
                    if kind is None:continue
                    key=row['split']+'_'+row['admission_pool']+'_'+kind
                    if key not in handles:handles[key]=(dest/(key+'.jsonl.pending')).open('w')
                    compact={k:v for k,v in row.items() if k not in ('qc3','qc4')}
                    compact.update(recovery=a,recovery_disposition=kind,training_admitted=False,
                        quality_evidence_file=str(e.BASE/'results'/(result['pair']+'.json')))
                    handles[key].write(json.dumps(compact,ensure_ascii=False)+'\n');counts[key]+=1;hours[key]+=row['duration_s']/3600
    finally:
        for f in handles.values():f.close()
    for key in handles:(dest/(key+'.jsonl.pending')).replace(dest/(key+'.jsonl'))
    e.write(dest/'index.json',dict(utc=e.baseline.full.now(),snapshot_pairs=len(paths),counts=dict(counts),hours=dict(hours),
        baseline_contract_sha256=e.sha(e.BASE/'contract.json'),recovery_rule_sha256=e.sha(e.ROOT/'scripts/listen_v8_recovery.py'),
        manifest_sha256={key:e.sha(dest/(key+'.jsonl')) for key in handles},training_admitted=False))
    print(json.dumps(dict(counts=dict(counts),hours=dict(hours))),flush=True)

if __name__=='__main__':main()
