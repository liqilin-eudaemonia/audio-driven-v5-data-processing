"""Cold-normalization serial/parallel comparison with waveform and audit equality."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time
import run_listen_v7_bulk as bulk


def main():
    production=bulk.OUT;catalog={r['pair']:r for r in bulk.read(production/'catalog.json')['rows']}
    chosen=[]
    for p in sorted((production/'results').glob('*.json')):
        pair=p.stem
        if (bulk.ROOT/'data_corpora/seamless_listen_qc3_full/normalized_pairs'/(pair+'.json')).exists():continue
        if (bulk.preview.OLD/'normalized_pairs'/(pair+'.json')).exists():continue
        r=bulk.read(p)
        if r.get('complete') and r.get('clips'):
            chosen.append(catalog[pair])
        if len(chosen)==4:break
    assert len(chosen)==4
    root=bulk.ROOT/'experiments/p3_listen_v7_performance/parallel_check'
    results={};timings={}
    for mode in ('serial','parallel'):
        bulk.OUT=root/mode;bulk.preview.OUT=bulk.OUT;bulk.subturns.OUT=bulk.OUT
        bulk.write(bulk.OUT/'human_rejections.json',bulk.read(production/'human_rejections.json'))
        start=time.perf_counter()
        if mode=='serial':result=[bulk.prepare(item) for item in chosen]
        else:
            with ThreadPoolExecutor(max_workers=4) as pool:result=list(pool.map(bulk.prepare,chosen))
        timings[mode]=time.perf_counter()-start;results[mode]=result
    for (a,x),(b,y) in zip(results['serial'],results['parallel']):
        assert a==b,'Planning changed with concurrency'
        assert [t['sha256'] for t in x['tracks']]==[t['sha256'] for t in y['tracks']],'Normalized PCM changed'
        assert [t['gain_db'] for t in x['tracks']]==[t['gain_db'] for t in y['tracks']]
    summary=dict(pairs=[r['pair'] for r in chosen],seconds=timings,
        speedup=timings['serial']/timings['parallel'],exact_audit_and_pcm_sha256_match=True,
        note='Four real source pairs, CPU preparation benchmark only; not a full-run speedup guarantee.')
    bulk.write(root/'validation.json',summary);print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
