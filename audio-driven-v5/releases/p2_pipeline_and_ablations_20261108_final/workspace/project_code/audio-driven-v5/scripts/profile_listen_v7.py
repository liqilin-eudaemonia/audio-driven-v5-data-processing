"""Measure existing screening without altering the frozen run or its thresholds."""
import cProfile
import io
import json
import pstats
import time
from pathlib import Path
import torch
import run_listen_v7_bulk as bulk


def main():
    torch.set_num_threads(2)
    out=bulk.ROOT/'experiments/p3_listen_v7_performance';out.mkdir(parents=True,exist_ok=True)
    production=bulk.OUT
    catalog=bulk.read(production/'catalog.json')['rows']
    pair='naturalistic_V03_S0847_I00000064'
    item=next(r for r in catalog if r['pair']==pair)
    audit=bulk.read(bulk.ROOT/'experiments/p3_listen_v7_validation/audit'/(pair+'.json'))
    bulk.OUT=out;bulk.preview.OUT=out;bulk.subturns.OUT=out
    begin=time.perf_counter();norm=bulk.normalize(item,audit['clips']);norm_s=time.perf_counter()-begin
    ast=bulk.EventModel(bulk.MODEL,batch_size=16);panns=bulk.PannsModel(batch_size=16)
    # Warm both models, then measure actual clip processing including CPU work.
    rejects=bulk.read(production/'human_rejections.json')['rows']
    positives=bulk.read(production/'human_retained.json')['rows']
    bulk.scan(audit['clips'][0],norm,ast,panns,rejects,positives)
    profile=cProfile.Profile();profile.enable();begin=time.perf_counter()
    results=[bulk.scan(r,norm,ast,panns,rejects,positives) for r in audit['clips'][:12]]
    elapsed=time.perf_counter()-begin;profile.disable()
    profile.dump_stats(str(out/'screening.prof'))
    stream=io.StringIO();pstats.Stats(profile,stream=stream).sort_stats('cumulative').print_stats(35)
    (out/'screening_profile.txt').write_text(stream.getvalue())
    report=dict(clips=len(results),seconds=elapsed,seconds_per_clip=elapsed/len(results),normalization_cached_seconds=norm_s)
    bulk.write(out/'baseline.json',report)
    print(json.dumps(report),flush=True);print(stream.getvalue(),flush=True)


if __name__=='__main__':main()
