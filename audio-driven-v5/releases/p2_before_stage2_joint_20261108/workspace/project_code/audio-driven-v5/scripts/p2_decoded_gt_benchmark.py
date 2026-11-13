"""CPU parity/timing and bounded prewarming for a fixed evaluation pool only."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT.parent/'OmniResponse')]


def main():
    p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True)
    p.add_argument('--workers',type=int,default=4);p.add_argument('--prewarm',action='store_true')
    a=p.parse_args();w=a.work.resolve()
    import cv2,numpy as np
    cv2.setNumThreads(1)
    from p2_testsets_eval.decoded_gt import DecodedGTCache
    from evaluation.seamless import reader
    from evaluation.video.paired_metrics.core import pixel_scores
    protocol=json.loads((w/'protocol.json').read_text())
    refs=[json.loads(line) for line in (Path(protocol['datasets']['seamless']['prepared'])/'references.jsonl').read_text().splitlines()]
    cache=DecodedGTCache(w/'decoded_gt_cache')
    records=[]
    # Cover a short and a long clip, not whichever files happen to be warm.
    refs.sort(key=lambda x:x['duration_s'])
    for row in [refs[0],refs[-1]]:
        ref=row['gt_reference'];start=time.perf_counter();original,fps,audit=reader.decode_reference(ref)
        original_seconds=time.perf_counter()-start
        start=time.perf_counter();cold,cf,ca=cache.get(ref);cold_s=time.perf_counter()-start
        start=time.perf_counter();warm,wf,wa=cache.get(ref);warm_s=time.perf_counter()-start
        assert np.array_equal(original,cold) and np.array_equal(original,warm)
        assert audit==ca==wa and fps==cf==wf
        # Nontrivial pixel scores: use the same deterministic altered image on
        # both paths, rather than comparing only identity/zero-error metrics.
        pred=original[0]//2
        before=pixel_scores(original[0],pred);after=pixel_scores(warm[0],pred)
        assert before==after
        records.append(dict(id=row['id'],frames=len(original),original_seconds=original_seconds,
            cold_cache_seconds=cold_s,warm_cache_seconds=warm_s,rgb_sha256=audit['rgb_sha256'],
            exact_pixels=True,exact_audit=True,exact_pixel_scores=True,original_scores=before,cached_scores=after))
        print(json.dumps(records[-1]),flush=True)
        del original,cold,warm
    (w/'DECODE_CACHE_PARITY.json').write_text(json.dumps(dict(success=True,records=records),indent=2)+'\n')
    if a.prewarm:
        started=time.time()
        def one(row):
            frames,_,audit=cache.get(row['gt_reference'])
            result=dict(id=row['id'],frames=len(frames),bytes=frames.nbytes,rgb_sha256=audit['rgb_sha256'])
            del frames
            return result
        out=[]
        with ThreadPoolExecutor(max_workers=a.workers) as pool:
            for value in pool.map(one,refs):
                out.append(value)
                if len(out)%20==0:print('prewarmed',len(out),'/',len(refs),flush=True)
        assert len(out)==len({x['id'] for x in out})==len(refs)
        report=dict(success=True,count=len(out),seconds=time.time()-started,bytes=sum(x['bytes'] for x in out),
            workers=a.workers,records=out,cache=cache.report())
        (w/'DECODE_CACHE_PREWARMED.json').write_text(json.dumps(report,indent=2)+'\n')
        print('Full fixed-pool prewarming complete',len(out),flush=True)


if __name__=='__main__':main()
