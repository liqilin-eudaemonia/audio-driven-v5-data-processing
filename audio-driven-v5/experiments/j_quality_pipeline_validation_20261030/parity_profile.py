"""Real CPU measurement parity and profile, run independently on each target node."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import argparse
import cProfile
import io
import json
from pathlib import Path
import pstats
import sys
import time

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from listen_visual_quality.measurement import measure_row
from listen_visual_quality.cli import load_policy,DEFAULT_MODEL
from listen_visual_quality.decision import decide,evidence_index
from listen_visual_quality.rules import analyze_focus
from listen_benchmark.j_quality_pipeline import compact_write

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--bounded-detector-threads',action='store_true');a=p.parse_args()
    os.sched_setaffinity(0,{sorted(os.sched_getaffinity(0))[2]})
    if a.bounded_detector_threads:
        from listen_visual_quality.measurement import backend
        from listen_benchmark.quality_cpu_execution import serial_landmarker
        vendor,*_=backend(str(ROOT));vendor.FaceLandmarker=serial_landmarker(vendor.FaceLandmarker)
    old=json.load(open(ROOT/'experiments/visual_quality_holdout120_20261030/data.json'))['results']
    policy=load_policy(ROOT/'configs/listen_visual_quality_v1.json')
    evidence=evidence_index(json.load(open(ROOT/'configs/listen_visual_quality_reviewed_evidence_20261030.json')))
    prof=cProfile.Profile();prof.enable();checks=[];start=time.time()
    for i in (19,24,38,48,51):
        r=old[i];got=measure_row({'sample_id':r['id'],'source':r['source']},policy,str(ROOT),str(DEFAULT_MODEL))
        assert got['status']=='ok',got
        for k in ('raw_measurements','focus_samples','crop'):assert got[k]==r[k],(i,k)
        expected=analyze_focus(r['focus_samples'],r['source']['duration_s'],policy)
        assert expected==got['focus']
        checks.append({'ordinal':i+1,'elapsed_s':got['elapsed_s'],'exact_match':True,'decision':decide(got,policy,evidence)['decision']})
    prof.disable();buf=io.StringIO();pstats.Stats(prof,stream=buf).sort_stats('cumtime').print_stats(30)
    a.output.mkdir(exist_ok=True,parents=True);(a.output/'profile5.txt').write_text(buf.getvalue())
    pretty=len(json.dumps(got,indent=2).encode());compact=len(json.dumps(got,separators=(',',':')).encode())
    result={'passed':True,'checks':checks,'elapsed_s':time.time()-start,
            'example_json_pretty_bytes':pretty,'example_json_compact_bytes':compact,'compact_size_ratio':compact/pretty}
    compact_write(a.output/'real5_parity.json',result);print(json.dumps(result))
