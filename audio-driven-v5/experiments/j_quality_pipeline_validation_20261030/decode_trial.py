"""Compare original and sequential decode on identical real J source intervals."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import argparse
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from listen_visual_quality.measurement import measure_row
from listen_visual_quality.cli import load_policy,DEFAULT_MODEL
from listen_visual_quality.io import read,write
from listen_benchmark.quality_decode_session import sequential_measurements

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--max-clips',type=int,default=12);a=p.parse_args()
    os.sched_setaffinity(0,{sorted(os.sched_getaffinity(0))[1]})
    selection=read(ROOT/'experiments/visual_quality_holdout120_20261030/selection.json')['rows']
    old_work=ROOT/'experiments/listen_j_annotation_node17_20261029'
    jobs=[]
    for r in selection:
        pair=r['source']['pair']
        if pair not in jobs:jobs.append(pair)
        if len(jobs)==2:break
    policy=load_policy(ROOT/'configs/listen_visual_quality_v1.json');results=[]
    for job in jobs:
        rows=read(old_work/'jobs'/f'{job}.json')['rows']
        # A continuous time order makes nearby decoder reuse possible for each participant.
        rows=sorted(rows,key=lambda r:(r['source']['listener_id'],r['source']['source_start_s']))[:a.max_clips]
        base=[];tick=time.monotonic()
        for row in rows:base.append(measure_row(row,policy,str(ROOT),str(DEFAULT_MODEL)))
        original_s=time.monotonic()-tick;tick=time.monotonic()
        with sequential_measurements(str(ROOT)) as stats:
            for row,expected in zip(rows,base):
                got=measure_row(row,policy,str(ROOT),str(DEFAULT_MODEL))
                assert got['status']==expected['status']=='ok',(row['sample_id'],got)
                for key in ('raw_measurements','focus_samples','focus','crop'):
                    assert got[key]==expected[key],(row['sample_id'],key)
        results.append({'source':job,'clips':len(rows),'original_s':original_s,
                        'sequential_s':time.monotonic()-tick,'decoder_stats':dict(stats),'exact_match':True})
        write(a.output,{'passed':True,'sources':results})
    print(results)
