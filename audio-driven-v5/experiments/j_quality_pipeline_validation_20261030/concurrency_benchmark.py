"""Identical real clips at several CPU concurrencies, with output equivalence checks."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import argparse
import concurrent.futures
import multiprocessing
from pathlib import Path
import random
import sys
import time
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from listen_visual_quality.measurement import measure_row
from listen_visual_quality.cli import load_policy,DEFAULT_MODEL
from listen_visual_quality.io import read,write,canonical_hash
from listen_visual_quality.rules import analyze_focus

def initialize(counter,cores,bounded):
    with counter.get_lock():i=counter.value;counter.value+=1
    os.sched_setaffinity(0,{cores[i%len(cores)]})
    if bounded:
        from listen_visual_quality.measurement import backend
        from listen_benchmark.quality_cpu_execution import serial_landmarker
        vendor,*_=backend(str(ROOT));vendor.FaceLandmarker=serial_landmarker(vendor.FaceLandmarker)

def run(task):
    row,policy,expected=task
    record=measure_row(row,policy,str(ROOT),str(DEFAULT_MODEL))
    if record['status']!='ok':raise RuntimeError(record.get('error'))
    actual=canonical_hash({k:record[k] for k in ('raw_measurements','focus_samples','focus','crop')})
    if actual!=expected:raise ValueError('Measurement changed: '+row['sample_id'])
    return record['elapsed_s']

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,nargs='+',default=[80,160,224]);p.add_argument('--clips',type=int,default=640)
    p.add_argument('--bounded-detector-threads',action='store_true')
    p.add_argument('--warm-round',action='store_true');a=p.parse_args()
    old=read(ROOT/'experiments/visual_quality_holdout120_20261030/data.json')['results']
    policy=load_policy(ROOT/'configs/listen_visual_quality_v1.json');tasks=[]
    for record in old:
        record['focus']=analyze_focus(record['focus_samples'],record['source']['duration_s'],policy)
        expected=canonical_hash({k:record[k] for k in ('raw_measurements','focus_samples','focus','crop')})
        tasks.append(({'sample_id':record['id'],'source':record['source']},policy,expected))
    random.Random(20261030).shuffle(tasks);tasks=(tasks*((a.clips+len(tasks)-1)//len(tasks)))[:a.clips]
    results=[];cores=sorted(os.sched_getaffinity(0))
    for n in a.workers:
        counter=multiprocessing.Value('i',0);tick=time.monotonic()
        with concurrent.futures.ProcessPoolExecutor(n,initializer=initialize,initargs=(counter,cores,a.bounded_detector_threads)) as pool:
            timings=list(pool.map(run,tasks,chunksize=1))
            cold_elapsed=time.monotonic()-tick
            if a.warm_round:
                tick=time.monotonic()
                timings=list(pool.map(run,tasks,chunksize=1))
            measured_elapsed=time.monotonic()-tick
        elapsed=time.monotonic()-tick
        results.append({'workers':n,'clips':len(timings),'elapsed_s':measured_elapsed,'clips_per_s':len(timings)/measured_elapsed,
                        'sum_worker_s':sum(timings),'exact_equivalence':True,'includes_process_startup':not a.warm_round,
                        'cold_elapsed_s':cold_elapsed,
                        'bounded_detector_threads':a.bounded_detector_threads})
        write(a.output,{'results':results,'clips_per_arm':a.clips,'cpu_affinity_count':len(cores)})
        print(results[-1],flush=True)
