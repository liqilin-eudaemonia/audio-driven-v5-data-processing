"""Index construction + actual retrieval parity on official released segments."""
import ast
import gc
import json
from pathlib import Path
import time
import tracemalloc
import numpy as np
import torch
from lm_listener_eval_fast import ROOT,FrameLookup,SOURCE,transformed_tree
OUT=ROOT/'experiments/p2_vico_additional_optimization_20261110'


def official(segments):
    tree=ast.parse(SOURCE.read_text());main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    nodes=[]
    for i,n in enumerate(main.body):
        if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ('frame_map','speaker_map'):
            nodes.extend(main.body[i:i+2])
    env=dict(segments=segments,np=np)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(SOURCE),'exec'),env)
    return env['frame_map'],env['speaker_map']


def timed(fn):
    gc.collect();tracemalloc.start();start=time.perf_counter();result=fn()
    seconds=time.perf_counter()-start;_,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
    return result,dict(seconds=seconds,python_peak_bytes=peak)


def main():
    source=ROOT/'evaluation/lm-listener/dataset/trevorconanstephen/segments_test_no_overlap.pth'
    segments=torch.load(source,map_location='cpu',weights_only=False)
    # Include all 4 duplicated Stephen intervals, plus short/long and Trevor/Conan records.
    selected=segments[:64]+segments[-64:]
    old,before=timed(lambda:official(selected))
    new,after=timed(lambda:(FrameLookup(selected,'p0'),FrameLookup(selected,'p1')))
    start=time.perf_counter();scalars=0
    for a,b in zip(old,new):
        for key,value in a.items():
            assert key in b and np.array_equal(value,b[key]);scalars+=len(value)
        assert 'missing_video_0' not in b
    lookup_seconds=time.perf_counter()-start
    # Explicit overlapping/duplicate rows and gaps verify last-write-wins.
    rows=[]
    for index,(start,end) in enumerate([(0,5),(2,7),(2,7),(10,12)]):
        rows.append(dict(fname='a_b.mp4',split_start_frame=start,split_end_frame=end,
            **{f'{role}_{key}':np.full((end-start,width),index,dtype=np.float32) for role in ('p0','p1') for key,width in [('exp',50),('pose',6),('shape',100)]}))
    for a,b in zip(official(rows),(FrameLookup(rows,'p0'),FrameLookup(rows,'p1'))):
        assert all(np.array_equal(value,b[key]) for key,value in a.items())
        assert 'a_b.mp4_7' not in b and 'a_b.mp4_9' not in b
    assert transformed_tree()
    report=dict(passed=True,official_segments_total=len(segments),benchmark_segments=len(selected),
        indexed_frames_per_role=len(old[0]),compared_scalars=scalars,exact=True,baseline=before,optimized=after,
        construction_speedup=before['seconds']/after['seconds'],full_parity_lookup_seconds=lookup_seconds,
        scope='Index construction only; not end-to-end scorer or model inference speed',overlap_duplicate_gap_tests=True)
    OUT.mkdir(exist_ok=True);(OUT/'LM_LISTENER_INDEX_BENCHMARK.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':main()
