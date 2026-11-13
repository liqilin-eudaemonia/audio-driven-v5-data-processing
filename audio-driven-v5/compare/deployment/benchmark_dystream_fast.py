"""Two real ViCo inputs, truncated to 2s, official DyStream inference/export parity."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import soundfile as sf
from vico_more_common import ROOT, read, save, sha
from vico_fast_registry import BASE
OUT=ROOT/'experiments/p2_vico_inference_fast_20261110/benchmark_dystream'


def prepare():
    p=read(BASE/'dystream/protocol.json')
    samples=[]
    for original in p['samples'][:2]:
        item=dict(original)
        for key in ['speaker','silent']:
            wave,sr=sf.read(original[key],dtype='float32')
            dest=OUT/'inputs'/original['id']/(key+'.wav');dest.parent.mkdir(parents=True,exist_ok=True)
            sf.write(dest,wave[:32000],sr,subtype='FLOAT')
            item[key]=str(dest)
        item.update(duration=2.,frames=60)
        samples.append(item)
    p['samples']=samples;p['benchmark']='two real inputs truncated to2sec; not full test43'
    save(OUT/'protocol.json',p)


def run(mode,repeat):
    work=OUT/(str(repeat)+'-'+mode);work.mkdir(parents=True,exist_ok=True)
    p=read(OUT/'protocol.json');save(work/'protocol.json',p)
    start=[]
    if mode=='baseline':import vico_dystream as module
    else:import vico_dystream_fast as module
    original=module.save
    def timed_save(path,data):
        result=original(path,data)
        if path.parent.name=='model_load':start.append(time.perf_counter())
        return result
    module.save=timed_save
    before=time.perf_counter()
    if mode=='baseline':
        (work/'claims').mkdir(exist_ok=True)
        module.worker(work)
    else:module.worker(work,p['samples'])
    end=time.perf_counter();assert len(start)==1
    save(work/'TIMING.json',dict(mode=mode,repeat=repeat,seconds=end-start[0],model_load_seconds=start[0]-before))


def summarize():
    import statistics
    rows=[];reference={}
    for repeat in range(2):
        for mode in ['baseline','fast']:
            work=OUT/(str(repeat)+'-'+mode);row=read(work/'TIMING.json')
            for s in read(OUT/'protocol.json')['samples']:
                uid=s['id'];d=work/'generated'/uid
                latents=np.load(d/'motion_latent.npy')
                score=work/'scoring/dystream'/(uid+'.listener.mp4')
                identity=(sha(score),latents)
                if uid not in reference:reference[uid]=identity
                assert identity[0]==reference[uid][0],(mode,repeat,uid,'scoring video differs')
                assert np.array_equal(identity[1],reference[uid][1]),(mode,repeat,uid,'motion differs')
            rows.append(row)
    baseline=statistics.median(r['seconds'] for r in rows if r['mode']=='baseline')
    fast=statistics.median(r['seconds'] for r in rows if r['mode']=='fast')
    save(OUT/'BENCHMARK.json',dict(success=True,clips=2,native_frames_per_clip=49,scoring_frames_per_clip=60,
        rounds=2,motion_exact=True,scoring_video_exact=True,baseline_median_seconds=baseline,
        fast_median_seconds=fast,speedup=baseline/fast,records=rows,
        limitation='2sec clips; includes real official motion generation, rendering, both native and scoring exports; model load excluded; single-worker baseline'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','baseline','fast','summarize']);p.add_argument('--repeat',type=int,default=0);a=p.parse_args()
    if a.mode=='prepare':prepare()
    elif a.mode=='summarize':summarize()
    else:run(a.mode,a.repeat)
