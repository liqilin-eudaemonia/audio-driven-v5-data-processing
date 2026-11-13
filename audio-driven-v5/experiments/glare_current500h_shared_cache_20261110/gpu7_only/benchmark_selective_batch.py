"""Compare large real batches with selective AMP and FP32 layout alternatives."""
import gc
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
import torch
import prepare_glare_nonprosody_cache as base
import run_glare_shared_cache as common
from glare_lia_fast_parallel import FastEncoder
from selective_batch import Candidate

OUT=common.WORK/'gpu7_only/lia_selective_batch_probe'


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='7'
    OUT.mkdir(exist_ok=True)
    torch.set_num_threads(2);torch.cuda.set_device(0)
    assert torch.cuda.mem_get_info()[0]>26*1024**3
    torch.cuda.set_per_process_memory_fraction(24*1024**3/torch.cuda.get_device_properties(0).total_memory)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)
    model=base.load_lia(base.load_protocol(base.DEFAULT))
    diagnostic=base.read(ROOT/'experiments/baseline_training_realdata_20261109/DATA_PROTOCOL.json')
    cohort=[]
    for sid in diagnostic['diagnostic_train']+diagnostic['diagnostic_validation']:
        cohort.append((sid,np.array(np.load(ROOT/'experiments/baseline_training_realdata_20261109/media'/sid/'listener.npy',mmap_mode='r'))))
    live=base.read(common.WORK/'workers/gpu7.json')['pid']
    for folder in list((Path('/dev/shm')/('glare-shared-7-'+str(live))).glob('group-*'))[:32]:
        for path in list(folder.glob('*/lia_rgb.npy'))[:2]:
            if len(cohort)>=8:break
            try:frames=np.array(np.load(path,mmap_mode='r'))
            except FileNotFoundError:continue
            cohort.append((path.parent.name,frames))
        if len(cohort)>=8:break
    base.save(OUT/'COHORT.json',dict(samples=[dict(sid=sid,frames=len(f)) for sid,f in cohort]))
    baseline=FastEncoder(model);baseline.warm()
    refs={sid:baseline.encode(model,f) for sid,f in cohort}
    direction=model.dec.direction.weight.detach().cpu()
    from float.utils import compute_alpha_stats
    rows=[]
    for mixed,batch in [(False,16),(False,32),(True,8),(True,16),(True,32),(True,64)]:
        candidate=None
        try:
            torch.cuda.reset_peak_memory_stats()
            candidate=Candidate(model,batch,mixed)
            parity=[]
            for sid,frames in cohort:
                new=candidate.encode(model,frames);old=refs[sid];diff=(new-old).abs()
                a,b=compute_alpha_stats(old[1:]@direction),compute_alpha_stats(new[1:]@direction)
                parity.append(dict(sid=sid,frames=len(frames),exact=torch.equal(old,new),max_error=float(diff.max()),
                                   relative_l2=float(torch.linalg.vector_norm(new-old)/torch.linalg.vector_norm(old)),
                                   statistics_relative_l2={k:float(torch.linalg.vector_norm(a[k]-b[k])/torch.linalg.vector_norm(a[k]).clamp_min(1e-9)) for k in a}))
            rounds=[]
            for use_new in (False,True,True,False):
                tick=time.monotonic()
                for sid,frames in cohort:(candidate if use_new else baseline).encode(model,frames)
                torch.cuda.synchronize();rounds.append(dict(candidate=use_new,seconds=time.monotonic()-tick))
            old_s=sum(r['seconds'] for r in rounds if not r['candidate']);new_s=sum(r['seconds'] for r in rounds if r['candidate'])
            rows.append(dict(batch=batch,mixed_precision=mixed,speedup_vs_current_fast2=old_s/new_s,
                             parity=parity,max_error=max(p['max_error'] for p in parity),relative_l2_max=max(p['relative_l2'] for p in parity),
                             statistics_relative_l2_max=max(v for p in parity for v in p['statistics_relative_l2'].values()),
                             peak_gib=torch.cuda.max_memory_allocated()/1024**3,rounds=rounds))
        except Exception as exc:
            rows.append(dict(batch=batch,mixed_precision=mixed,error=repr(exc),adopted=False))
        base.save(OUT/'PROGRESS.json',dict(trials=rows))
        print(json.dumps({k:v for k,v in rows[-1].items() if k not in ['parity','rounds']}),flush=True)
        del candidate
        gc.collect();torch.cuda.empty_cache()
    base.save(OUT/'RESULT.json',dict(trials=rows,samples=len(cohort),frames=sum(len(f) for _,f in cohort),
                                  physical_gpu=7,production_adopted=False,production_cache_written=False,
                                  candidate_source_sha256=base.sha(Path(__file__).with_name('selective_batch.py')),
                                  own_production_worker_paused=True,other_jobs_kept_running=True,
                                  tail_padding='Only unused inference batch slots; output always has the exact real frame count; no batch normalization or cross-frame mixing.'))


if __name__=='__main__':main()
