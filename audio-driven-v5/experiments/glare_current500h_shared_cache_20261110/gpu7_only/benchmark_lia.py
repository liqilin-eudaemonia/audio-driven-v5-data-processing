"""Frozen LIA versus exact-batch transfer/deferred-copy/CUDA-graph execution."""
import hashlib
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
from glare_lia_execution import Encoder

OUT=common.WORK/'gpu7_only/lia_optimization'


def main():
    OUT.mkdir(exist_ok=True)
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='7'
    torch.set_num_threads(2);torch.cuda.set_device(0)
    assert torch.cuda.device_count()==1 and torch.cuda.mem_get_info()[0]>16*1024**3
    torch.cuda.set_per_process_memory_fraction(12*1024**3/torch.cuda.get_device_properties(0).total_memory)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)
    p=base.load_protocol(base.DEFAULT);model=base.load_lia(p)
    cohort=[]
    d=base.read(ROOT/'experiments/baseline_training_realdata_20261109/DATA_PROTOCOL.json')
    for sid in d['diagnostic_train']+d['diagnostic_validation']:
        frames=np.load(ROOT/'experiments/baseline_training_realdata_20261109/media'/sid/'listener.npy',mmap_mode='r')
        cohort.append((sid,np.array(frames)))
    # Read only bounded current in-flight shared-memory groups. Copying an open
    # mmap remains safe if the producer unlinks its path during this read.
    state=base.read(common.WORK/'status_all.json');live=state['worker_groups']['visual']['7']['process_pid']
    directory=Path('/dev/shm')/('glare-shared-7-'+str(live))
    seen={sid for sid,_ in cohort}
    for folder in list(directory.glob('group-*'))[:40]:
        for path in list(folder.glob('*/lia_rgb.npy'))[:2]:
            if len(cohort)>=16:break
            sid=path.parent.name
            if sid in seen:continue
            try:frames=np.array(np.load(path,mmap_mode='r'))
            except FileNotFoundError:continue
            cohort.append((sid,frames));seen.add(sid)
        if len(cohort)>=16:break
    assert len(cohort)>=8,'Not enough real in-flight groups to validate'
    if 1 not in {len(f)%4 for _,f in cohort}:
        cohort.append(('real_frame_tail1_boundary',cohort[0][1][:49].copy()))
    base.save(OUT/'COHORT.json',dict(samples=[dict(sid=sid,frames=len(f),input_sha256=hashlib.sha256(f.tobytes()).hexdigest()) for sid,f in cohort],gpu=7))
    # Warm all native tail batch shapes before comparing execution methods.
    for n in [1,2,3,4]:base.encode(model,cohort[0][1][:n])
    profile=[]
    with torch.inference_mode():
        for start in range(0,16,4):
            tick=time.monotonic()
            host=torch.from_numpy(np.array(cohort[0][1][start:start+4])).permute(0,3,1,2).float()
            cpu_s=time.monotonic()-tick
            a,b,c,d=[torch.cuda.Event(enable_timing=True) for _ in range(4)]
            a.record();tick=time.monotonic();x=host.cuda();h2d_s=time.monotonic()-tick;b.record()
            c.record();tick=time.monotonic()
            x=x/127.5-1;appearance,_,_=model.enc(x,input_target=None)
            output=model.dec.direction(model.enc.fc(appearance));submit_s=time.monotonic()-tick;d.record()
            tick=time.monotonic();output.cpu();wait_s=time.monotonic()-tick
            profile.append(dict(cpu_float_prepare_s=cpu_s,h2d_wall_s=h2d_s,
                gpu_submit_wall_s=submit_s,output_wait_wall_s=wait_s,
                h2d_gpu_ms=a.elapsed_time(b),forward_gpu_ms=c.elapsed_time(d)))
    base.save(OUT/'BASELINE_PROFILE.json',dict(batches=profile,
        caution='Output wait overlaps GPU forward; do not add event time to wall time. Shared GPU scheduling affects event intervals.'))
    with torch.inference_mode():
        a=torch.from_numpy(cohort[0][1][:4].copy()).permute(0,3,1,2).float().cuda()/127.5-1
        b=torch.from_numpy(cohort[0][1][:4].copy()).cuda().permute(0,3,1,2).float()/127.5-1
        assert torch.equal(a,b) and a.stride()==b.stride()
    methods={};screen=[];references={}
    for sid,frames in cohort[:4]:references[sid]=base.encode(model,frames)
    for mode in ['deferred','upload','graph']:
        try:
            engine=Encoder(model,mode)
            if mode=='graph':
                for n in [1,2,3,4]:engine.graph(n)
            for sid,frames in cohort[:4]:
                output=engine.encode(model,frames)
                error=float((output-references[sid]).abs().max())
                screen.append(dict(mode=mode,sid=sid,max_error=error,exact=torch.equal(output,references[sid])))
                torch.testing.assert_close(output,references[sid],rtol=1e-5,atol=1e-5)
            methods[mode]=engine
        except Exception as exc:
            screen.append(dict(mode=mode,error=repr(exc),rejected=True))
        base.save(OUT/'SCREEN.json',dict(rows=screen))
    assert methods,'No numerical-equivalent candidate'
    # Screen timings use a balanced baseline/candidate/candidate/baseline order.
    trials=[]
    for mode,engine in methods.items():
        rounds=[]
        for use_new in [False,True,True,False]:
            tick=time.monotonic()
            for sid,frames in cohort[:4]:
                out=engine.encode(model,frames) if use_new else base.encode(model,frames)
            torch.cuda.synchronize();rounds.append(dict(candidate=use_new,seconds=time.monotonic()-tick))
        old=sum(r['seconds'] for r in rounds if not r['candidate']);new=sum(r['seconds'] for r in rounds if r['candidate'])
        trials.append(dict(mode=mode,rounds=rounds,speedup=old/new,old_s=old,new_s=new))
        base.save(OUT/'SCREEN_TIMINGS.json',dict(trials=trials))
    best=max(trials,key=lambda r:r['speedup']);engine=methods[best['mode']]
    parity=[]
    for sid,frames in cohort:
        old=base.encode(model,frames);new=engine.encode(model,frames)
        torch.testing.assert_close(new,old,rtol=1e-5,atol=1e-5)
        parity.append(dict(sid=sid,frames=len(frames),max_error=float((old-new).abs().max()),exact=torch.equal(old,new)))
    rounds=[]
    for use_new in [False,True,True,False]:
        tick=time.monotonic()
        for sid,frames in cohort:
            out=engine.encode(model,frames) if use_new else base.encode(model,frames)
        torch.cuda.synchronize();rounds.append(dict(candidate=use_new,seconds=time.monotonic()-tick))
    old=sum(r['seconds'] for r in rounds if not r['candidate']);new=sum(r['seconds'] for r in rounds if r['candidate'])
    result=dict(passed=True,mode=best['mode'],samples=len(cohort),frames=sum(len(f) for _,f in cohort),
        speedup=old/new,baseline_s=old,candidate_s=new,rounds=rounds,parity=parity,
        all_tensors_exact=all(r['exact'] for r in parity),max_error=max(r['max_error'] for r in parity),
        tolerance=dict(rtol=1e-5,atol=1e-5),batch=4,dtype='float32',screen=screen,trials=trials,
        profile=engine.profile(),execution_source_sha256=base.sha(ROOT/'scripts/glare_lia_execution.py'),
        native_source_unchanged=True,cache_payloads_modified=False,
        limitation='Shared GPU7; ABBA on same real pixels. Live end-to-end throughput must be checked separately.')
    base.save(OUT/'RESULT.json',result)
    print(json.dumps({k:result[k] for k in ['passed','mode','samples','speedup','all_tensors_exact','max_error']}),flush=True)


if __name__=='__main__':main()
