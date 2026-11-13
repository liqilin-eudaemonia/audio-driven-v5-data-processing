"""Paired native GLARE loss/gradient sensitivity; no optimizer step or cache write."""
import gc
import json
import math
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
import torch
from baseline_training import training
import run_glare_shared_cache as common
import prepare_glare_nonprosody_cache as base
from glare_lia_fast_parallel import FastEncoder
from selective_precision import SelectiveEncoder

OUT=common.WORK/'gpu7_only/lia_selective_training_check'
DIAGNOSTIC=ROOT/'experiments/baseline_training_realdata_20261109'


def gradients(model):
    result={}
    for name,p in model.named_parameters():
        if not p.requires_grad:
            assert p.grad is None
        elif p.grad is not None:
            assert torch.isfinite(p.grad).all(),name
            result[name]=p.grad.detach().cpu().clone()
    assert sum(float(v.square().sum()) for v in result.values())>0
    return result


def compare(a,b):
    assert set(a)==set(b)
    norm_a=norm_b=error=dot=0.
    exact=True
    for name in a:
        x,y=a[name],b[name]
        norm_a+=float(x.square().sum(dtype=torch.float64))
        norm_b+=float(y.square().sum(dtype=torch.float64))
        error+=float((x-y).square().sum(dtype=torch.float64))
        dot+=float((x*y).sum(dtype=torch.float64))
        exact=exact and torch.equal(x,y)
    return dict(exact=exact,relative_l2=math.sqrt(error/max(norm_a,1e-30)),
                cosine=dot/math.sqrt(max(norm_a*norm_b,1e-60)),native_norm=math.sqrt(norm_a),candidate_norm=math.sqrt(norm_b))


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='7'
    OUT.mkdir(exist_ok=True)
    # Diagnostic acceptance limits, not a claim of full-training equivalence.
    gates=dict(loss_relative_max=.005,gradient_relative_l2_max=.02,gradient_cosine_min=.9998)
    base.save(OUT/'GATES.json',dict(**gates,at=time.time(),scope='Paired short diagnostic; no optimization steps.'))
    torch.set_num_threads(2);torch.cuda.set_device(0)
    assert torch.cuda.mem_get_info()[0]>24*1024**3
    torch.cuda.set_per_process_memory_fraction(20*1024**3/torch.cuda.get_device_properties(0).total_memory)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)
    trials=base.read(common.WORK/'gpu7_only/lia_layout_probe/RESULT.json')['trials']
    best=next(t for t in trials if t['label']=='channels_last_amp_fp32_blur')
    training.seed(137)
    model,implementation=training.build('glare','cuda:0')
    checkpoint=DIAGNOSTIC/'runs/glare/checkpoint.pt'
    checkpoint_sha=base.sha(checkpoint)
    loaded=torch.load(checkpoint,map_location='cpu',weights_only=False)
    params=dict(model.named_parameters())
    assert set(loaded['model'])=={n for n,p in params.items() if p.requires_grad}
    with torch.no_grad():
        for name,value in loaded['model'].items():params[name].copy_(value)
    del loaded
    diagnostic=base.read(DIAGNOSTIC/'DATA_PROTOCOL.json')
    ids=diagnostic['diagnostic_train']+diagnostic['diagnostic_validation']
    model.motion_autoencoder.eval().requires_grad_(False)
    baseline=FastEncoder(model.motion_autoencoder);baseline.warm()
    candidate=SelectiveEncoder(model.motion_autoencoder)
    data={};features=[]
    for sid in ids:
        frames=np.array(np.load(DIAGNOSTIC/'media'/sid/'listener.npy',mmap_mode='r'))
        old=baseline.encode(model.motion_autoencoder,frames).clone()
        new=candidate.encode(model.motion_autoencoder,frames).clone()
        source=training.feature('glare',DIAGNOSTIC,sid)
        a,b=dict(source),dict(source);a['motion']=old;b['motion']=new
        data[sid]=(a,b)
        features.append(dict(sid=sid,max_error=float((old-new).abs().max()),relative_l2=float(torch.linalg.vector_norm(new-old)/torch.linalg.vector_norm(old))))
    # Adjacent, alternating pairs reduce bias from other GPU jobs changing load.
    speed=[]
    for repeat in range(3):
        for index,sid in enumerate(ids):
            frames=np.array(np.load(DIAGNOSTIC/'media'/sid/'listener.npy',mmap_mode='r'))
            order=(False,True) if (repeat+index)%2==0 else (True,False)
            times={}
            for use_new in order:
                tick=time.monotonic()
                (candidate if use_new else baseline).encode(model.motion_autoencoder,frames)
                torch.cuda.synchronize();times['candidate' if use_new else 'native']=time.monotonic()-tick
            speed.append(dict(sid=sid,repeat=repeat,**times))
    random=np.random.default_rng(137)
    pairs=np.asarray([[r['native'],r['candidate']] for r in speed])
    indexes=random.integers(0,len(pairs),size=(2000,len(pairs)))
    sums=pairs[indexes].sum(1);ratios=sums[:,0]/sums[:,1]
    speed_report=dict(pairs=speed,speedup=float(pairs[:,0].sum()/pairs[:,1].sum()),
                      bootstrap_95_interval=np.quantile(ratios,[.025,.975]).tolist(),
                      baseline='Current production FP32 fast2; own producer paused, other jobs unchanged.')
    base.save(OUT/'PAIRED_SPEED.json',speed_report)
    del baseline,candidate
    gc.collect();torch.cuda.empty_cache()
    # Match the established diagnostic training arithmetic after teacher extraction.
    torch.backends.cudnn.allow_tf32=False
    results=[];determinism=None
    for sid in ids:
        for seed in (137,139):
            cases=[];native=None
            for index,value in enumerate(data[sid]):
                training.seed(seed);training.set_train(model,'glare');model.zero_grad(set_to_none=True)
                loss,logs=training.call(model,'glare',value,None,'cuda:0')
                assert torch.isfinite(loss)
                numbers={k:float(v.detach()) for k,v in logs.items() if torch.is_tensor(v) and v.numel()==1}
                loss.backward();grad=gradients(model)
                if index==0:
                    native=grad
                    if determinism is None:
                        old_loss=float(loss.detach())
                        training.seed(seed);training.set_train(model,'glare');model.zero_grad(set_to_none=True)
                        repeat,_=training.call(model,'glare',value,None,'cuda:0');repeat.backward()
                        checked=compare(native,gradients(model))
                        determinism=dict(loss_exact=float(repeat.detach())==old_loss,gradient_exact=checked['exact'])
                        assert all(determinism.values()),determinism
                    cases.append(dict(losses=numbers))
                else:
                    cases.append(dict(losses=numbers,gradient=compare(native,grad)))
            rel={k:abs(cases[1]['losses'][k]-cases[0]['losses'][k])/max(abs(cases[0]['losses'][k]),1e-12) for k in cases[0]['losses']}
            row=dict(sid=sid,seed=seed,native=cases[0],candidate=cases[1],loss_relative=rel,
                     passed=rel['loss']<=gates['loss_relative_max'] and cases[1]['gradient']['relative_l2']<=gates['gradient_relative_l2_max'] and cases[1]['gradient']['cosine']>=gates['gradient_cosine_min'])
            results.append(row);base.save(OUT/'PROGRESS.json',dict(rows=results,determinism=determinism))
            print(json.dumps(dict(sid=sid,seed=seed,passed=row['passed'],loss_relative=rel['loss'],gradient=cases[1]['gradient'])),flush=True)
            del native,grad
    assert base.sha(checkpoint)==checkpoint_sha
    result=dict(passed=all(r['passed'] for r in results),rows=results,gates=gates,determinism=determinism,features=features,
                selected_batch=4,paired_speed=speed_report,checkpoint=str(checkpoint),checkpoint_sha256=checkpoint_sha,
                original_checkpoint_unchanged=True,production_cache_written=False,optimizer_steps=0,
                scope='Four diagnostic clips, two fixed seeds; native FLOAT loss and all trainable gradients. Not full-training or final-generation quality equivalence.',
                candidate_source_sha256=base.sha(Path(__file__).with_name('selective_precision.py')),
                implementation=implementation)
    base.save(OUT/'RESULT.json',result)


if __name__=='__main__':main()
