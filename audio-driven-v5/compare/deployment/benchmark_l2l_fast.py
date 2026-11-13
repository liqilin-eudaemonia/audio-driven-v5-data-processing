"""Real ViCo43 batch, all four official L2L weight sets, identical RNG states."""
import json
import os
from pathlib import Path
import statistics
import sys
import time
import types
import numpy as np
from vico_more_common import ROOT, read, save, sha
from vico_fast_registry import node8_guard, NAMES
from vico_l2l_fast import ARCHIVE, BASE, REPO, install_resident_inputs, aligned_window


def run_loop(module, config, args, vq, generator, x, a, frames, fast):
    import torch
    if fast:
        x=torch.from_numpy(x).cuda();a=torch.from_numpy(a).cuda()
        history=torch.zeros((len(x),32,56),device='cuda')
    else:
        history=np.zeros((len(x),32,56),np.float32)
    tokens=None;shape=None
    with torch.no_grad():
        for t in range(0,frames,8):
            past=history if tokens is None else (tokens[:,-4:] if fast else tokens[:,-4:].cpu().numpy())
            xw,aw=aligned_window(x,a,t)
            prediction,prob,inputs,shape=module.generate_prediction(config,args,vq,generator,xw,past,aw,32,8,t//8,4,btc=shape)
            if tokens is None:tokens=inputs['listener_past']
            tokens=torch.cat((tokens,prediction[:,0]),dim=1)
            if not fast:prob.cpu().numpy()
        code=tokens[:,4:]
        decoded=[vq.module.decode_to_img(code[:,t:t+4],shape).cpu().numpy() for t in range(0,code.shape[1],4)]
    return code.cpu().numpy(),np.concatenate(decoded,axis=1)


def main():
    import torch
    node8_guard();torch.set_num_threads(2)
    sys.path.insert(0,str(REPO));os.chdir(REPO)
    import test_vq_decoder as module
    from modules.fact_model import setup_model
    from vqgan.vqmodules.gan_models import setup_vq_transformer
    from utils.load_utils import bilateral_filter
    out=ROOT/'experiments/p2_vico_inference_fast_20261110/benchmark_l2l'
    out.mkdir(parents=True,exist_ok=True)
    frames=128;rounds=3
    items=read(BASE/'protocol.json')['samples'];batch=len(items)
    original=module.create_data_vq
    mel=[np.load(ARCHIVE/'audio_features'/(s['id']+'.npy')) for s in items]
    speaker=[bilateral_filter(np.load(ARCHIVE/'features/speaker'/(s['id']+'.npz'))['motion'].astype(np.float32)[None])[0] for s in items]
    reports=[]
    for name in NAMES:
        config=read(REPO/'configs/vq'/('delta_v6.json' if name=='conan' else name+'_delta_v6.json'))
        vc=read(REPO/config['l_vqconfig']);args=types.SimpleNamespace(sample_idx=None)
        cp=REPO/config['model_path']/(config['tag']+config['pipeline']+'_best.pth')
        vp=REPO/'vqgan'/vc['model_path']/(vc['tag']+vc['pipeline']+'_best.pth')
        xs=np.load(REPO/config['model_path']/(config['tag']+config['pipeline']+'_preprocess_core.npz'))
        mx,sx=xs['body_mean_X'].reshape(56),xs['body_std_X'].reshape(56)
        ma,sa=xs['body_mean_audio'].reshape(128),xs['body_std_audio'].reshape(128)
        x=np.zeros((batch,frames+32,56),np.float32);a=np.zeros((batch,(frames+32)*4,128),np.float32)
        for i,s in enumerate(items):
            n=min(frames,s['frames']);xn=(speaker[i]-mx)/sx;an=(mel[i]-ma)/sa
            x[i,32:32+n]=xn[:n];x[i,32+n:]=xn[n-1]
            a[i,128:128+n*4]=an[:n*4];a[i,128+n*4:]=(0-ma)/sa
        vq,vo,_=setup_vq_transformer(args,vc,load_path=str(vp),test=True);vq.eval()
        gen,go,_=setup_model(config,vc,mask_index=0,test=True,load_path=str(cp));gen.eval()
        rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state_all()
        module.create_data_vq=original
        run_loop(module,config,args,vq,gen,x,a,32,False)
        rows=[];reference=None
        for repeat in range(rounds):
            for fast in ([False,True] if repeat%2==0 else [True,False]):
                if fast:install_resident_inputs(module)
                else:module.create_data_vq=original
                torch.set_rng_state(rng);torch.cuda.set_rng_state_all(cuda_rng)
                torch.cuda.synchronize();start=time.perf_counter()
                tokens,motion=run_loop(module,config,args,vq,gen,x,a,frames,fast)
                torch.cuda.synchronize();elapsed=time.perf_counter()-start
                if reference is None:reference=(tokens,motion)
                assert np.array_equal(tokens,reference[0]),(name,'tokens differ')
                assert np.array_equal(motion,reference[1]),(name,'motion differs',np.abs(motion-reference[1]).max())
                row=dict(fast=fast,repeat=repeat,seconds=elapsed)
                rows.append(row);print(name,row,flush=True)
        baseline=statistics.median(r['seconds'] for r in rows if not r['fast'])
        optimized=statistics.median(r['seconds'] for r in rows if r['fast'])
        report=dict(model=name,batch=batch,frames=frames,rounds=rounds,token_exact=True,motion_exact=True,
            baseline_median_seconds=baseline,fast_median_seconds=optimized,speedup=baseline/optimized,
            records=rows,checkpoint_sha256=sha(cp),vq_sha256=sha(vp))
        reports.append(report);save(out/(name+'.json'),report)
        del vq,gen,vo,go;torch.cuda.empty_cache()
    save(out/'BENCHMARK.json',dict(success=True,models=reports,all_tokens_exact=True,all_motion_exact=True,
        limitation='Full 43-clip batch, first128 frames, model load excluded; no full-test speed extrapolation'))


if __name__=='__main__':main()
