"""Large-batch equivalence and throughput test using the exact reviewed prompts."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path('/private/liqi_workspace/project_code/audio-driven-v5')
sys.path.insert(0,str(ROOT/'src'))


def worker(node,gpu):
    import torch
    from p2_listen.contracts import read,atomic_json
    from p2_listen.config import activate,extractor_config
    from p3_listen.prompts import FixedPrompt
    old=ROOT/'experiments/p2_system_prompt_cache_smoke_20261101/j24_v2'
    m=read(old/'manifest.json');activate(m['config']);torch.set_num_threads(2)
    from omni_av.ltx_runtime import NativeTextConditioner
    encoder=NativeTextConditioner(extractor_config(m['config']),'cuda:0')
    ids=(m['sample_ids']*2)[:32]
    prompts=[read(old/'samples'/(sid+'.json'))['prompt'] for sid in ids]
    torch.cuda.reset_peak_memory_stats();times=[];values=None
    for repeat in range(2):
        torch.cuda.synchronize();start=time.monotonic()
        values=encoder.encode_batch([FixedPrompt(p) for p in prompts])
        torch.cuda.synchronize();times.append(time.monotonic()-start)
    rows=[]
    for sid,value in zip(ids,values):
        reference=torch.load(old/'cache/text_guided'/sid/'payload.pt',map_location='cpu',weights_only=True)['positive']
        if not torch.equal(value['mask'].cpu(),reference['mask']):raise ValueError('Mask changed')
        row=dict(sample_id=sid)
        for part in ('video','audio'):
            a=value[part].float().cpu();b=reference[part].float()
            if not torch.isfinite(a).all():raise ValueError('Nonfinite text features')
            rel=float(torch.linalg.vector_norm(a-b)/torch.linalg.vector_norm(b).clamp_min(1e-10))
            cos=float(torch.nn.functional.cosine_similarity(a.reshape(1,-1),b.reshape(1,-1)))
            row[part]=dict(relative_l2=rel,cosine=cos,max_abs=float((a-b).abs().max()),bit_exact=bool(torch.equal(a,b)))
            if rel>0.015 or cos<0.9999:raise ValueError('Large-batch BF16 features differ beyond two epsilon tolerance: '+str(row))
        rows.append(row)
    out=ROOT/'experiments/p2_system_prompt_cache_smoke_20261101/batch32'
    atomic_json(out/(f'{node}-gpu{gpu}.json'),dict(success=True,node=node,gpu=gpu,batch=32,iteration_s=times,
        samples_per_s=32/times[-1],peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,results=rows))


def main():
    p=argparse.ArgumentParser();p.add_argument('--node',required=True);p.add_argument('--gpu',type=int);a=p.parse_args()
    if a.gpu is not None:worker(a.node,a.gpu);return
    processes=[];logs=[]
    out=ROOT/'experiments/p2_system_prompt_cache_smoke_20261101/batch32';out.mkdir(parents=True,exist_ok=True)
    try:
        for gpu in range(8):
            log=(out/f'{a.node}-gpu{gpu}.log').open('w');logs.append(log)
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',
                     HF_ENABLE_PARALLEL_LOADING='true',HF_PARALLEL_LOADING_WORKERS='4',TOKENIZERS_PARALLELISM='false',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
            processes.append(subprocess.Popen([sys.executable,__file__,'--node',a.node,'--gpu',str(gpu)],env=env,stdout=log,stderr=subprocess.STDOUT))
        codes=[p.wait() for p in processes]
        if any(codes):raise RuntimeError(str(codes))
        print(json.dumps(dict(node=a.node,success=True,gpus=8,batch=32)),flush=True)
    finally:
        for process in processes:
            if process.poll() is None:process.terminate()
        for log in logs:log.close()


if __name__=='__main__':main()
