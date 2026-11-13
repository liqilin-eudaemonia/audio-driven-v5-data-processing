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


import torch
from p2_listen.cache import cpu
from p2_plan_cache.encoding import encode_text

@torch.no_grad()
def encode_text_batch(encoder,prompts):
    """One large native Gemma forward, bounded native projection workspaces."""
    wrapper=encoder.encoder.tokenizer
    if any(len(wrapper.tokenizer(p.strip(),truncation=False)['input_ids'])>1024 for p in prompts):
        return [cpu(encode_text(encoder,p,8192)[0]) for p in prompts]
    pairs=[wrapper.tokenize_with_weights(p)['gemma'] for p in prompts]
    device=encoder.encoder.model.device
    ids=torch.tensor([[v[0] for v in row] for row in pairs],device=device)
    mask=torch.tensor([[v[1] for v in row] for row in pairs],device=device)
    values=[]
    with torch.autocast('cuda',dtype=torch.bfloat16):
        out=encoder.encoder.model.model(input_ids=ids,attention_mask=mask,output_hidden_states=True)
        for start in range(0,len(prompts),8):
            result=encoder.processor.process_hidden_states(tuple(h[start:start+8] for h in out.hidden_states),mask[start:start+8])
            for i,prompt in enumerate(prompts[start:start+8]):
                values.append(dict(video=result.video_encoding[i:i+1].detach().cpu(),
                    audio=result.audio_encoding[i:i+1].detach().cpu(),mask=result.attention_mask[i:i+1].bool().cpu(),prompt=prompt))
            del result
    return values



def worker(node,gpu):
    import torch
    from p2_listen.contracts import read,atomic_json
    from p2_listen.config import activate,extractor_config
    old=ROOT/'experiments/p2_system_prompt_cache_smoke_20261101/j24_v2'
    m=read(old/'manifest.json');activate(m['config']);torch.set_num_threads(2)
    from omni_av.ltx_runtime import NativeTextConditioner
    encoder=NativeTextConditioner(extractor_config(m['config']),'cuda:0')
    ids=(m['sample_ids']*3)[:64]
    prompts=[read(old/'samples'/(sid+'.json'))['prompt'] for sid in ids]
    torch.cuda.reset_peak_memory_stats();times=[];values=None
    for repeat in range(2):
        torch.cuda.synchronize();start=time.monotonic()
        values=encode_text_batch(encoder,prompts)
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
    out=ROOT/'experiments/p2_system_prompt_cache_smoke_20261101/batch64_optimized'
    atomic_json(out/(f'{node}-gpu{gpu}.json'),dict(success=True,node=node,gpu=gpu,batch=64,iteration_s=times,
        samples_per_s=64/times[-1],peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,results=rows))


def main():
    p=argparse.ArgumentParser();p.add_argument('--node',required=True);p.add_argument('--gpu',type=int);a=p.parse_args()
    if a.gpu is not None:worker(a.node,a.gpu);return
    processes=[];logs=[]
    out=ROOT/'experiments/p2_system_prompt_cache_smoke_20261101/batch64_optimized';out.mkdir(parents=True,exist_ok=True)
    try:
        for gpu in range(8):
            log=(out/f'{a.node}-gpu{gpu}.log').open('w');logs.append(log)
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',
                     HF_ENABLE_PARALLEL_LOADING='true',HF_PARALLEL_LOADING_WORKERS='4',TOKENIZERS_PARALLELISM='false',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
            processes.append(subprocess.Popen([sys.executable,__file__,'--node',a.node,'--gpu',str(gpu)],env=env,stdout=log,stderr=subprocess.STDOUT))
        codes=[p.wait() for p in processes]
        if any(codes):raise RuntimeError(str(codes))
        print(json.dumps(dict(node=a.node,success=True,gpus=8,batch=64)),flush=True)
    finally:
        for process in processes:
            if process.poll() is None:process.terminate()
        for log in logs:log.close()


if __name__=='__main__':main()
