"""Native model block streaming under the current GPU7 memory allocation."""
import argparse
import copy
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1];SELF=Path(__file__).resolve()
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT.parent/'OmniResponse')]
import p2_responsenet_full586_all_lora_20261111 as task
from p2_testsets_eval.protocol import read,save,sha,PYTHON,METRIC_PYTHON
import p2_eval_block_streaming_20261111 as streaming


def initialize():
    assert os.environ['CUDA_VISIBLE_DEVICES']=='7'
    from p2_whisper_av.bootstrap import activate_ltx
    activate_ltx()
    import torch
    torch.set_num_threads(4);torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(20/(torch.cuda.get_device_properties(0).total_memory/2**30),0)
    from p2_testsets_eval import runtime as rt
    from p2_listen.config import extractor_config
    d=read(task.COMMON/'DATA_CONFIG.json');return torch,rt,d,extractor_config(d)


def gemma(smoke=False):
    torch,rt,d,cfg=initialize();p=task.verify(task.COMMON)
    items=read(task.COMMON/'shared_items.json');p=copy.deepcopy(p);p['text_batch']=2
    assert all(rt.ready(task.COMMON/'conditions'/x['key']/'qwen') for x in items)
    if smoke:
        from p3_listen.prompts import FixedPrompt
        selected=[x for x in items if rt.ready(task.COMMON/'conditions'/x['key']/'text')][:2]
        encoder=streaming.text_conditioner(cfg);records=[]
        save(task.WORK/'STREAM_GEMMA_LAYOUT.json',encoder.encoder.model._eval_stream_info)
        for x in selected:
            f=task.COMMON/'conditions'/x['key']
            reference=torch.load(f/'text/payload.pt',map_location='cpu',weights_only=True)
            value=encoder.encode_batch([FixedPrompt(reference['prompt'])])[0]
            rec=dict(key=x['key'],prompt_equal=value['prompt']==reference['prompt'])
            for key in ('video','audio','mask'):
                a=value[key].detach().cpu();b=reference[key];assert a.shape==b.shape and a.dtype==b.dtype
                rec[key]=dict(bitwise_equal=torch.equal(a,b),max_abs=float((a.float()-b.float()).abs().max()),
                    mean_abs=float((a.float()-b.float()).abs().mean()),reference_max=float(b.float().abs().max()))
            records.append(rec)
        save(task.WORK/'STREAM_GEMMA_SMOKE.json',dict(at=time.time(),records=records,source_sha256=sha(ROOT/'src/p2_eval_block_streaming_20261111.py'),
            note='Reference was historical batched native cache; all source weights are copied bitwise, native forward/precision unchanged',
            gpu_peak_mib=torch.cuda.max_memory_allocated()/2**20))
        encoder.close();return
    import omni_av.ltx_runtime as ltx
    with patch.object(rt,'data_config',lambda:(d,cfg)),patch.object(ltx,'NativeTextConditioner',streaming.text_conditioner):
        rt.cache(task.COMMON,items,0,p)
    assert all(rt.ready(task.COMMON/'conditions'/x['key']/'text') for x in items)
    save(task.COMMON/'CACHE_COMPLETE.json',dict(success=True,at=time.time(),pairs=586,planner_step=2752,
        Gemma_execution='Original native bf16 forward, CPU-pinned weight streaming, textbatch2',streaming_sha256=sha(ROOT/'src/p2_eval_block_streaming_20261111.py')))


def render(step=6000,smoke=False):
    torch,rt,d,cfg=initialize()
    if smoke:
        source=ROOT/'experiments/p2_vico43_all_lora6000_sft2752_20261111'
        work=task.WORK/'stream_validation/vico6000';work.mkdir(parents=True,exist_ok=True)
        p=read(source/'protocol.json');p.update(shared_item_queue=False)
        save(work/'protocol.json',p)
        for name in ('export','conditions','whisper','negative'):task.link(source/name,work/name)
        save(work/'DATA_CONFIG.json',read(source/'DATA_CONFIG.json'))
        d=read(work/'DATA_CONFIG.json');from p2_listen.config import extractor_config
        cfg=extractor_config(d)
        all_items=read(source/'shards/00.json');items=[min(all_items,key=lambda x:x['geometry']['native_frames'])]
    else:
        work=task.run_work(step);p=task.verify(work);items=read(task.COMMON/'shared_items.json')
        d=read(work/'DATA_CONFIG.json');from p2_listen.config import extractor_config
        cfg=extractor_config(d)
    import p2_ltx6000_latest_sft_dual200_20261107 as native
    import p2_ltx_events.inference as previous
    from p2_testsets_eval import shared_items
    original_claim=shared_items.claim_items
    def bounded_claim(*args,**kwargs):
        iterator=original_claim(*args,**kwargs)
        try:
            for index,item in enumerate(iterator):
                if index>=8:break
                yield item
        finally:iterator.close()
    def load(path,device='cuda:0'):
        model,strategy,sampler=streaming.render_loader(work,path,device,step)
        assert len(model.text_targets)==1632
        save(work/'STREAM_MODEL_ACCEPTANCE.json',dict(passed=True,at=time.time(),step=step,
            strict_original_loader=True,targets=1632,layout=model.renderer._eval_stream_info,
            adapter_sha256=sha(ROOT/'src/p2_eval_block_streaming_20261111.py')))
        return model,strategy,sampler
    loader=task.loading(step,work)
    with patch.object(rt,'data_config',lambda:(d,cfg)),patch.object(previous,'load_model',load),\
         patch.object(native,'ARM',task.arm(step)),patch.object(shared_items,'claim_items',bounded_claim):
        (native.render if step==6000 else loader.render)(work,items,0,p)
    if smoke:
        x=items[0];f='latents/'+task.arm(step)+'/'+x['key']+'/payload.pt'
        value=torch.load(work/f,map_location='cpu',weights_only=True)
        expected=torch.load(source/f,map_location='cpu',weights_only=True)
        report=dict(at=time.time(),id=x['id'],seed=p['seed'],fields={},passed=True)
        for key in ('video','audio'):
            a=value[key];b=expected[key]
            close=torch.allclose(a,b,atol=1e-5,rtol=1e-5)
            report['fields'][key]=dict(bitwise_equal=torch.equal(a,b),max_abs=float((a-b).abs().max()),
                mean_abs=float((a-b).abs().mean()),allclose_1e_5=close)
            report['passed']&=close
        report['GPU_peak_mib']=torch.cuda.max_memory_allocated()/2**20
        save(task.WORK/'STREAM_RENDER_SMOKE.json',report)
        assert report['passed'],report


def launch(stage,step=6000):
    log=task.WORK/'streaming_logs'/f'{stage}_{step}_{time.time_ns()}.log';log.parent.mkdir(exist_ok=True)
    command=[str(PYTHON),'-u',str(SELF),stage,'--step',str(step)]
    with log.open('ab',buffering=0) as stream:
        p=subprocess.Popen(command,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='7',PYTHONPATH='',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='2',PYTORCH_ALLOC_CONF='expandable_segments:True',TOKENIZERS_PARALLELISM='false'),
            stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
    save(task.WORK/'streaming_processes'/f'{stage}_{step}.json',dict(pid=p.pid,stage=stage,at=time.time(),log=str(log),command=command,physical_gpu=7))
    return p,log


def probe():
    for stage in ('gemma-smoke','render-smoke'):
        p,log=launch(stage)
        if p.wait():raise RuntimeError('Streaming validation failed: '+str(log))
    save(task.WORK/'STREAM_VALIDATION_FINISHED.json',dict(at=time.time(),render=read(task.WORK/'STREAM_RENDER_SMOKE.json'),gemma=read(task.WORK/'STREAM_GEMMA_SMOKE.json')))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['probe','gemma-smoke','render-smoke','gemma','render'])
    p.add_argument('--step',type=int,choices=[6000,6500],default=6000);a=p.parse_args()
    if a.action=='probe':probe()
    elif a.action.startswith('gemma'):gemma(a.action=='gemma-smoke')
    else:render(a.step,a.action=='render-smoke')
