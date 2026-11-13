"""Native model block streaming under the current GPU7 memory allocation."""
import argparse
import copy
import gc
import fcntl
import statistics
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


def verify_stream():
    for path,digest in read(task.WORK/'STREAM_PROTOCOL.json')['code'].items():assert sha(path)==digest,path


def initialize():
    verify_stream()
    assert os.environ['CUDA_VISIBLE_DEVICES']=='7'
    from p2_whisper_av.bootstrap import activate_ltx
    activate_ltx()
    import torch
    torch.set_num_threads(4);torch.cuda.set_device(0)
    cap=float(os.environ.get('P2_STREAM_MEMORY_GIB','20'))
    torch.cuda.set_per_process_memory_fraction(cap/(torch.cuda.get_device_properties(0).total_memory/2**30),0)
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
    original_class=ltx.NativeTextConditioner
    factory=lambda config,device='cuda:0':streaming.text_conditioner(config,device,_base=original_class)
    with patch.object(rt,'data_config',lambda:(d,cfg)),patch.object(ltx,'NativeTextConditioner',factory):
        rt.cache(task.COMMON,items,0,p)
    assert all(rt.ready(task.COMMON/'conditions'/x['key']/'text') for x in items)
    save(task.COMMON/'CACHE_COMPLETE.json',dict(success=True,at=time.time(),pairs=586,planner_step=2752,
        Gemma_execution='Original native bf16 forward, CPU-pinned weight streaming, textbatch2',streaming_sha256=sha(ROOT/'src/p2_eval_block_streaming_20261111.py')))


def render(step=6000,smoke=False,slot=0):
    torch,rt,d,cfg=initialize()
    if smoke:
        source=ROOT/f'experiments/p2_vico43_all_lora{step}_sft2752_20261111'
        work=task.WORK/f'stream_validation/vico{step}';work.mkdir(parents=True,exist_ok=True)
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
        save(work/('STREAM_MODEL_ACCEPTANCE.json' if slot==0 else f'STREAM_MODEL_ACCEPTANCE_{slot}.json'),dict(passed=True,at=time.time(),step=step,
            strict_original_loader=True,targets=1632,layout=model.renderer._eval_stream_info,
            adapter_sha256=sha(ROOT/'src/p2_eval_block_streaming_20261111.py')))
        return model,strategy,sampler
    loader=task.loading(step,work)
    original_progress=rt.progress
    def measured_progress(w,s,r,**kw):
        return original_progress(w,s,slot,**kw,GPU_peak_mib=torch.cuda.max_memory_allocated()/2**20)
    with patch.object(rt,'data_config',lambda:(d,cfg)),patch.object(previous,'load_model',load),\
         patch.object(native,'ARM',task.arm(step)),patch.object(shared_items,'claim_items',bounded_claim),\
         patch.object(rt,'progress',measured_progress):
        (native.render if step==6000 else loader.render)(work,items,0,p)
    save(work/('STREAM_RENDER_BATCH_LAST.json' if slot==0 else f'STREAM_RENDER_BATCH_LAST_{slot}.json'),dict(at=time.time(),step=step,
        GPU_peak_mib=torch.cuda.max_memory_allocated()/2**20,adapter_sha256=sha(ROOT/'src/p2_eval_block_streaming_20261111.py')))
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
        save(task.WORK/('STREAM_RENDER_SMOKE.json' if step==6000 else 'STREAM_RENDER6500_SMOKE.json'),dict(report,step=step))
        assert report['passed'],report


def launch(stage,step=6000,rank=0,memory=20):
    log=task.WORK/'streaming_logs'/f'{stage}_{step}_{time.time_ns()}.log';log.parent.mkdir(exist_ok=True)
    command=[str(METRIC_PYTHON if stage=='metric' else PYTHON),'-u',str(SELF),stage,'--step',str(step),'--rank',str(rank)]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='7',PYTHONPATH='/private/liqi_workspace/.cache/paired_video_metrics/python' if stage=='metric' else '',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='2',PYTORCH_ALLOC_CONF='expandable_segments:True',TOKENIZERS_PARALLELISM='false',P2_EVAL_MEMORY_GIB=str(memory),P2_STREAM_MEMORY_GIB=str(memory))
    with log.open('ab',buffering=0) as stream:
        p=subprocess.Popen(command,cwd=ROOT,env=env,
            stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
    save(task.WORK/'streaming_processes'/f'{stage}_{step}_{rank}.json',dict(pid=p.pid,stage=stage,at=time.time(),log=str(log),command=command,physical_gpu=7,memory_cap_gib=memory))
    return p,log


def metric(step,rank):
    verify_stream();task.verify(task.run_work(step))
    import p2_testsets_eval.metrics as native
    from p2_pipelined_testsets_eval import pipelined_command
    original=native.subprocess.run
    bootstrap="import runpy,sys,torch; torch.cuda.set_device(0); torch.cuda.set_per_process_memory_fraction(20/(torch.cuda.get_device_properties(0).total_memory/2**30),0); args=sys.argv[1:]; module=args[0]=='-m'; sys.argv=args[1:] if module else args; runpy.run_module(args[1],run_name='__main__') if module else runpy.run_path(args[0],run_name='__main__')"
    def execute(command,*args,**kw):
        cmd=list(pipelined_command(command))
        if cmd and cmd[0]==str(METRIC_PYTHON):cmd=[cmd[0],'-c',bootstrap,*cmd[1:]]
        return original(cmd,*args,**kw)
    with patch.object(native.subprocess,'run',execute):native.metrics_worker(task.run_work(step),rank,[task.arm(step)])


def checked(stage,step=6000,rank=0):
    for attempt in range(3):
        while task.gpu()['free_gib']<22:time.sleep(30)
        process,log=launch(stage,step,rank)
        if process.wait()==0:return
        save(task.WORK/'STREAM_RETRY.json',dict(stage=stage,step=step,rank=rank,attempt=attempt+1,at=time.time(),log=str(log)))
        if attempt==2:raise RuntimeError('Repeated streaming stage failure: '+str(log))
        time.sleep(15)


def resume_native():
    stop=read(task.WORK/'STOP');assert stop['owner_script']==SELF.name
    (task.WORK/'STOP').unlink()
    command=[str(PYTHON),'-u',str(task.SELF),'controller']
    with (task.WORK/'streaming_native_handoff.log').open('ab',buffering=0) as log:
        p=subprocess.Popen(command,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'),
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    save(task.WORK/'LAUNCH.json',dict(pid=p.pid,at=time.time(),command=command,reason='Enough GPU capacity; native resident model resumes existing outputs'))
    save(task.WORK/'STREAM_NATIVE_HANDOFF.json',dict(at=time.time(),pid=p.pid))


def wait_previous_workers():
    # A controller-only repair preserves in-flight GPU work and adopts its
    # committed outputs. Only exact recorded command/PID pairs are observed.
    while True:
        active=[]
        for f in (task.WORK/'streaming_processes').glob('*.json'):
            r=read(f);proc=Path('/proc')/str(r['pid'])
            try:
                stat=proc.joinpath('stat').read_text().rsplit(')',1)[1].split()
                cmd=proc.joinpath('cmdline').read_bytes().replace(bytes([0]),b' ').decode(errors='replace')
            except FileNotFoundError:continue
            if stat[0]!='Z' and all(str(x) in cmd for x in r['command'][1:]):active.append(r['pid'])
        if not active:return
        save(task.WORK/'STREAM_ADOPTING.json',dict(at=time.time(),preserved_gpu_workers=active))
        current=read(task.WORK/'status.json') if (task.WORK/'status.json').exists() else {}
        step=current.get('step',6000);step=step if step in (6000,6500) else 6000
        latents,videos=task.counts(step)
        save(task.WORK/'status.json',dict(stage='adopting_inflight_render',at=time.time(),step=step,
            latents=latents,videos=videos,total=586,preserved_gpu_workers=active))
        time.sleep(15)


def render_batch(step):
    work=task.run_work(step);peak_file=work/'STREAM_RENDER_BATCH_LAST.json'
    known_peak=read(peak_file)['GPU_peak_mib']/1024 if peak_file.exists() else None
    disabled=task.WORK/'STREAM_PARALLEL_DISABLED.json'
    if disabled.exists() or known_peak is None or known_peak>8.5:
        checked('render',step);return
    while task.gpu()['free_gib']<22:time.sleep(30)
    before=task.counts(step)[0];begin=time.monotonic();jobs=[]
    baseline=[]
    for f in (work/'latents'/task.arm(step)/'responsenet').glob('*/metadata.json'):
        m=read(f);baseline.append((f.stat().st_mtime,m['seconds']))
    baseline=statistics.median(v for _,v in sorted(baseline)[-8:]) if baseline else None
    # The observed maximum-length512 clip peaks at7.5GiB. Two9.5GiB
    # process budgets leave context/headroom inside the current23GiB.
    for rank in (0,1):jobs.append(launch('render',step,rank,memory=9.5))
    failures=[]
    for process,log in jobs:
        if process.wait():failures.append(str(log))
    completed=task.counts(step)[0]-before;seconds=time.monotonic()-begin
    evidence=dict(at=time.time(),step=step,completed=completed,seconds=seconds,
        seconds_per_clip=seconds/completed if completed else None,previous_single_clip_median=baseline,
        concurrency=2,per_process_cap_gib=9.5,failures=failures,comparison='Different production clips; same max-frame bucket, no re-generation')
    save(task.WORK/'STREAM_PARALLEL_LAST.json',evidence)
    if failures or not completed or (baseline and seconds/completed>baseline*1.1):
        save(disabled,evidence)
        if failures:checked('render',step)


def controller():
    with (task.WORK/'stream_controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        verify_stream()
        assert read(task.WORK/'STREAM_RENDER_SMOKE.json')['passed']
        assert read(task.WORK/'STREAM_RENDER6500_SMOKE.json')['passed']
        assert read(task.WORK/'STOP')['owner_script']==SELF.name
        save(task.WORK/'STREAM_CONTROLLER.json',dict(pid=os.getpid(),at=time.time(),physical_gpus=[7]))
        wait_previous_workers()
        if not (task.COMMON/'CACHE_COMPLETE.json').exists():
            save(task.WORK/'status.json',dict(stage='streaming_Gemma_conditions',at=time.time(),physical_gpu=7))
            checked('gemma')
        for step in (6000,6500):
            if (task.run_work(step)/'HANDOFF_COMPLETE.json').exists():continue
            if task.counts(step)[0]>task.counts(step)[1]:checked('decode',step)
            while True:
                if (task.WORK/'STREAM_STOP').exists():return
                latents,videos=task.counts(step)
                if videos==586:break
                if task.gpu()['free_gib']>=64:resume_native();return
                save(task.WORK/'status.json',dict(stage='streaming_render_decode',step=step,at=time.time(),latents=latents,videos=videos,total=586,physical_gpu=7))
                if latents<586:render_batch(step)
                checked('decode',step)
            process=task.launch('bind',step);task.wait_job(process)
            for rank in range(8):
                save(task.WORK/'status.json',dict(stage='streaming_metrics',step=step,rank=rank,total_shards=8,at=time.time()))
                checked('metric',step,rank)
            task.wait_job(task.launch('merge',step))
            result=read(task.run_work(step)/'RESULTS.json')[task.arm(step)+'/responsenet']
            for name in ('psnr','ssim','lpips','fid','fvd'):
                assert result['coverage'][name]['pairs']==586,(name,result['coverage'][name])
            task.final(step)
        combined={}
        for step in (6000,6500):combined.update(read(task.run_work(step)/'RESULTS.json'))
        save(task.WORK/'RESULTS.json',combined)
        save(task.WORK/'COMPLETE.json',dict(success=True,at=time.time(),outputs=1172,pairs_per_model=586,physical_gpus=[7],execution='Original native model forward with block streaming'))
        save(task.WORK/'status.json',dict(stage='complete',outputs=1172,at=time.time()))
        if (task.WORK/'STOP').exists() and read(task.WORK/'STOP').get('owner_script')==SELF.name:(task.WORK/'STOP').unlink()


def probe():
    for stage in ('gemma-smoke','render-smoke'):
        p,log=launch(stage)
        if p.wait():raise RuntimeError('Streaming validation failed: '+str(log))
    save(task.WORK/'STREAM_VALIDATION_FINISHED.json',dict(at=time.time(),render=read(task.WORK/'STREAM_RENDER_SMOKE.json'),gemma=read(task.WORK/'STREAM_GEMMA_SMOKE.json')))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['probe','gemma-smoke','render-smoke','gemma','render','decode','metric','controller'])
    p.add_argument('--step',type=int,choices=[6000,6500],default=6000);p.add_argument('--rank',type=int,default=0);a=p.parse_args()
    if a.action=='probe':probe()
    elif a.action=='controller':
        try:controller()
        except Exception as exc:
            save(task.WORK/'status.json',dict(stage='streaming_failed',error=repr(exc),at=time.time()))
            raise
    elif a.action=='decode':task.worker('decode',a.step,0)
    elif a.action=='metric':metric(a.step,a.rank)
    elif a.action.startswith('gemma'):gemma(a.action=='gemma-smoke')
    else:render(a.step,a.action=='render-smoke',a.rank)
