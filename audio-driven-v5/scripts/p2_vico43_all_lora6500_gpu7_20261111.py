"""ViCo test43, all-attention/FFN LoRA6500, native accelerated single-GPU path."""
import argparse
import concurrent.futures
import copy
import fcntl
import inspect
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from contextlib import contextmanager
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
SELF=Path(__file__).resolve()
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'compare/deployment')]
from p2_testsets_eval.protocol import read,save,sha

WORK=ROOT/'experiments/p2_vico43_all_lora6500_sft2752_20261111'
SOURCE=ROOT/'experiments/p2_vico43_combined8000_sft2752_20261107'
EXPORT=WORK/'export'
TRAIN=ROOT/'experiments/p2_ltx_all_lora_full28_batch28_20261109'
ARM='all_lora6500_sft2752'
PY=ROOT.parent/'Omni-LTX-stage2/.venv/bin/python'
MPY=ROOT/'compare/.envs/vico_eval/bin/python'
MORE=ROOT/'compare/.envs/vico_more/bin/python'
UUID='GPU-4d5ee808-e829-c824-c506-cad80ff133f5'


def link(src,dst):
    src,dst=Path(src),Path(dst)
    dst.parent.mkdir(parents=True,exist_ok=True)
    if dst.exists():assert dst.resolve()==src.resolve(),dst
    else:dst.symlink_to(src,target_is_directory=src.is_dir())


def followup():
    import p2_all_lora_followups_eval_20261109 as loading
    # The user explicitly requested6500 from the same complete-model lineage.
    # Extend this process-local selection without editing the old queue whitelist.
    loading.STEP=6500
    loading.TRAIN=TRAIN
    loading.WORK=WORK
    loading.ARM=ARM
    return loading


def prepare(work):
    work.mkdir(parents=True,exist_ok=True)
    dest=work/'protocol.json'
    if dest.exists():return verify(work)
    from p2_whisper_av.bootstrap import activate_ltx
    activate_ltx()
    from p2_ltx_all_lora.config import source_identity,validate
    from p2_ltx_all_lora.data import data_config
    loading=followup()
    from p2_testsets_eval.runtime import ready
    import p2_vico256_combined_eval as legacy
    export_started=time.time()
    rec=loading.export(work)
    export_seconds=time.time()-export_started
    assert rec['success'] and rec['all_model_tensors_bitwise_equal'] and rec['step']==6500
    original=TRAIN/'step_006500.pt'
    assert rec['original']==str(original) and rec['original_sha256']==read(original.with_suffix('.complete.json'))['sha256']
    assert loading.stamp(rec['checkpoint'])==rec['stamp']
    assert rec['identity']['source']==loading.expected_source()
    c=loading.model_config(rec['identity']['experiment']);assert len(rec['identity']['text_targets'])==1632
    # The established export checked() already verifies the full source SHA.
    started=time.time();original_sha=rec['original_sha256'];assert original_sha==read(original.with_suffix('.complete.json'))['sha256']
    parent=read(SOURCE/'protocol.json')
    assert parent['generation']==dict(steps=30,video_cfg=3,audio_cfg=7) and parent['seed']==137
    assert read(SOURCE/'COMPLETE.json')
    items=read(SOURCE/'shards/00.json');assert len(items)==len({x['id'] for x in items})==43
    identities={}
    for item in items:
        folder=SOURCE/'conditions'/item['key']
        assert ready(folder/'qwen') and ready(folder/'text') and ready(SOURCE/'whisper'/item['key'])
        plan=read(folder/'qwen/plan.json')
        assert plan['adapter']==parent['planner_adapter_identity'] and not plan['target_description_access']
        _,source=legacy.source_wave(item,folder)
        assert source['waveform_sha256']==read(SOURCE/'whisper'/item['key']/'metadata.json')['source']['waveform_sha256']
        identities[item['id']]=dict(image_sha256=sha(folder/'first_frame.png'),input_audit_sha256=sha(folder/'input_audit.json'),
            planning_sha256=sha(folder/'qwen/plan.json'),source_waveform_sha256=source['waveform_sha256'])
    for name in ['conditions','whisper','negative']:
        link(SOURCE/name,work/name)
    link(EXPORT,work/'export')
    for name in ['INPUT_PREFLIGHT.json','shards/00.json']:
        (work/name).parent.mkdir(parents=True,exist_ok=True);(work/name).write_bytes((SOURCE/name).read_bytes())
    bins=[[] for _ in range(4)];loads=[0]*4
    for item in sorted(items,key=lambda x:(-x['geometry']['native_frames'],x['id'])):
        i=min(range(4),key=lambda j:(loads[j],j));bins[i].append(item);loads[i]+=item['geometry']['native_frames']
    for i,rows in enumerate(bins):save(work/'decode_shards'/f'{i:02d}.json',rows)
    p=copy.deepcopy(parent)
    p.update(schema='vico43-full-attention-ffn-lora6500-single-GPU7-v1',created_at=time.time(),
        arms=[dict(name=ARM,kind='ltx_all_lora_full',step=6500,checkpoint=rec['checkpoint'],checkpoint_sha256=rec['checkpoint_sha256'],
            original_checkpoint=str(original),original_sha256=original_sha,planner=parent['planner'])],
        checkpoint_identity=rec['identity'],render_gpus=[7],render_shards=1,workers=1,
        shared_item_queue=False,worker_cli=str(SELF),authorized_scope='node11 physicalGPU7 only; preserve unrelated jobs',
        metric_groups=['official_image_identity5','CustomListener_public_definition_motion','DIM_EMOCA12'],
        DIM_variance='mean_j population_Var_over_pooled_valid_frames(coeff_j); also preserve published flattened formula',
        acceleration='Reused SFT2752/H/Gemma/Whisper/negative caches; verified mmap model export; separate DiT and VAE lifetimes; up to4 disjoint decoders with memory admission; CPU CPBD overlaps decoding',
        training_batch28_continuation=rec['identity']['experiment']['batch28_continuation'],
        original_protocol=str(SOURCE/'protocol.json'),original_protocol_sha256=sha(SOURCE/'protocol.json'),
        input_identity_sha256=sha(work/'INPUT_PREFLIGHT.json'),decoder_loads=loads)
    files=[SELF,ROOT/'src/p2_all_lora_followups_eval_20261109.py',ROOT/'scripts/p2_ltx6000_latest_sft_dual200_20261107.py',
        ROOT/'scripts/p2_vico256_combined_eval.py',ROOT/'src/p2_testsets_eval/runtime.py',ROOT/'src/p2_testsets_eval/shared_items.py',
        ROOT/'src/p2_testsets_eval/whisper_runtime.py',ROOT/'scripts/p2_vico_official_metrics.py']
    p['execution_code']={str(f):sha(f) for f in files}
    save(work/'DATA_CONFIG.json',data_config(c));save(work/'SOURCE_INPUTS.json',identities)
    save(dest,p)
    save(work/'CPU_ACCEPTANCE.json',dict(passed=True,official_test43=True,source_only=True,full_checkpoint_sha256=original_sha,
        verified_export_seconds=export_seconds,full_checkpoint_hash_verified_by='existing followup export / checked()',export_receipt_sha256=sha(EXPORT/'RECEIPT.json'),
        same_model_tensors_as_user_checkpoint=True,LoRA_targets=1632,seed=137,SFT_step=2752,all_training_sources_exact=True,training_source_count=len(rec['identity']['source']),original_export_raw_SHA_verified=True))
    return p


def verify(work):
    p=read(work/'protocol.json')
    for path,value in p['execution_code'].items():assert sha(path)==value,path
    assert sha(work/'INPUT_PREFLIGHT.json')==p['input_identity_sha256']
    assert sha(SOURCE/'protocol.json')==p['original_protocol_sha256']
    return p


def gpu():
    row=subprocess.check_output(['nvidia-smi','-i','7','--query-gpu=uuid,memory.free,memory.total','--format=csv,noheader,nounits'],text=True).strip().split(',')
    assert row[0].strip()==UUID
    return int(row[1]),int(row[2])


def worker(work,stage,slot,memory_gib):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='7'
    p=verify(work);free,total=gpu();assert free>=memory_gib*1024+16384
    from p2_whisper_av.bootstrap import activate_ltx
    activate_ltx()
    import torch
    torch.set_num_threads(4);torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(memory_gib*1024/total)
    from p2_testsets_eval import runtime as rt
    from p2_listen.config import extractor_config
    import p2_vico256_combined_eval as legacy
    d=read(work/'DATA_CONFIG.json');cfg=extractor_config(d)
    import omni_av.data as data
    original_observation=data.Observation
    class FullViCoObservation(original_observation):
        def __post_init__(self):
            if self.output_duration_s<=30:return super().__post_init__()
            assert self.sample_id and 0<=self.speaker_start_s<self.speaker_end_s<=self.output_duration_s<=35
    data.Observation=FullViCoObservation
    progress=rt.progress
    with (work/f'{stage}_{slot}.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        save(work/'devices'/f'{stage}_{slot}.json',dict(pid=os.getpid(),host=os.uname().nodename,
            GPU_UUID=UUID,physical_gpu=7,memory_cap_gib=memory_gib,free_before_mib=free,at=time.time()))
        with patch.object(rt,'data_config',lambda:(d,cfg)),patch.object(rt,'progress',lambda w,s,r,**kw:progress(w,s,slot,**kw)):
            if stage=='latent':
                import p2_ltx_events.inference as old
                all_lora=followup()
                def load(path,device='cuda:0'):
                    result=all_lora.load_model(work,path,device)
                    assert len(result[0].text_targets)==1632
                    save(work/'MODEL_ACCEPTANCE.json',dict(passed=True,strict_all_LoRA_load=True,targets=1632,
                        original_sha256=p['arms'][0]['original_sha256'],export_sha256=p['arms'][0]['checkpoint_sha256'],pid=os.getpid()))
                    return result
                with patch.object(old,'load_model',load):
                    all_lora.render(work,read(work/'shards/00.json'),0,p)
                assert sum(rt.ready(work/'latents'/ARM/x['key']) for x in read(work/'shards/00.json'))==43
                save(work/'LATENT_COMPLETE.json',dict(passed=True,outputs=43,at=time.time()))
            else:
                assert stage=='decode'
                namespace=dict(legacy.__dict__,ARM=ARM,CURRENT={})
                def scoped_save(path,value):
                    if Path(path)==work/'status.json':path=work/'progress/decode_export'/f'{slot:02d}.json'
                    save(path,value)
                namespace['save']=scoped_save
                source=inspect.getsource(legacy.render_export)
                old_call="wr.render(work,items,0,p,p['arms'])";assert source.count(old_call)==1
                source=source.replace(old_call,'None # Decode committed latents without loading DiT')
                source=source.replace('if done==2:','if False:')
                needle="if len(list((work/'conversion').glob('*.verified.json')))!=43:raise ValueError('Incomplete scored media')"
                assert source.count(needle)==1
                source=source.replace(needle,"if any(not (work/'conversion'/f'{x[\"id\"]}.verified.json').exists() for x in items):raise ValueError('Incomplete decode shard')")
                source=source.replace("work/'GENERATION_COMPLETE.json'",f"work/'DECODE_SHARD_{slot:02d}.json'")
                source=source.replace('dict(outputs=43,at=time.time()', 'dict(outputs=len(items),at=time.time()')
                exec(compile(source,str(SELF)+':native_decode_export','exec'),namespace)
                with rt.cached_loaders(['load_video_vae_decoder','load_audio_vae_decoder','load_vocoder']):
                    namespace['render_export'](work,read(work/'decode_shards'/f'{slot:02d}.json'),p,rt,None)


def metric(work,child=None,index=None):
    import p2_vico_official_metrics as m
    m.ARM=ARM;m.__file__=str(SELF)
    if child:return m.child(work,child,index)
    m.run(work)
    r=read(work/'RESULTS.json');r.update(model='All attention/FFN LoRA6500 + SFT2752',checkpoint=str(TRAIN/'step_006500.pt'))
    save(work/'IMAGE_RESULTS.json',r)


def cpbd(work):
    import p2_vico_official_metrics as m
    m.ARM=ARM;m.__file__=str(SELF)
    import pandas as pd
    ids=pd.read_csv(read(work/'protocol.json')['datasets']['vico']['annotation']).uuid.tolist()
    futures={};done=set()
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        while len(done)<43:
            for i,future in list(futures.items()):
                if future.done():future.result();done.add(i);del futures[i]
            for i,uid in enumerate(ids):
                if i in done or i in futures:continue
                if (work/'conversion'/(uid+'.verified.json')).exists():futures[i]=pool.submit(m.execute,work,'cpbd',i)
            time.sleep(5)
    save(work/'CPBD_OVERLAP_COMPLETE.json',dict(passed=True,clips=43,at=time.time()))


def run(work):
    p=verify(work)
    with (work/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        save(work/'CONTROLLER.json',dict(pid=os.getpid(),at=time.time(),host=os.uname().nodename,physical_gpu=7))
        children=[]
        def start(action,python=PY,slot=0,cap=0):
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='' if action=='cpbd' else '7',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='2',
                PYTHONPATH='',PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false',PYTORCH_ALLOC_CONF='expandable_segments:True')
            args=[str(python),'-u',str(SELF),action,'--work',str(work),'--slot',str(slot),'--memory-gib',str(cap)]
            log=work/'logs'/f'{action}_{slot}_{time.time_ns()}.log';log.parent.mkdir(exist_ok=True)
            with log.open('a') as stream:proc=subprocess.Popen(args,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
            children.append(proc);save(work/'processes'/f'{action}_{slot}.json',dict(pid=proc.pid,stage=action,slot=slot,memory_cap_gib=cap,log=str(log)))
            return proc,log
        def status(stage,**extra):
            save(work/'status.json',dict(stage=stage,pid=os.getpid(),latents=len(list((work/'latents'/ARM/'vico').glob('*/complete.json'))),
                videos=len(list((work/'conversion').glob('*.verified.json'))),total=43,at=time.time(),**extra))
        try:
            if not (work/'LATENT_COMPLETE.json').exists():
                while True:
                    free,_=gpu()
                    if free>=80*1024:break
                    status('waiting_GPU7_memory',free_mib=free,required_mib=80*1024);time.sleep(20)
                cap=min(96.,free/1024-16.5)
                status('generating_latents',memory_cap_gib=cap)
                proc,log=start('latent',cap=cap);code=proc.wait()
                if code:raise RuntimeError(f'Latent worker exited{code}: {log}')
            assert read(work/'LATENT_COMPLETE.json')['outputs']==43
            cpbd_proc=None
            if not (work/'CPBD_OVERLAP_COMPLETE.json').exists():cpbd_proc,_=start('cpbd',MPY)
            active={};pending=[i for i in range(4) if not (work/f'DECODE_SHARD_{i:02d}.json').exists()]
            while active or pending:
                for slot,(proc,log) in list(active.items()):
                    if proc.poll() is not None:
                        if proc.returncode:raise RuntimeError(f'Decode shard{slot} exited{proc.returncode}: {log}')
                        del active[slot]
                free,total=gpu();budget=29*1024
                # Reserve the remaining hard budget of live processes, plus16GiB.
                text=subprocess.check_output(['nvidia-smi','-i','7','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True)
                used={int(a):int(b) for line in text.splitlines() if len(v:=line.replace(' ','').split(','))==2 and (a:=v[0]).isdigit() and (b:=v[1]).isdigit()}
                required=16384+budget+sum(max(0,budget-used.get(proc.pid,0)) for proc,_ in active.values())
                if pending and len(active)<4 and free>=required:
                    slot=pending.pop(0);active[slot]=start('decode',slot=slot,cap=28.)
                status('decoding_and_CPU_scoring',pending_decode_shards=pending,active_decode_pids=[x[0].pid for x in active.values()])
                time.sleep(10)
            assert len(list((work/'conversion').glob('*.verified.json')))==43
            if cpbd_proc is not None:assert cpbd_proc.wait()==0
            save(work/'GENERATION_COMPLETE.json',dict(passed=True,outputs=43,at=time.time()))
            status('official_image_identity_metrics')
            proc,log=start('metrics',MPY);assert proc.wait()==0,str(log)
            status('motion_and_DIM_metrics')
            # Follow-up scoring owns no inference model and never touches other cards.
            with (work/'motion_pipeline.log').open('a') as log:
                subprocess.run([str(MORE),'-u',str(ROOT/'scripts/p2_vico43_all_lora6500_metrics_20261111.py'),'run','--work',str(work)],
                    cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='7',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='2',PYTHONPATH=''),
                    stdout=log,stderr=subprocess.STDOUT,check=True)
            assert read(work/'FINAL_AUDIT.json')['passed']
            save(work/'COMPLETE.json',dict(passed=True,outputs=43,at=time.time(),protocol_sha256=sha(work/'protocol.json')))
            status('complete')
        except BaseException as error:
            status('failed',error=repr(error))
            for proc in children:
                if proc.poll() is None:proc.terminate()
            raise


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',nargs='?',default='metrics',choices=['prepare','run','latent','decode','metrics','cpbd'])
    parser.add_argument('--work',type=Path,default=WORK);parser.add_argument('--slot',type=int,default=0);parser.add_argument('--memory-gib',type=float,default=0)
    parser.add_argument('--child');parser.add_argument('--index',type=int)
    a=parser.parse_args();work=a.work.resolve()
    if a.child:metric(work,a.child,a.index)
    elif a.action=='prepare':prepare(work)
    elif a.action=='run':run(work)
    elif a.action in ['latent','decode']:worker(work,a.action,a.slot,a.memory_gib)
    elif a.action=='cpbd':cpbd(work)
    else:metric(work)
