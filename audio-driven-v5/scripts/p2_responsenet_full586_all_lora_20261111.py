"""Full sealed ResponseNet586, two complete LoRA models, node11 GPU7 only.

Reuses native SFT/Whisper, sampling, export, binder and metric formulas.
The controller waits for the already running ViCo6500 before GPU admission.
"""
import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
SELF=Path(__file__).resolve()
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT.parent/'OmniResponse')]
from p2_testsets_eval.protocol import read,save,sha,rows,geometry,partition,PYTHON,METRIC_PYTHON,OMNI

WORK=ROOT/'experiments/p2_responsenet_full586_all_lora6000_6500_20261111'
COMMON=WORK/'conditioning'
PARENT=ROOT/'experiments/p2_ltx6000_latest_sft_dual200_20261107'
BASE=ROOT/'experiments/p2_testsets_4ckpt_24gpu_20261102'
PREPARED=OMNI/'evaluation/responsenet_pairs/prepared_20261102'
UPSTREAM=ROOT/'experiments/p2_vico43_all_lora6500_sft2752_20261111'
UUID='GPU-4d5ee808-e829-c824-c506-cad80ff133f5'
REF_SHA='65dbb308d04505c276fa64f84e89f0cc05013fb7919fd79b6da6907669f27be3'


def arm(step):return f'all_lora{step}_sft2752'
def run_work(step):return WORK/f'step_{step:06d}'


def link(src,dst):
    src,dst=Path(src),Path(dst)
    dst.parent.mkdir(parents=True,exist_ok=True)
    if dst.exists() or dst.is_symlink():assert dst.resolve()==src.resolve(),str(dst)
    else:dst.symlink_to(src,target_is_directory=src.is_dir())


def code_identity():
    paths=[SELF,ROOT/'src/p2_all_lora6000_eval_20261109.py',ROOT/'src/p2_all_lora_followups_eval_20261109.py',
        ROOT/'src/p2_ltx6000_sft_eval_20261107.py',ROOT/'scripts/p2_ltx6000_latest_sft_dual200_20261107.py',
        ROOT/'scripts/p2_responsenet_random200.py',ROOT/'src/p2_testsets_eval/protocol.py',
        ROOT/'src/p2_testsets_eval/runtime.py',ROOT/'src/p2_testsets_eval/shared_items.py',
        ROOT/'src/p2_testsets_eval/whisper_runtime.py',ROOT/'src/p2_testsets_eval/cluster.py',
        ROOT/'src/p2_testsets_eval/metrics.py',ROOT/'scripts/p2_pipelined_testsets_eval.py',
        ROOT/'scripts/p2_pipelined_video_metrics.py',ROOT/'src/p2_testsets_eval/metric_pipeline.py']
    return {str(p):sha(p) for p in paths}


def verify(work):
    p=read(work/'protocol.json')
    assert p['execution_code']==code_identity(),'Evaluation source changed'
    assert sha(PREPARED/'references.jsonl')==REF_SHA
    assert sha(p['shared_items_manifest'])==p['shared_items_sha256']
    assert p['physical_gpus']==[7] and p['total_samples']==586
    return p


def loading(step,work):
    if step==6000:
        import p2_all_lora6000_eval_20261109 as m
    else:
        import p2_all_lora_followups_eval_20261109 as m
        m.STEP=6500;m.TRAIN=ROOT/'experiments/p2_ltx_all_lora_full28_batch28_20261109'
        m.WORK=work;m.ARM=arm(step)
    return m


def prepare():
    WORK.mkdir(parents=True,exist_ok=True)
    if (WORK/'PREPARED.json').exists():
        verify(COMMON)
        for step in (6000,6500):verify(run_work(step))
        return
    from p2_whisper_av.bootstrap import activate_ltx
    activate_ltx()
    from p2_testsets_eval.runtime import ready
    from p2_ltx_all_lora.data import data_config
    parent=read(PARENT/'protocol.json');assert parent['seed']==137
    refs=rows(PREPARED/'references.jsonl')
    assert sha(PREPARED/'references.jsonl')==REF_SHA==read(PREPARED/'seal.json')['references_sha256']
    assert len(refs)==len({r['id'] for r in refs})==586
    items=[dict(key='responsenet/'+r['id'],id=r['id'],dataset='responsenet',reference=r,geometry=geometry(r,'responsenet')) for r in refs]
    shared=COMMON/'shared_items.json';save(shared,items)
    existing={x['key']:x for x in read(PARENT/'shared_items.json') if x['dataset']=='responsenet'}
    assert len(existing)==200
    identities={};reuse=0
    for x in items:
        key=x['key'];src=PARENT/'conditions'/key if key in existing else BASE/'conditions'/key
        dst=COMMON/'conditions'/key
        if key in existing:
            assert existing[key]['reference']==x['reference'] and existing[key]['geometry']==x['geometry']
            assert read(src/'qwen/plan.json')['adapter']==parent['planner_adapter_identity']
            for name in ('qwen','text'):
                assert ready(src/name);link(src/name,dst/name)
            assert ready(PARENT/'whisper'/key)
            link(PARENT/'whisper'/key,COMMON/'whisper'/key);reuse+=1
        for name in ('first_frame.png','input_audit.json'):link(src/name,dst/name)
        audit=read(src/'input_audit.json')
        assert audit['source_only'] and not audit['listener_future_media_in_model_input']
        assert Path(audit['speaker_audio']).resolve()==Path(x['reference']['speaker']['audio']).resolve()
        identities[key]={name:sha(src/name) for name in ('first_frame.png','input_audit.json')}
    save(COMMON/'SOURCE_MEDIA_IDENTITY.json',identities)
    link(PARENT/'negative/00',COMMON/'negative/00')
    link(PARENT/'adapter_session',COMMON/'adapter_session')
    spec=dict(prepared=str(PREPARED),references_sha256=REF_SHA,count=586,face_padding=64)
    common=copy.deepcopy(parent)
    # Remove historical selection/queue paths; all science settings stay explicit.
    for key in ('selection','selection_sha256','execution_code','upstream_work','upstream_completion',
                'metrics_decoded_gt_cache','checkpoint_identity','arms'):
        common.pop(key,None)
    common.update(schema='responsenet-full586-all-lora-gpu7-v1',created_at=time.time(),datasets={'responsenet':spec},
        arms=[],workers=8,total_samples=586,total_outputs=586,physical_gpus=[7],worker_cli=str(SELF),
        shared_item_queue=True,shared_items_manifest=str(shared),shared_items_sha256=sha(shared),
        source_media_identity=str(COMMON/'SOURCE_MEDIA_IDENTITY.json'),
        source_media_identity_sha256=sha(COMMON/'SOURCE_MEDIA_IDENTITY.json'),
        sft_hidden_cache=str(COMMON/'adapter_session'),planner_batch=4,text_batch=8,whisper_batch=8,
        execution_code=code_identity(),authorized_scope='node11 physical GPU7 only; preserve unrelated tasks',
        upstream_completion=str(UPSTREAM/'COMPLETE.json'),
        metric_sampling='Original native eight video formulas; global586 FID/FVD from pooled features',
        audio_scoring='Same current200 pipeline: acoustic/speech activity, UTMOS with original support gates and N/A coverage',
        dataset_scope='all586 prepared speaker-to-listener pairs from135/140 sources; five zero-pair sources retained in original audit',
        frame_policy='512x512 at each pair native fps/frame count; native VAE padding removed on export; GT PIL bicubic640->512; face padding64 only for detector',
        cache_reuse=dict(SFT2752_existing=200,SFT2752_missing=386,Whisper_existing=200,Whisper_missing=386),
        group_counts={g:sum(r['group']==g for r in refs) for g in sorted({r['group'] for r in refs})},
        historical_note='Old dataset paragraph says no audio-quality model; this run follows the latest200 nine-metric pipeline, retaining UTMOS N/A/support. No GLARE/SyncNet or ViCo metrics added.')
    save(COMMON/'protocol.json',common)
    cluster=read(PARENT/'cluster.json');cluster['nodes']=[n for n in cluster['nodes'] if n['name']=='node11']
    assert len(cluster['nodes'])==1
    cluster['nodes'][0]['gpus']=[7];cluster['training_script']=str(SELF)
    for step,export in [(6000,ROOT/'experiments/p2_ltx_all_lora6000_seamless200_sft2752_eval32_20261109/export'),
                        (6500,UPSTREAM/'export')]:
        work=run_work(step);work.mkdir(exist_ok=True)
        rec=read(export/'RECEIPT.json');m=loading(step,work)
        assert rec['success'] and rec['all_model_tensors_bitwise_equal'] and rec['step']==step
        assert rec['stamp']==m.stamp(rec['checkpoint'])
        assert read(Path(rec['original']).with_suffix('.complete.json'))['sha256']==rec['original_sha256']
        c=rec['identity']['experiment'] if step==6000 else m.model_config(rec['identity']['experiment'])
        save(work/'DATA_CONFIG.json',data_config(c))
        if step==6000:save(COMMON/'DATA_CONFIG.json',data_config(c))
        link(export,work/'export')
        for name in ('conditions','whisper','negative'):link(COMMON/name,work/name)
        for rank,shard in enumerate(partition(items,8)):save(work/'shards'/f'{rank:02d}.json',shard)
        save(work/'cluster.json',cluster)
        p=copy.deepcopy(common)
        p.update(arms=[dict(name=arm(step),kind='ltx_all_lora_full',step=step,checkpoint=rec['checkpoint'],
            checkpoint_sha256=rec['checkpoint_sha256'],original_checkpoint=rec['original'],
            original_sha256=rec['original_sha256'],planner=parent['planner'])],checkpoint_identity=rec['identity'],
            cluster=str(work/'cluster.json'),metrics_cluster=str(work/'cluster.json'),
            training_global_batch=rec['identity']['experiment']['training']['global_batch'])
        save(work/'protocol.json',p)
        verify(work)
    assert reuse==200
    save(WORK/'PREPARED.json',dict(success=True,at=time.time(),pairs=586,outputs=1172,sources=135,
        source_minutes=sum(r['duration_s'] for r in refs)/60,reused_conditions=200,new_conditions=386,
        references_sha256=REF_SHA,physical_gpus=[7],steps=[6000,6500],groups=common['group_counts']))
    save(WORK/'status.json',dict(stage='waiting_vico6500',at=time.time(),pairs_per_model=586,total_outputs=1172))


def worker(stage,step=6000,slot=0):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='7'
    from p2_whisper_av.bootstrap import activate_ltx
    activate_ltx()
    import torch
    torch.set_num_threads(4);torch.cuda.set_device(0)
    memory=float(os.environ['P2_EVAL_MEMORY_GIB'])
    torch.cuda.set_per_process_memory_fraction(memory/(torch.cuda.get_device_properties(0).total_memory/2**30),0)
    from p2_testsets_eval import runtime as rt,whisper_runtime as wr
    from p2_listen.config import extractor_config
    work=COMMON if stage=='cache' else run_work(step)
    p=verify(work);items=read(COMMON/'shared_items.json')
    d=read(work/'DATA_CONFIG.json');cfg=extractor_config(d)
    path=work/'locks'/f'{stage}_{slot}.lock';path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        old=rt.progress
        with patch.object(rt,'data_config',lambda:(d,cfg)),patch.object(wr,'data_config',lambda:(d,cfg)),\
             patch.object(rt,'progress',lambda w,s,r,**kw:old(w,s,slot,**kw)):
            if stage=='cache':
                from p2_responsenet_random200 import rn_speaker_wave
                from p2_ltx6000_latest_sft_dual200_20261107 import cached_conditioning
                from p2_ltx6000_sft_eval_20261107 import SFTInferenceSession
                with patch.object(wr,'speaker_wave',rn_speaker_wave):wr.cache(work,items,0,p)
                with patch.object(rt,'conditioning',cached_conditioning),patch('p2_j_sft_hidden.Reader',SFTInferenceSession):
                    rt.cache(work,items,0,p)
                assert all(rt.ready(work/'whisper'/x['key']) and rt.ready(work/'conditions'/x['key']/'qwen')
                    and rt.ready(work/'conditions'/x['key']/'text') for x in items)
                save(COMMON/'CACHE_COMPLETE.json',dict(success=True,at=time.time(),pairs=586,planner_step=2752))
            elif stage=='render':
                import p2_ltx_events.inference as previous
                import p2_ltx6000_latest_sft_dual200_20261107 as native
                m=loading(step,work)
                def load(path,device='cuda:0'):
                    result=m.load_model(work,path,device)
                    assert len(result[0].text_targets)==1632
                    save(work/'MODEL_ACCEPTANCE.json',dict(success=True,at=time.time(),step=step,physical_gpu=7,
                        strict_all_lora_load=True,targets=1632,checkpoint_sha256=p['arms'][0]['original_sha256']))
                    return result
                with patch.object(previous,'load_model',load),patch.object(native,'ARM',arm(step)):
                    (native.render if step==6000 else m.render)(work,items,0,p)
            else:rt.decode(work,items,slot,p,p['arms'])


def gpu():
    raw=subprocess.check_output(['nvidia-smi','-i','7','--query-gpu=uuid,memory.free,memory.total,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=20)
    uid,free,total,util=[v.strip() for v in raw.split(',')]
    assert uid==UUID,'Wrong physical host/GPU'
    return dict(free_gib=float(free)/1024,total_gib=float(total)/1024,util=float(util))


def launch(stage,step=6000,slot=0,memory=28):
    work=COMMON if stage=='cache' else run_work(step)
    work.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='7',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='2',
        TOKENIZERS_PARALLELISM='false',PYTORCH_ALLOC_CONF='expandable_segments:True',
        PYTHONPATH='',P2_EVAL_MEMORY_GIB=str(memory))
    if stage=='metric':
        env['PYTHONPATH']='/private/liqi_workspace/.cache/paired_video_metrics/python'
        command=[str(METRIC_PYTHON),str(ROOT/'scripts/p2_pipelined_testsets_eval.py'),'metrics','--work',str(work),
            '--rank',str(slot),'--arms',arm(step)]
    elif stage=='bind':
        env['PYTHONPATH']='/private/liqi_workspace/.cache/paired_video_metrics/python'
        command=[str(METRIC_PYTHON),str(ROOT/'scripts/p2_testsets_eval.py'),'bind','--work',str(work)]
    elif stage=='merge':
        env['CUDA_VISIBLE_DEVICES']='';env['PYTHONPATH']='/private/liqi_workspace/.cache/paired_video_metrics/python'
        command=[str(METRIC_PYTHON),str(ROOT/'scripts/p2_testsets_eval.py'),'merge','--work',str(work)]
    else:command=[str(PYTHON),str(SELF),'worker','--stage',stage,'--step',str(step),'--slot',str(slot)]
    log=work/'logs'/f'{stage}_{slot}_{time.time_ns()}.log';log.parent.mkdir(exist_ok=True)
    with log.open('ab',buffering=0) as stream:
        process=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
    info=dict(pid=process.pid,stage=stage,step=step,slot=slot,at=time.time(),log=str(log),command=command,memory_gib=memory,physical_gpu=7)
    save(work/'processes'/f'{stage}_{slot}.json',info)
    return dict(process=process,info=info)


def wait_job(job):
    rc=job['process'].wait()
    if rc:raise RuntimeError(f"Stage failed, returncode={rc}: {job['info']['log']}")


def counts(step):
    work=run_work(step);name=arm(step)
    return (sum((work/'latents'/name/x['key']/'complete.json').exists() for x in read(COMMON/'shared_items.json')),
            sum((work/'generated'/name/x['key']/'complete.json').exists() for x in read(COMMON/'shared_items.json')))


def final(step):
    from p2_testsets_eval.runtime import ready
    work=run_work(step);p=verify(work);name=arm(step)
    for x in read(COMMON/'shared_items.json'):
        f=work/'generated'/name/x['key'];assert ready(f)
        m=read(f/'metadata.json');g=x['geometry'];e=m['export']
        assert m['step']==step and m['seed']==137 and m['checkpoint_sha256']==p['arms'][0]['original_sha256']
        assert m['planning_adapter']==p['planner_adapter_identity'] and m['generation']==p['generation']
        assert not m['listener_future_input'] and not m['target_description_access']
        assert e['frames']==g['frames'] and abs(e['fps']-g['fps'])<1e-3
        assert e['width']==e['height']==512
    result=read(work/'RESULTS.json')[name+'/responsenet']
    assert result['requested_pairs']==result['audio']['requested_pairs']==586
    assert not result['initialization_errors'] and not result['audio']['pairs_with_errors']
    save(work/'FINAL_AUDIT.json',dict(success=True,at=time.time(),outputs=586,step=step,
        physical_gpus=[7],seed=137,planner_step=2752,references_sha256=REF_SHA,
        metric_coverage=result['coverage'],status=result['status']))
    save(work/'HANDOFF_COMPLETE.json',dict(success=True,at=time.time(),outputs=586))


def controller():
    prepare()
    with (WORK/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        save(WORK/'CONTROLLER.json',dict(pid=os.getpid(),host=os.uname().nodename,at=time.time(),physical_gpus=[7]))
        # Never duplicate orphaned jobs after a controller-only failure.
        for work in (COMMON,run_work(6000),run_work(6500)):
            for file in (work/'processes').glob('*.json'):
                old=read(file);stat=Path('/proc')/str(old['pid'])/'stat'
                if stat.exists() and stat.read_text().rsplit(')',1)[1].split()[0]!='Z':
                    command=(stat.parent/'cmdline').read_bytes()
                    if str(work).encode() in command:raise RuntimeError('Recorded worker still live: '+str(file))
        while not (UPSTREAM/'COMPLETE.json').exists():
            if (WORK/'STOP').exists():return
            if (UPSTREAM/'status.json').exists() and read(UPSTREAM/'status.json').get('stage')=='failed':
                raise RuntimeError('Upstream ViCo6500 failed; preserve its GPU allocation for repair')
            save(WORK/'status.json',dict(stage='waiting_vico6500',at=time.time(),upstream=str(UPSTREAM)))
            time.sleep(60)
        # COMPLETE is written at the end; wait for its controller/recorded GPU children to exit too.
        while True:
            active=[]
            for f in [UPSTREAM/'LAUNCH.json',*list((UPSTREAM/'processes').glob('*.json'))]:
                if not f.exists():continue
                pid=read(f)['pid'];s=Path('/proc')/str(pid)/'stat'
                if s.exists() and s.read_text().rsplit(')',1)[1].split()[0]!='Z':active.append(pid)
            if not active:break
            time.sleep(60)
        if not (COMMON/'CACHE_COMPLETE.json').exists():
            while gpu()['free_gib']<64:time.sleep(60)
            save(WORK/'status.json',dict(stage='caching_missing386_SFT2752_and_Whisper',at=time.time()))
            wait_job(launch('cache',memory=48))
        for step in (6000,6500):
            if (run_work(step)/'HANDOFF_COMPLETE.json').exists():continue
            verify(run_work(step));render=None;decoders={};render_done=False;attempts={};next_status=0
            render_min_free=80 if (run_work(step)/'RESOURCE_WAIT.json').exists() else 64
            while True:
                if (WORK/'STOP').exists():
                    save(WORK/'status.json',dict(stage='stop_requested_waiting_current_workers',at=time.time()))
                    if render:wait_job(render)
                    for job in decoders.values():wait_job(job)
                    return
                if render and render['process'].poll() is not None:
                    if render['process'].returncode and 'out of memory' in Path(render['info']['log']).read_text(errors='replace').lower():
                        render_min_free=80
                        save(run_work(step)/'RESOURCE_WAIT.json',dict(at=time.time(),minimum_free_gib=80,
                            reason='bounded64GiB_admission_OOM_wait_for80GiB',completed_latents_preserved=True))
                        render=None
                    else:
                        wait_job(render);render=None;render_done=True
                        assert counts(step)[0]==586,'Renderer returned before completing all586 latents'
                for slot,job in list(decoders.items()):
                    if job['process'].poll() is not None:
                        rc=job['process'].returncode
                        del decoders[slot]
                        if rc:
                            attempts[slot]=attempts.get(slot,0)+1
                            if attempts[slot]>=3:wait_job(job)
                latents,decoded=counts(step);resources=gpu()
                if latents==586:render_done=True
                if not render and not render_done and not decoders and resources['free_gib']>=render_min_free:
                    # Same bounded64GiB admission used by the existing full-LoRA
                    # Seamless200 queue; raise to80 after an actual OOM.
                    render=launch('render',step,memory=min(96,resources['free_gib']-4));resources=gpu()
                # Native DiT remains resident; one VAE may consume otherwise free memory.
                limit=1 if render else 2
                if (render_done or latents>=10) and latents-decoded>len(decoders) and len(decoders)<limit and resources['free_gib']>=40:
                    slot=next(i for i in range(2) if i not in decoders)
                    decoders[slot]=launch('decode',step,slot,memory=28)
                if time.time()>=next_status:
                    save(WORK/'status.json',dict(stage='generation_and_export',at=time.time(),step=step,
                        latents=latents,videos=decoded,total=586,render_pid=render['info']['pid'] if render else None,
                        decoder_pids=[j['info']['pid'] for j in decoders.values()],gpu=resources))
                    next_status=time.time()+60
                if decoded==586 and not render and not decoders:break
                time.sleep(15)
            wait_job(launch('bind',step))
            pending=list(range(8));active={}
            while pending or active:
                for slot,job in list(active.items()):
                    if job['process'].poll() is not None:wait_job(job);del active[slot]
                if pending and len(active)<2 and gpu()['free_gib']>=28:
                    slot=pending.pop(0)
                    marker=run_work(step)/'progress/metrics'/f'{arm(step)}_{slot:02d}.json'
                    if not (marker.exists() and read(marker).get('status')=='complete'):
                        active[slot]=launch('metric',step,slot,memory=24)
                save(WORK/'status.json',dict(stage='scoring',step=step,at=time.time(),remaining_shards=len(pending)+len(active),total_shards=8))
                if pending or active:time.sleep(30)
            wait_job(launch('merge',step));final(step)
        combined={}
        for step in (6000,6500):combined.update(read(run_work(step)/'RESULTS.json'))
        save(WORK/'RESULTS.json',combined)
        save(WORK/'COMPLETE.json',dict(success=True,at=time.time(),outputs=1172,pairs_per_model=586,physical_gpus=[7]))
        save(WORK/'status.json',dict(stage='complete',at=time.time(),outputs=1172))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['prepare','controller','worker','final'])
    parser.add_argument('--stage',choices=['cache','render','decode'])
    parser.add_argument('--step',type=int,choices=[6000,6500],default=6000)
    parser.add_argument('--slot',type=int,default=0)
    a=parser.parse_args()
    try:
        if a.action=='prepare':prepare()
        elif a.action=='controller':controller()
        elif a.action=='final':final(a.step)
        else:worker(a.stage,a.step,a.slot)
    except Exception as exc:
        if a.action=='controller':save(WORK/'status.json',dict(stage='failed',at=time.time(),error=repr(exc),traceback=traceback.format_exc()))
        raise


if __name__=='__main__':main()
