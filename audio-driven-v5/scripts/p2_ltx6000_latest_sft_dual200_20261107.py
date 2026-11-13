"""Queue LTX-hidden-Event6000 + latest SFT2752 after the running6309 dual200."""
import argparse,copy,fcntl,json,os,subprocess,sys,time,traceback
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];SELF=Path(__file__).resolve()
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT.parent/'OmniResponse')]
from p2_testsets_eval.protocol import read,save,sha,partition,PYTHON
WORK=ROOT/'experiments/p2_ltx6000_latest_sft_dual200_20261107'
UPSTREAM=ROOT/'experiments/p2_event_whisper6309_dual200_node11_20261107'
TRAIN=ROOT/'experiments/p2_ltx_clip_train31_cadence_20261107'
SFT=ROOT/'experiments/p2_planning_sft_upright_3b_32gpu_resume_20261106/training'
ADAPTER=SFT/'final_adapter'
ARM='ltx_hidden_event_feature_whisper_6000_sft2752'


def install():
    from p2_ltx6000_sft_eval_20261107 import install as pinned
    pinned()


def code_identity():
    paths=[SELF,ROOT/'src/p2_ltx6000_sft_eval_20261107.py',ROOT/'src/p2_testsets_eval/runtime.py',ROOT/'src/p2_testsets_eval/whisper_runtime.py',ROOT/'src/p2_testsets_eval/shared_items.py',ROOT/'src/p2_testsets_eval/cluster.py',ROOT/'src/p2_testsets_eval/metrics.py',ROOT/'src/p2_j_sft_hidden.py',ROOT/'src/p2_plan_cache/encoding.py',ROOT/'src/p2_planning_sft/data.py',ROOT/'scripts/p2_responsenet_random200.py']
    return {str(f):sha(f) for f in paths}


def prepare(work):
    work.mkdir(exist_ok=True)
    if (work/'protocol.json').exists():
        p=read(work/'protocol.json')
        if p['execution_code']!=code_identity():raise ValueError('Queued evaluation source changed')
        return p
    save(work/'status.json',dict(stage='cpu_preparing',at=time.time()))
    install()
    import torch
    torch.set_num_threads(2)
    from p2_ltx_events.config import validate,source_identity
    from p2_ltx_events.data import data_config
    from p2_ltx_events.checkpoint import checked
    from p2_listen.cache import atomic_torch
    from p2_j_sft_hidden import adapter_identity
    from p2_combined_whisper_eval import link
    parent=read(UPSTREAM/'protocol.json');items=read(UPSTREAM/'shared_items.json')
    assert len(items)==400 and sha(UPSTREAM/'shared_items.json')==parent['shared_items_sha256']
    assert read(SFT/'COMPLETE.json')['success'] and read(SFT/'COMPLETE.json')['global_step']==2752
    assert read(SFT.parent/'COMPLETE.json')['success']
    adapter=adapter_identity(ADAPTER)
    original=TRAIN/'step_006000.pt';marker=read(original.with_suffix('.complete.json'))
    target=work/'checkpoints/step_006000.pt';receipt=target.with_suffix('.export.json')
    if receipt.exists():
        audit=read(receipt)
        if audit['original_sha256']!=marker['sha256']:raise ValueError('Original6000 changed')
        saved=checked(target)
    else:
        saved=checked(original)
        if saved['identity']!=read(TRAIN/'run.json') or saved['identity']['source']!=source_identity():raise ValueError('Wrong LTX6000 source')
        atomic_torch(target,{k:saved[k] for k in ['schema','step','identity','model']})
        save(target.with_suffix('.complete.json'),dict(step=6000,sha256=sha(target)))
        exported=checked(target)
        assert exported['identity']==saved['identity'] and set(exported['model'])==set(saved['model'])
        assert all(torch.equal(v,exported['model'][k]) for k,v in saved['model'].items())
        audit=dict(original_checkpoint=str(original),original_sha256=marker['sha256'],inference_checkpoint=str(target),inference_sha256=read(target.with_suffix('.complete.json'))['sha256'],step=6000,model_bitwise_equal=True,optimizer_omitted=True)
        save(receipt,audit)
    identity=saved['identity'];c=validate(identity['experiment']);assert saved['step']==6000
    assert c['hidden_event_supervision']==dict(readout='last_block_before_output',clip_weight=1.,temporal_weight=0.)
    assert c['loss']=='av_event_feature' and c['qwen_hidden_source']=='base_pre_sft'
    d=data_config(c)
    from p2_listen.config import extractor_config
    cfg=extractor_config(d)
    assert Path(read(ADAPTER/'adapter_config.json')['base_model_name_or_path']).resolve()==Path(cfg.omni_path).resolve()
    # Reuse only source media and speaker Whisper. Both Qwen/H and Gemma are new.
    for dataset,spec in parent['datasets'].items():
        assert sha(Path(spec['prepared'])/'references.jsonl')==spec['references_sha256']
        link(UPSTREAM/'whisper'/dataset,work/'whisper'/dataset)
    sources={}
    for x in items:
        src=UPSTREAM/'conditions'/x['key'];dst=work/'conditions'/x['key'];dst.mkdir(parents=True,exist_ok=True)
        for name in ['first_frame.png','input_audit.json']:link(src/name,dst/name)
        sources[x['key']]={name:sha(src/name) for name in ['first_frame.png','input_audit.json']}
    for rank in range(8):link(UPSTREAM/'negative'/f'{rank:02d}',work/'negative'/f'{rank:02d}')
    for rank,shard in enumerate(partition(items,8)):save(work/'shards'/f'{rank:02d}.json',shard)
    save(work/'shared_items.json',items);save(work/'SOURCE_MEDIA_IDENTITY.json',sources)
    from p2_planning_sft.data import SYSTEM,USER
    session=work/'adapter_session';save(session/'protocol.json',dict(schema='latest-sft-inference-session-v1',adapter=adapter,data_config=d,training_complete=str(SFT/'COMPLETE.json'),training_complete_sha256=sha(SFT/'COMPLETE.json'),global_step=2752))
    cluster=read(UPSTREAM/'cluster.json');cluster['nodes'][0]['gpus']=list(range(8));cluster['training_script']=str(SELF);save(work/'cluster.json',cluster)
    p=copy.deepcopy(parent);p.update(schema='ltx-hidden-event6000-latest-sft2752-dual200-v1',created_at=time.time(),
        arms=[dict(name=ARM,kind='ltx_hidden_event_feature_whisper',step=6000,checkpoint=str(target),checkpoint_sha256=audit['inference_sha256'],original_checkpoint=str(original),original_sha256=marker['sha256'],planner='SFT_Qwen2.5-Omni-3B_upright2752')],
        checkpoint_identity=identity,workers=8,total_samples=400,total_outputs=400,cluster=str(work/'cluster.json'),metrics_cluster=str(work/'cluster.json'),worker_cli=str(SELF),
        shared_items_manifest=str(work/'shared_items.json'),shared_items_sha256=sha(work/'shared_items.json'),
        planner='SFT_Qwen2.5-Omni-3B_upright2752',planner_adapter=str(ADAPTER),planner_adapter_identity=adapter,sft_hidden_cache=str(session),
        source_media_identity=str(work/'SOURCE_MEDIA_IDENTITY.json'),source_media_identity_sha256=sha(work/'SOURCE_MEDIA_IDENTITY.json'),
        conditioning='Latest SFT2752 plans description and extracts its H; fresh Gemma; same speaker Whisper and first frame',
        training_qwen_hidden_source='base_pre_sft',inference_qwen_hidden_source='SFT2752',
        authorized_scope='node11 GPUs0..7 after upstream6309 evaluation; no other-task termination',
        comparison='Different LTX hidden Event readout/current500h plus latest SFT; not a single-variable causal ablation',
        upstream_completion=str(UPSTREAM/'HANDOFF_COMPLETE.json'),upstream_work=str(UPSTREAM),execution_order='wait upstream complete -> latestSFT planning/H/Gemma -> shared8 GPU render -> decode -> native metrics -> full200 merge',execution_code=code_identity())
    save(work/'protocol.json',p)
    from p2_ltx_events import config as C
    save(work/'FROZEN_EXECUTION.json',dict(executed=C._eval6000_executed,expected_source=source_identity(),workspace_preserved=True))
    save(work/'CPU_PRECHECK.json',dict(success=True,at=time.time(),ltx_step=6000,planner_step=2752,adapter=adapter,export=audit,outputs=400,science=dict(seed=p['seed'],generation=p['generation'],datasets=p['datasets']),new_conditions_required=True,gt_future_or_caption_input=False,latest_system=SYSTEM,latest_user=USER,gpu_check='pending after upstream'))
    return p


def cached_conditioning(item,folder):
    import numpy as np
    from PIL import Image
    from p2_testsets_eval.whisper_runtime import speaker_wave
    p=read(folder.parents[2]/'protocol.json')
    if sha(p['source_media_identity'])!=p['source_media_identity_sha256']:raise ValueError('Source identity manifest changed')
    identities=read(p['source_media_identity'])
    for name,h in identities[item['key']].items():
        if sha(folder/name)!=h:raise ValueError('Source media changed')
    if item['dataset']=='responsenet':
        from p2_responsenet_random200 import rn_speaker_wave
        wave,audit=rn_speaker_wave(item,folder)
    else:wave,audit=speaker_wave(item,folder)
    source=read(folder.parents[2]/'whisper'/item['key']/'metadata.json')['source']
    if audit['waveform_sha256']!=source['waveform_sha256'] or audit['source_sha256']!=source['source_sha256']:
        raise ValueError('Planning audio differs from frozen speaker Whisper waveform')
    with Image.open(folder/'first_frame.png') as image:image=image.convert('RGB')
    assert image.size==(512,512) and np.isfinite(wave).all()
    return dict(sample_id=item['id'],duration=item['geometry']['duration_s'],image=image,audio=wave)


@contextmanager
def runtime(work):
    install()
    from p2_ltx_events.data import data_config
    from p2_listen.config import extractor_config
    from p2_ltx6000_sft_eval_20261107 import SFTInferenceSession
    p=read(work/'protocol.json');d=data_config(p['checkpoint_identity']['experiment']);cfg=extractor_config(d)
    with patch('p2_testsets_eval.runtime.data_config',lambda:(d,cfg)),patch('p2_testsets_eval.runtime.conditioning',cached_conditioning),patch('p2_j_sft_hidden.Reader',SFTInferenceSession):yield


def render(work,items,rank,p):
    import torch
    from p2_testsets_eval import runtime as rt
    from p2_testsets_eval.whisper_runtime import matched_sample
    from p2_testsets_eval.shared_items import claim_items,all_items
    from p2_ltx_events.inference import load_model
    from p2_listen.cache import atomic_torch,cpu,tree_to
    from p2_listen.features import unload,make_observation
    from p2_whisper_av.features import SpeakerFeatures
    from omni_av_p2.connector import ConditionInputs
    from omni_av_p2.inference import reference_latents
    from omni_av_p2.ltx import unpack_latents
    from omni_av.ltx_runtime import with_text
    items=all_items(work,items,p);arm=p['arms'][0];device='cuda:0';_,cfg=rt.data_config()
    if all(rt.ready(work/'latents'/ARM/x['key']) or rt.ready(work/'generated'/ARM/x['key']) for x in items):return
    model,strategy,sampler=load_model(arm['checkpoint'],device)
    model.eval().requires_grad_(False);model.renderer.eval().requires_grad_(False);model.renderer.set_gradient_checkpointing(False)
    negative=tree_to(torch.load(work/'negative'/f'{rank:02d}/payload.pt',map_location='cpu',weights_only=True),device)
    with rt.cached_loaders(['load_video_vae_encoder']),torch.inference_mode():
        for i,item in enumerate(claim_items(work,items,p,arm,'render')):
            begin=time.monotonic();folder=work/'conditions'/item['key'];out=work/'latents'/ARM/item['key'];g=item['geometry']
            assert rt.ready(folder/'qwen') and rt.ready(folder/'text')
            meta=read(work/'whisper'/item['key']/'metadata.json');plan=read(folder/'qwen/plan.json')
            assert meta['encoder']==p['whisper_encoder'] and meta['feature_code']==p['feature_code']
            assert meta['source']['input_audit_sha256']==sha(folder/'input_audit.json')
            assert plan['adapter']==p['planner_adapter_identity'] and plan['target_description_access'] is False
            payload=torch.load(folder/'qwen/hidden.pt',map_location='cpu',weights_only=True);text=torch.load(folder/'text/payload.pt',map_location='cpu',weights_only=True)
            assert text['prompt']==payload['prompt']==plan['prompt']
            inputs=ConditionInputs(**tree_to(payload['condition'],device))
            with torch.autocast('cuda',dtype=torch.bfloat16):u=model.connector(inputs)
            speaker=SpeakerFeatures(**torch.load(work/'whisper'/item['key']/'payload.pt',map_location='cpu',weights_only=True)).validate().to(device)
            native=dict(fps=g['fps'],height=g['height'],width=g['width'],frames=g['native_frames'],duration_s=g['native_duration_s'])
            av=reference_latents(cfg,make_observation(rt.observation(item,folder)),native,device)
            torch.manual_seed(p['seed']);prepared=strategy.prepare_training_inputs(with_text(av,tree_to(text,device)),sampler)
            torch.manual_seed(p['seed']);sampled=matched_sample(model,prepared,u,speaker,negative,**p['generation']);video,audio=unpack_latents(sampled,av)
            assert torch.isfinite(video).all() and torch.isfinite(audio).all()
            atomic_torch(out/'payload.pt',cpu(dict(video=video,audio=audio)))
            save(out/'metadata.json',dict(key=item['key'],arm=ARM,checkpoint=arm['original_checkpoint'],checkpoint_sha256=arm['original_sha256'],inference_checkpoint=arm['checkpoint'],step=6000,seed=p['seed'],geometry=g,planning_sha256=sha(folder/'qwen/plan.json'),planning_adapter=p['planner_adapter_identity'],target_description_access=False,listener_future_input=False,generation=p['generation'],whisper_source=meta,sampling=p['sampling'],seconds=time.monotonic()-begin))
            rt.complete(out,[out/'payload.pt',out/'metadata.json']);rt.progress(work,'render',rank,done=i+1,total=len(items),arm=ARM,last_seconds=time.monotonic()-begin)
            del payload,text,inputs,u,speaker,av,prepared,sampled,video,audio
    del model,strategy,sampler,negative;unload()


def worker(a):
    import torch
    physical=int(os.environ.get('CUDA_VISIBLE_DEVICES','-1'))
    if physical not in range(8) or (a.stage!='cache' and physical!=a.rank):raise ValueError('Wrong physical GPU')
    work=a.work.resolve();p=read(work/'protocol.json')
    assert all(sha(f)==h for f,h in p['execution_code'].items())
    torch.set_num_threads(4);torch.cuda.set_device(0);items=read(work/'shards'/f'{a.rank:02d}.json')
    if a.limit:items=items[:a.limit]
    (work/'locks').mkdir(exist_ok=True)
    with (work/'locks'/f'{a.stage}_{a.rank:02d}.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with runtime(work):
            from p2_testsets_eval import runtime as rt
            save(work/'devices'/f'{a.stage}_{a.rank:02d}.json',dict(physical_gpu=physical,rank=a.rank,host=os.uname().nodename,pid=os.getpid(),stage=a.stage,at=time.time()))
            rt.progress(work,a.stage,a.rank,status='starting',samples=len(items))
            if a.stage=='cache':rt.cache(work,items,a.rank,p)
            elif a.stage=='render':render(work,items,a.rank,p)
            else:
                from p2_testsets_eval.shared_items import all_items
                rt.decode(work,all_items(work,items,p),a.rank,p,p['arms'])
            rt.progress(work,a.stage,a.rank,status='complete')


def final(work,p):
    # Native metric gate/audit are generic except for their displayed step/arm;
    # use an explicit local audit for this independent6000/SFT group.
    from p2_testsets_eval.runtime import ready
    results=read(work/'RESULTS.json');items=read(work/'shared_items.json')
    for x in items:
        folder=work/'generated'/ARM/x['key'];assert ready(folder)
        m=read(folder/'metadata.json');assert m['step']==6000 and m['planning_adapter']==p['planner_adapter_identity']
        assert m['checkpoint_sha256']==p['arms'][0]['original_sha256'] and not m['target_description_access'] and not m['listener_future_input']
    for d in p['datasets']:
        s=results[ARM+'/'+d];assert s['requested_pairs']==200 and s['audio']['requested_pairs']==200 and s['audio']['pairs_with_errors']==0
        assert all(s['coverage'][k]['pairs']==200 for k in ['psnr','ssim','lpips','fid','fvd'])
    save(work/'FINAL_AUDIT.json',dict(success=True,at=time.time(),outputs=400,ltx_step=6000,planner_step=2752,source_only=True))
    save(work/'COMPARISON.json',dict(current=results,upstream=read(UPSTREAM/'RESULTS.json'),notes=[p['comparison']]))
    (work/'RESULTS.md').write_text('# LTX主干整段Event＋Feature＋Whisper6000＋最新SFT2752\n\n两套固定200条。全部分数、覆盖及逐条结果见RESULTS.json和metrics/；与6309组存在训练数据、Event读出位置、Feature和规划器差异，不能视为单变量消融。\n')


def cache_jobs(work):
    """Keep the eight frozen shards; assign each to a sufficiently free GPU.

    A busy GPU7 does not strand the other seven. Reassign its shard after an
    empty card finishes, preserving batch membership and the native cache math.
    """
    import concurrent.futures,queue
    from listen_cluster import transport
    from p2_testsets_eval.cluster import state
    p=read(work/'protocol.json');cfg=read(work/'cluster.json');node=cfg['nodes'][0]
    text=subprocess.check_output(transport(cfg,node,['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits']),text=True,timeout=25)
    eligible=[int(t[0]) for line in text.splitlines() if (t:=line.split(',')) and float(t[1])<2048 and float(t[2])<10]
    assert eligible and set(eligible)<=set(range(8))
    slots=queue.Queue()
    for g in eligible:slots.put(g)
    def one(rank):
        g=slots.get()
        try:
            cmd=['env','CUDA_VISIBLE_DEVICES='+str(g),'OMP_NUM_THREADS=4','OPENBLAS_NUM_THREADS=2','TOKENIZERS_PARALLELISM=false','PYTORCH_ALLOC_CONF=expandable_segments:True',str(PYTHON),'-u',str(SELF),'worker','--work',str(work),'--stage','cache','--rank',str(rank)]
            log=work/'logs/cache'/f'rank{rank:02d}.log';log.parent.mkdir(parents=True,exist_ok=True)
            for attempt in range(1,4):
                if (work/'STOP').exists():raise RuntimeError('User STOP')
                with log.open('ab') as stream:code=subprocess.call(transport(cfg,node,cmd),stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
                if code==0:return dict(rank=rank,physical_gpu=g,success=True,attempt=attempt)
                time.sleep(3)
            raise RuntimeError('Cache failed: '+str(log))
        finally:slots.put(g)
    state(work,'cache',eligible_gpus=eligible,shards=8)
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(eligible)) as pool:results=list(pool.map(one,range(8)))
    from p2_testsets_eval.runtime import ready
    for item in read(work/'shared_items.json'):
        folder=work/'conditions'/item['key']
        assert ready(folder/'qwen') and ready(folder/'text')
        assert read(folder/'qwen/plan.json')['adapter']==p['planner_adapter_identity']
    save(work/'stages/cache.json',dict(complete=True,results=results,at=time.time(),all400_latest_sft=True))


def queue(work):
    work.mkdir(exist_ok=True)
    with (work/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        from p2_speaker_whisper_after_annotation import process_record
        from p2_testsets_eval.cluster import state,run_cluster,bind
        from listen_cluster import transport
        save(work/'CONTROLLER.json',process_record());p=prepare(work);cfg=read(work/'cluster.json')
        while not (UPSTREAM/'HANDOFF_COMPLETE.json').exists():
            if (work/'STOP').exists():state(work,'stopped');return
            state(work,'waiting_upstream6309_complete',upstream=str(UPSTREAM),upstream_status=read(UPSTREAM/'status.json'));time.sleep(15)
        assert read(UPSTREAM/'FINAL_AUDIT.json')['success']
        for part in ['cache','render','decode']:
            if (work/'stages'/(part+'.json')).exists():continue
            while True:
                if (work/'STOP').exists():state(work,'stopped');return
                try:
                    text=subprocess.check_output(transport(cfg,cfg['nodes'][0],['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits']),text=True,timeout=25)
                    rows={int(t[0]):[float(v) for v in t[1:]] for line in text.splitlines() if (t:=line.split(','))}
                    available=all(rows[g][0]<2048 and rows[g][1]<10 for g in range(7)) and (part=='cache' or rows[7][0]<76000)
                except Exception as exc:text=repr(exc);available=False
                save(work/'resources.json',dict(at=time.time(),csv=text,available=available))
                if available:break
                state(work,'waiting_authorized_node11',pending=part);time.sleep(15)
            if part=='cache':cache_jobs(work)
            else:run_cluster(work,part)
        from p2_testsets_eval.runtime import ready
        assert all(ready(work/'generated'/ARM/x['key']) for x in read(work/'shared_items.json'))
        save(work/'GENERATION_COMPLETE.json',dict(total_outputs=400,at=time.time()))
        bind(work)
        # Native smoke selects two real generated examples for each dataset.
        from p2_testsets_eval.metrics import prepare_smoke,queue as score
        save(work/'SMOKE_COMPLETE.json',dict(status='generation_passed',outputs=400,ltx_step=6000,planner_step=2752,at=time.time()))
        prepare_smoke(work);score(work)
        if (work/'STOP').exists():return
        final(work,p);save(work/'HANDOFF_COMPLETE.json',dict(outputs=400,ltx_step=6000,planner_step=2752,at=time.time()));state(work,'complete')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','queue','worker']);parser.add_argument('--work',type=Path,default=WORK);parser.add_argument('--stage',choices=['cache','render','decode']);parser.add_argument('--rank',type=int,default=0);parser.add_argument('--limit',type=int);parser.add_argument('--arms',nargs='+');a=parser.parse_args()
    try:
        if a.action=='prepare':prepare(a.work)
        elif a.action=='worker':worker(a)
        else:queue(a.work)
    except Exception as exc:
        save(a.work/'FAILURE.json',dict(error=repr(exc),traceback=traceback.format_exc(),at=time.time()));raise
