"""Sequential all-attention/FFN LoRA checkpoint inference on the existing Seamless200."""
import copy
import fcntl
import json
import os
from pathlib import Path
import socket
import time
from unittest.mock import patch

from p2_testsets_eval.protocol import ROOT,read,save,sha,partition

BASE_TRAIN=ROOT/'experiments/p2_ltx_all_attention_ffn_lora31_20261108'
BATCH28_TRAIN=ROOT/'experiments/p2_ltx_all_lora_full28_batch28_20261109'
PARENT=ROOT/'experiments/p2_ltx6000_latest_sft_dual200_20261107'
CLI=ROOT/'scripts/p2_all_lora_followups_gpu7_20261109.py'
STEP=5500
TRAIN=BASE_TRAIN
WORK=ROOT/'experiments/p2_ltx_all_lora5500_seamless200_sft2752_gpu7_20261109'
ARM='ltx_all_lora_event_feature_whisper_5500_sft2752'


def configure(step):
    global STEP,TRAIN,WORK,ARM
    if step not in (5500,7000,7500):raise ValueError('Unapproved checkpoint step')
    STEP=step;TRAIN=BASE_TRAIN if step==5500 else BATCH28_TRAIN
    WORK=ROOT/f'experiments/p2_ltx_all_lora{step}_seamless200_sft2752_gpu7_20261109'
    ARM=f'ltx_all_lora_event_feature_whisper_{step}_sft2752'

from p2_ltx_all_lora.config import source_identity as _NATIVE_SOURCE,validate as _NATIVE_VALIDATE


def expected_source():
    result=_NATIVE_SOURCE()
    assert len(result)==282
    if STEP!=5500:
        folders=[('p2_ltx_split24_8_batch31_20261109',('runtime.py','worker.py','training_full.py','training_noevent.py')),
                 ('p2_full31_return_20261109',('runtime31.py','worker.py','training.py','canary.py')),
                 ('p2_full28_batch28_20261109',('runtime28.py','worker.py','training.py','canary.py','PLAN.json'))]
        for folder,names in folders:
            for name in names:
                f=ROOT/'experiments'/folder/name;result[str(f)]=sha(f)
    assert len(result)==(282 if STEP==5500 else 295)
    assert result==read(TRAIN/'run.json')['source']
    return result


def model_config(c):
    # Match runtime28.model: build with the original normalized model config,
    # while retaining the actual batch28 checkpoint identity verbatim.
    if STEP==5500:return _NATIVE_VALIDATE(c)
    v=copy.deepcopy(c);plan=v.pop('batch28_continuation')
    assert plan==read(ROOT/'experiments/p2_full28_batch28_20261109/PLAN.json')
    assert c['training']['global_batch']==28 and c['training']['steps']==plan['new_end_step']
    assert plan['prefix_pairs']==plan['start_step']*31
    import math
    assert plan['new_end_step']==plan['start_step']+math.ceil((plan['total_pairs']-plan['prefix_pairs'])/28)
    assert sha(plan['parent_config'])==plan['parent_config_sha256']
    v['training']['global_batch']=31;v['training']['steps']=plan['original_steps']
    parent=read(plan['parent_config']);parent['output']=v['output'];assert parent==v
    return _NATIVE_VALIDATE(v)


def stamp(path):
    p=Path(path);s=p.stat()
    return dict(path=str(p.resolve()),bytes=s.st_size,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns)


def export(work):
    from p2_whisper_av.bootstrap import activate_ltx
    activate_ltx()
    import torch
    torch.set_num_threads(2)
    from p2_ltx_all_lora.checkpoint import checked
    from p2_ltx_all_lora.config import validate,source_identity
    from p2_listen.cache import atomic_torch
    original=TRAIN/f'step_{STEP:06d}.pt';marker=read(original.with_suffix('.complete.json'))
    f=work/'export/RECEIPT.json'
    if f.exists():
        rec=read(f)
        assert rec['original_sha256']==marker['sha256']
        assert stamp(rec['checkpoint'])==rec['stamp']
        return rec
    v=checked(original);assert v['step']==STEP and len(v['rank_rng'])==(31 if STEP==5500 else 28)
    assert v['identity']==read(TRAIN/'run.json') and v['identity']['source']==expected_source()
    c=model_config(v['identity']['experiment'])
    assert c['loss_weights']==dict(av=1.,event=.05,video_feature=.025,audio_feature=.025)
    assert v['identity']['ltx_lora_report']['projections']==1632
    assert v['identity']['ltx_lora_report']['parameters']==84221952
    assert all(torch.isfinite(t).all() for s in v['optimizer']['state'].values() for t in s.values() if torch.is_tensor(t))
    assert {int(s['step']) for s in v['optimizer']['state'].values() if 'step' in s}=={STEP}
    target=work/'export'/f'step_{STEP:06d}.model.pt'
    atomic_torch(target,{k:v[k] for k in ['schema','step','identity','model']})
    exported=torch.load(target,map_location='cpu',weights_only=True,mmap=True)
    assert exported['identity']==v['identity'] and set(exported['model'])==set(v['model'])
    assert all(torch.equal(t,exported['model'][k]) for k,t in v['model'].items())
    rec=dict(success=True,at=time.time(),original=str(original),original_sha256=marker['sha256'],
        checkpoint=str(target),checkpoint_sha256=sha(target),stamp=stamp(target),
        identity=v['identity'],schema=v['schema'],step=STEP,all_model_tensors_bitwise_equal=True,
        full_model_adam_and_rng_preserved=True,rank_rng_count=len(v['rank_rng']))
    save(f,rec);return rec


def execution_code():
    paths=[Path(__file__).resolve(),CLI,ROOT/'src/p2_progressive_eval_20261108.py',
        ROOT/'scripts/p2_ltx6000_latest_sft_dual200_20261107.py',
        ROOT/'src/p2_testsets_eval/runtime.py',ROOT/'src/p2_testsets_eval/shared_items.py',
        ROOT/'src/p2_testsets_eval/whisper_runtime.py',ROOT/'src/p2_testsets_eval/cluster.py',
        ROOT/'src/p2_testsets_eval/metrics.py',ROOT/'scripts/p2_pipelined_testsets_eval.py',
        ROOT/'scripts/p2_pipelined_video_metrics.py',ROOT/'src/p2_testsets_eval/metric_pipeline.py']
    return {str(p):sha(p) for p in paths}


def prepare(work):
    work.mkdir(parents=True,exist_ok=True);rec=export(work)
    if (work/'protocol.json').exists():
        p=read(work/'protocol.json');assert p['execution_code']==execution_code()
        assert p['arms'][0]['original_sha256']==rec['original_sha256']
        return p
    from p2_combined_whisper_eval import link
    from p2_testsets_eval.runtime import ready
    parent=read(PARENT/'protocol.json');assert parent['seed']==137 and read(PARENT/'FINAL_AUDIT.json')['success']
    assert sha(PARENT/'shared_items.json')==parent['shared_items_sha256']
    items=[x for x in read(PARENT/'shared_items.json') if x['dataset']=='seamless']
    assert len(items)==len({x['key'] for x in items})==200
    spec=parent['datasets']['seamless'];assert sha(Path(spec['prepared'])/'references.jsonl')==spec['references_sha256']
    for folder in ['conditions/seamless','whisper/seamless','negative/00']:
        link(PARENT/folder,work/folder)
    # The original multi-GPU run indexed blank Gemma by rank. Confirm the fixed
    # rank00 negative used here is numerically identical across all original ranks.
    import torch
    def equal(a,b):
        if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
        if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
        if isinstance(a,(tuple,list)):return type(a)==type(b) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
        return a==b
    negative=torch.load(PARENT/'negative/00/payload.pt',map_location='cpu',weights_only=True)
    for rank in range(1,8):
        assert equal(negative,torch.load(PARENT/'negative'/f'{rank:02d}'/'payload.pt',map_location='cpu',weights_only=True))
    for x in items:
        f=work/'conditions'/x['key'];assert ready(f/'qwen') and ready(f/'text') and ready(work/'whisper'/x['key'])
        assert read(f/'qwen/plan.json')['adapter']==parent['planner_adapter_identity']
    cluster=copy.deepcopy(read(TRAIN/'cluster.json'))
    cluster['nodes']=[n for n in cluster['nodes'] if n['name']=='node11']
    for n in cluster['nodes']:
        n['gpus']=[7]
        if n['name']=='node11':n['ssh_key']='/private/ssh_keys/root_dedicated-developjob-js-infer-machine-ws3qe_ed25519'
    assert [n['name'] for n in cluster['nodes']]==['node11']
    cluster['training_script']=str(CLI);save(work/'cluster.json',cluster)
    from p2_ltx_all_lora.data import data_config
    save(work/'DATA_CONFIG.json',data_config(model_config(rec['identity']['experiment'])))
    save(work/'shared_items.json',items)
    for rank,rows in enumerate(partition(items,4)):save(work/'shards'/f'{rank:02d}.json',rows)
    p=copy.deepcopy(parent);p.update(schema='all-ltx-lora-followups-seamless200-gpu7-v1',created_at=time.time(),
        arms=[dict(name=ARM,kind='ltx_all_lora_full',step=STEP,checkpoint=rec['checkpoint'],
            checkpoint_sha256=rec['checkpoint_sha256'],original_checkpoint=rec['original'],
            original_sha256=rec['original_sha256'],planner=parent['planner'])],
        checkpoint_identity=rec['identity'],datasets={'seamless':spec},workers=4,total_samples=200,total_outputs=200,
        cluster=str(work/'cluster.json'),metrics_cluster=str(work/'cluster.json'),worker_cli=str(CLI),
        shared_items_manifest=str(work/'shared_items.json'),shared_items_sha256=sha(work/'shared_items.json'),
        metrics_workers_per_gpu=2,metrics_inference_workers_per_gpu=2,metrics_use_released_inference_nodes=False,
        execution_code=execution_code(),authorized_scope='node11 GPU7 only; four logical scoring shards; preserve all other tasks',physical_gpus=[7],training_global_batch=rec['identity']['experiment']['training']['global_batch'],
        conditioning='Unchanged SFT2752 plans/H/Gemma, speaker Whisper and listener first frame',
        execution_order='Shared4 render; released GPUs decode and score ready shards; native full200 merge')
    save(work/'protocol.json',p)
    save(work/'CPU_ACCEPTANCE.json',dict(success=True,at=time.time(),outputs=200,seed=137,planner_step=2752,
         model_export_bitwise_equal=True,all_training_sources_exact=True,training_source_count=len(rec['identity']['source']),fixed200=True,no_cache_increment=True,
         rank00_negative_equals_all_original8=True))
    return p


def pinned(work):
    import torch
    r=read(work/'export/RECEIPT.json');assert stamp(r['checkpoint'])==r['stamp']
    v=torch.load(r['checkpoint'],map_location='cpu',weights_only=True,mmap=True)
    assert stamp(r['checkpoint'])==r['stamp'] and v['identity']==r['identity']
    assert v['schema']==r['schema'] and v['step']==STEP
    return v


def load_model(work,path,device='cuda:0'):
    from p2_ltx_all_lora import inference,checkpoint,config
    p=read(work/'protocol.json');assert str(path)==p['arms'][0]['checkpoint']
    with patch.object(checkpoint,'checked',lambda _:pinned(work)), patch.object(config,'source_identity',expected_source), patch.object(config,'validate',model_config):
        model,strategy,sampler=inference.load_model(path,device)
    return model,strategy,sampler


def worker(work,stage,rank,slot=0):
    assert rank==3 and 0<=slot<4 and os.environ.get('CUDA_VISIBLE_DEVICES')==str(rank+4)
    p=read(work/'protocol.json');assert p['execution_code']==execution_code()
    node=read(work/'cluster.json')['nodes'][0]
    assert node['host'] in os.popen('hostname -I').read().split()
    from p2_whisper_av.bootstrap import activate_ltx
    activate_ltx()
    import torch
    torch.set_num_threads(4);torch.cuda.set_device(0)
    from p2_listen.config import extractor_config
    from p2_testsets_eval import runtime as rt
    from p2_testsets_eval.shared_items import all_items
    d=read(work/'DATA_CONFIG.json');cfg=extractor_config(d)
    f=work/'locks'/f'{stage}_{rank:02d}_{slot}.lock';f.parent.mkdir(exist_ok=True)
    with f.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        save(work/'devices'/f'{stage}_{rank:02d}_{slot}.json',dict(pid=os.getpid(),at=time.time(),
             host=socket.gethostname(),rank=rank,physical_gpu=rank+4,stage=stage))
        old_progress=rt.progress
        def progress(w,s,r,**kw):return old_progress(w,s,rank+4*slot,**kw)
        with patch.object(rt,'data_config',lambda:(d,cfg)),patch.object(rt,'progress',progress):
            if stage=='render':
                import p2_ltx6000_latest_sft_dual200_20261107 as native
                import p2_ltx_events.inference as previous
                def loading(path,device='cuda:0'):
                    result=load_model(work,path,device)
                    save(work/'model_acceptance'/f'{rank:02d}.json',dict(success=True,at=time.time(),rank=rank,
                         physical_gpu=rank+4,strict_all_lora_load=True,targets=len(result[0].text_targets)))
                    return result
                with patch.object(native,'ARM',ARM),patch.object(previous,'load_model',loading):
                    render(work,all_items(work,[],p),0,p)
            else:rt.decode(work,all_items(work,[],p),rank,p,p['arms'])
        progress(work,stage,0,status='complete',slot=slot)


def final(work):
    from p2_testsets_eval.runtime import ready
    p=read(work/'protocol.json')
    for x in read(work/'shared_items.json'):
        f=work/'generated'/ARM/x['key'];assert ready(f);m=read(f/'metadata.json')
        assert m['step']==STEP and m['seed']==137 and m['checkpoint_sha256']==p['arms'][0]['original_sha256']
        assert m['planning_adapter']==p['planner_adapter_identity'] and m['generation']==p['generation']
        assert not m['target_description_access'] and not m['listener_future_input']
    r=read(work/'RESULTS.json')[ARM+'/seamless']
    assert r['requested_pairs']==r['audio']['requested_pairs']==200 and r['audio']['pairs_with_errors']==0
    assert all(r['coverage'][k]['pairs']==200 for k in ['psnr','ssim','lpips','fid','fvd'])
    save(work/'FINAL_AUDIT.json',dict(success=True,at=time.time(),step=STEP,outputs=200,world=1,node='node11',physical_gpus=[7],logical_metric_shards=4,
         planner_step=2752,seed=137,all_attention_ffn_lora=True))
    save(work/'COMPARISON.json',dict(current=r,previous_all_lora6000=read(ROOT/'experiments/p2_ltx_all_lora6000_seamless200_sft2752_eval32_20261109/RESULTS.json')[
         'ltx_all_lora_event_feature_whisper_6000_sft2752/seamless']))
    save(work/'HANDOFF_COMPLETE.json',dict(success=True,at=time.time(),outputs=200))


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
            save(out/'metadata.json',dict(key=item['key'],arm=ARM,checkpoint=arm['original_checkpoint'],checkpoint_sha256=arm['original_sha256'],inference_checkpoint=arm['checkpoint'],step=STEP,seed=p['seed'],geometry=g,planning_sha256=sha(folder/'qwen/plan.json'),planning_adapter=p['planner_adapter_identity'],target_description_access=False,listener_future_input=False,generation=p['generation'],whisper_source=meta,sampling=p['sampling'],seconds=time.monotonic()-begin))
            rt.complete(out,[out/'payload.pt',out/'metadata.json']);rt.progress(work,'render',rank,done=i+1,total=len(items),arm=ARM,last_seconds=time.monotonic()-begin)
            del payload,text,inputs,u,speaker,av,prepared,sampled,video,audio
    del model,strategy,sampler,negative;unload()

