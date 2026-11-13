"""Native Feature/Whisper evaluation adapters; immutable inputs and fast pinned exports."""
from contextlib import contextmanager
import copy
import importlib
import os
from pathlib import Path
import time
from unittest.mock import patch

from p2_testsets_eval.protocol import ROOT, read, save, sha, partition

FEATURE = ROOT/'experiments/p2_current500h_feature6000_sft2752_dual200_20261108'
WHISPER = ROOT/'experiments/p2_current500h_whisper6000_sft2752_dual200_20261108'
CONDITIONS = ROOT/'experiments/p2_ltx6000_latest_sft_dual200_20261107'
INFO = {
    'feature': dict(work=FEATURE, train=ROOT/'experiments/p2_current500h_av_feature31_20261107',
                    package='p2_current_av_feature', arm='current500h_av_feature_6000_sft2752'),
    'whisper': dict(work=WHISPER, train=ROOT/'experiments/p2_current500h_av_whisper31_20261107',
                    package='p2_current_av_whisper', arm='current500h_av_whisper_6000_sft2752'),
}


def stamp(path):
    p=Path(path);s=p.stat()
    return dict(path=str(p.resolve()),bytes=s.st_size,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns)


def native(name):
    from p2_whisper_av.bootstrap import activate_ltx
    activate_ltx()
    package=INFO[name]['package']
    return (importlib.import_module(package+'.checkpoint'),importlib.import_module(package+'.config'),
            importlib.import_module(package+'.model'),importlib.import_module(package+'.data'))


def export_model(root,name):
    import torch
    from p2_listen.cache import atomic_torch
    ck,config,_,data=native(name)
    original=INFO[name]['train']/'step_006000.pt'
    marker=read(original.with_suffix('.complete.json'))
    receipt=root/'exports'/name/'RECEIPT.json'
    if receipt.exists():
        value=read(receipt)
        assert value['original_sha256']==marker['sha256']
        assert stamp(value['model_checkpoint'])==value['model_stamp']
        assert sha(value['model_checkpoint'])==value['model_sha256']
        return value
    saved=ck.checked(original)
    assert saved['step']==6000 and saved['identity']==read(INFO[name]['train']/'run.json')
    assert saved['identity']['source']==config.source_identity() and len(saved['rank_rng'])==31
    config.validate(saved['identity']['experiment'])
    path=root/'exports'/name/'step_006000.model.pt'
    atomic_torch(path,{k:saved[k] for k in ['schema','identity','step','model']})
    exported=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
    assert exported['identity']==saved['identity'] and exported['schema']==saved['schema']
    assert set(exported['model'])==set(saved['model'])
    assert all(torch.equal(v,exported['model'][k]) for k,v in saved['model'].items())
    value=dict(success=True,at=time.time(),model=name,step=6000,
        original_checkpoint=str(original),original_sha256=marker['sha256'],
        model_checkpoint=str(path),model_sha256=sha(path),model_stamp=stamp(path),
        all_model_tensors_bitwise_equal=True,optimizer_and_rng_preserved_in_original=True,
        identity=saved['identity'],schema=saved['schema'],original_bytes=original.stat().st_size,
        gpu_read_policy='CPU full SHA and tensor equality; worker mmap with before/after size+mtime+ctime pin and strict native load')
    save(receipt,value)
    return value


def prepare_whisper(root):
    info=export_model(root,'whisper')
    from p2_combined_whisper_eval import link
    from p2_testsets_eval.runtime import ready
    _,config,_,data=native('whisper')
    parent=read(FEATURE/'protocol.json');p=copy.deepcopy(parent);work=WHISPER
    work.mkdir(parents=True,exist_ok=True)
    if (work/'protocol.json').exists():
        old=read(work/'protocol.json')
        assert old['arms'][0]['original_sha256']==info['original_sha256']
        return old
    original=read(CONDITIONS/'protocol.json')
    c=info['identity']['experiment'];encoder=read(Path(c['whisper_cache'])/'protocol.json')['encoder']
    assert encoder==original['whisper_encoder']
    assert c['loss']=='av' and c['loss_weights']==dict(av=1.,event=0.,video_feature=0.,audio_feature=0.)
    items=read(FEATURE/'shared_items.json')
    assert sha(FEATURE/'shared_items.json')==parent['shared_items_sha256'] and len(items)==400
    for ds in parent['datasets']:
        link(CONDITIONS/'conditions'/ds,work/'conditions'/ds)
        link(CONDITIONS/'whisper'/ds,work/'whisper'/ds)
    for x in items:
        f=work/'conditions'/x['key'];w=work/'whisper'/x['key']
        assert ready(f/'qwen') and ready(f/'text') and ready(w)
        plan=read(f/'qwen/plan.json');meta=read(w/'metadata.json')
        assert plan['adapter']==parent['planner_adapter_identity'] and not plan['target_description_access']
        assert meta['encoder']==encoder and meta['feature_code']==original['feature_code']
        assert meta['source']['input_audit_sha256']==sha(f/'input_audit.json')
    for rank in range(8):link(CONDITIONS/'negative'/f'{rank:02d}',work/'negative'/f'{rank:02d}')
    save(work/'shared_items.json',items)
    p.update(schema='current500h-av-whisper6000-sft2752-eval32-v1',created_at=time.time(),
        arms=[dict(name=INFO['whisper']['arm'],kind='current500h_av_whisper',step=6000,
              checkpoint=info['model_checkpoint'],checkpoint_sha256=info['model_sha256'],
              original_checkpoint=info['original_checkpoint'],original_sha256=info['original_sha256'],
              planner=parent['planner'])],checkpoint_identity=info['identity'],
        shared_items_manifest=str(work/'shared_items.json'),shared_items_sha256=sha(work/'shared_items.json'),
        whisper_encoder=encoder,feature_code=original['feature_code'],workers=32,
        cluster=str(root/'cluster.json'),metrics_cluster=str(root/'cluster.json'),
        metrics_workers_per_gpu=2,metrics_inference_workers_per_gpu=2,
        metrics_use_released_inference_nodes=False,
        authorized_scope='User authorized four nodes32 GPUs after complete6000 save; no unrelated jobs stopped',
        comparison='Same current500h pureAV6000 training schedule and same SFT2752 dual200 conditions; only extra speaker Whisper branches',
        sampling=original['sampling'])
    save(work/'protocol.json',p);save(work/'DATA_CONFIG.json',data.data_config(c))
    for rank,rows in enumerate(partition(items,32)):save(work/'shards'/f'{rank:02d}.json',rows)
    save(work/'CPU_ACCEPTANCE.json',dict(success=True,at=time.time(),step=6000,planner_step=2752,
         source_exact=True,export_bitwise_equal=True,shared_SFT_conditions=400,
         speaker_whisper_cache_exact=True,feature_or_event_labels_read=False))
    save(work/'status.json',dict(stage='prepared32',at=time.time()))
    return p


def pinned(root,name):
    import torch
    value=read(root/'exports'/name/'RECEIPT.json');path=value['model_checkpoint']
    assert stamp(path)==value['model_stamp'],'Inference export changed since full CPU verification'
    saved=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
    assert stamp(path)==value['model_stamp']
    assert saved['identity']==value['identity'] and saved['schema']==value['schema'] and saved['step']==6000
    return saved,value


def load_model(root,name,cfg):
    import torch
    from p2_listen.features import weights_identity
    from omni_av_p2.ltx import load_renderer
    saved,receipt=pinned(root,name)
    _,config,models,_=native(name)
    identity=saved['identity'];c=config.validate(identity['experiment'])
    assert identity['source']==config.source_identity()
    d=read(INFO[name]['work']/'DATA_CONFIG.json')
    assert identity['assets']=={k:weights_identity(d,k) for k in ['omni','ltx','gemma']}
    renderer,strategy,sampler=load_renderer(cfg,'cuda:0');renderer.set_gradient_checkpointing(False)
    torch.manual_seed(c['training']['seed'])
    model=models.Model(c,cfg,renderer).to('cuda:0')
    assert model.text_targets==identity['text_targets'] and not model.qwen_targets
    model.load_state_dict(saved['model'],strict=True)
    model.eval().requires_grad_(False);renderer.eval().requires_grad_(False)
    return model,strategy,sampler


@contextmanager
def runtime(name):
    native(name)
    from p2_listen.config import extractor_config
    d=read(INFO[name]['work']/'DATA_CONFIG.json');cfg=extractor_config(d)
    with patch('p2_testsets_eval.runtime.data_config',lambda:(d,cfg)):
        yield cfg


def render_whisper(root,rank,cfg):
    import torch
    from p2_testsets_eval import runtime as rt
    from p2_testsets_eval.shared_items import claim_items,all_items
    from p2_testsets_eval.whisper_runtime import matched_sample
    from p2_listen.cache import atomic_torch,cpu,tree_to
    from p2_listen.features import make_observation,unload
    from p2_whisper_av.features import SpeakerFeatures
    from omni_av_p2.connector import ConditionInputs
    from omni_av_p2.inference import reference_latents
    from omni_av_p2.ltx import unpack_latents
    from omni_av.ltx_runtime import with_text
    work=WHISPER;p=read(work/'protocol.json');arm=p['arms'][0]
    model,strategy,sampler=load_model(root,'whisper',cfg)
    save(work/'distributed32/model_acceptance'/f'{rank:02d}.json',dict(success=True,at=time.time(),rank=rank,
         physical_gpu=rank%8,host=os.uname().nodename,strict_native_whisper_load=True,step=6000))
    negative=tree_to(torch.load(work/'negative/07/payload.pt',map_location='cpu',weights_only=True),'cuda:0')
    with rt.cached_loaders(['load_video_vae_encoder']),torch.inference_mode():
        for i,item in enumerate(claim_items(work,all_items(work,[],p),p,arm,'render')):
            start=time.monotonic();folder=work/'conditions'/item['key'];g=item['geometry']
            plan=read(folder/'qwen/plan.json');meta=read(work/'whisper'/item['key']/'metadata.json')
            assert plan['adapter']==p['planner_adapter_identity'] and not plan['target_description_access']
            assert meta['encoder']==p['whisper_encoder'] and meta['feature_code']==p['feature_code']
            assert meta['source']['input_audit_sha256']==sha(folder/'input_audit.json')
            payload=torch.load(folder/'qwen/hidden.pt',map_location='cpu',weights_only=True)
            text=torch.load(folder/'text/payload.pt',map_location='cpu',weights_only=True)
            assert text['prompt']==payload['prompt']==plan['prompt']
            inputs=ConditionInputs(**tree_to(payload['condition'],'cuda:0'))
            with torch.autocast('cuda',dtype=torch.bfloat16):u=model.connector(inputs)
            speaker=SpeakerFeatures(**torch.load(work/'whisper'/item['key']/'payload.pt',map_location='cpu',weights_only=True)).validate().to('cuda:0')
            shape=dict(fps=g['fps'],height=g['height'],width=g['width'],frames=g['native_frames'],duration_s=g['native_duration_s'])
            av=reference_latents(cfg,make_observation(rt.observation(item,folder)),shape,'cuda:0')
            torch.manual_seed(p['seed']);prepared=strategy.prepare_training_inputs(with_text(av,tree_to(text,'cuda:0')),sampler)
            torch.manual_seed(p['seed']);sampled=matched_sample(model,prepared,u,speaker,negative,**p['generation'])
            video,audio=unpack_latents(sampled,av)
            assert torch.isfinite(video).all() and torch.isfinite(audio).all()
            out=work/'latents'/arm['name']/item['key'];atomic_torch(out/'payload.pt',cpu(dict(video=video,audio=audio)))
            save(out/'metadata.json',dict(key=item['key'],arm=arm['name'],step=6000,geometry=g,seed=p['seed'],
                 generation=p['generation'],checkpoint=arm['original_checkpoint'],checkpoint_sha256=arm['original_sha256'],
                 planning_adapter=p['planner_adapter_identity'],planning_sha256=sha(folder/'qwen/plan.json'),
                 target_description_access=False,listener_future_input=False,whisper_condition=True,whisper_source=meta,
                 seconds=time.monotonic()-start))
            rt.complete(out,[out/'payload.pt',out/'metadata.json'])
            rt.progress(work,'render',100+rank,done=i+1,total=400,last_seconds=time.monotonic()-start)
            del payload,text,inputs,u,speaker,av,prepared,sampled,video,audio
    del model,strategy,sampler,negative
    unload()


def worker(root,name,stage,rank,slot):
    assert name in INFO and stage in ('render','decode')
    assert isinstance(rank,int) and 0<=rank<32 and 0<=slot<4
    ready=read(root/'CODE_READY.json')
    assert all(sha(p)==h for p,h in ready['files'].items()),'Frozen evaluation execution changed'
    import torch
    cluster=read(root/'cluster.json');node=cluster['nodes'][rank//8]
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==str(rank%8)
    import socket
    ips=os.popen('hostname -I').read().split()
    assert node['host'] in ips,'Wrong worker node'
    torch.set_num_threads(4);torch.cuda.set_device(0)
    work=INFO[name]['work'];p=read(work/'protocol.json')
    save(work/'distributed32/devices'/f'{stage}_{rank:02d}_{slot}.json',dict(pid=os.getpid(),host=socket.gethostname(),
         rank=rank,physical_gpu=rank%8,stage=stage,slot=slot,at=time.time()))
    lock=work/'distributed32/locks'/f'{stage}_{rank:02d}_{slot}.lock';lock.parent.mkdir(parents=True,exist_ok=True)
    import fcntl
    with lock.open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        from p2_testsets_eval import runtime as rt
        from p2_testsets_eval.shared_items import all_items
        old_progress=rt.progress
        def progress(w,s,r,**kw):return old_progress(w,s,100+rank+32*slot,**kw)
        with runtime(name) as cfg,patch.object(rt,'progress',progress):
            if stage=='decode':rt.decode(work,all_items(work,[],p),100+rank+32*slot,p,p['arms'])
            elif name=='whisper':render_whisper(root,rank,cfg)
            else:
                import p2_current_feature6000_sft2752_eval_20261108 as feature
                old_save=feature.base.save
                def output_save(path,value):
                    if Path(path).name=='GPU_MODEL_ACCEPTANCE.json':
                        path=work/'distributed32/model_acceptance'/f'{rank:02d}.json'
                        value=dict(value,rank=rank,physical_gpu=rank%8,host=socket.gethostname())
                    return old_save(path,value)
                with patch.object(feature.base,'load_model',lambda p,c:load_model(root,name,c)),patch.object(feature.base,'save',output_save):
                    feature.base.render(work,p,cfg)
            progress(work,stage,0,status='complete',rank_global=rank,slot=slot)


def update_feature_topology(root):
    out=FEATURE/'distributed32';out.mkdir(exist_ok=True)
    p=read(FEATURE/'protocol.json')
    if not (out/'original_protocol.json').exists():save(out/'original_protocol.json',p)
    items=read(FEATURE/'shared_items.json')
    p.update(workers=32,cluster=str(root/'cluster.json'),metrics_cluster=str(root/'cluster.json'),
             metrics_workers_per_gpu=2,metrics_inference_workers_per_gpu=2,metrics_use_released_inference_nodes=False,
             authorized_scope='Four nodes32 GPUs; original completed outputs and current GPU7 worker preserved')
    save(FEATURE/'protocol.json',p)
    for rank,rows in enumerate(partition(items,32)):save(FEATURE/'shards'/f'{rank:02d}.json',rows)
    return p


def final(work,name):
    from p2_testsets_eval.runtime import ready
    p=read(work/'protocol.json');arm=p['arms'][0];results=read(work/'RESULTS.json')
    for x in read(work/'shared_items.json'):
        f=work/'generated'/arm['name']/x['key'];assert ready(f)
        m=read(f/'metadata.json')
        assert m['step']==6000 and m['checkpoint_sha256']==arm['original_sha256']
        assert m['planning_adapter']==p['planner_adapter_identity']
        assert m['target_description_access'] is False and m['listener_future_input'] is False
        assert m['whisper_condition']==(name=='whisper')
    for ds in p['datasets']:
        s=results[arm['name']+'/'+ds]
        assert s['requested_pairs']==s['audio']['requested_pairs']==200 and s['audio']['pairs_with_errors']==0
        assert all(s['coverage'][k]['pairs']==200 for k in ['psnr','ssim','lpips','fid','fvd'])
    save(work/'FINAL_AUDIT.json',dict(success=True,at=time.time(),outputs=400,step=6000,
         planner_step=2752,model=name,world=32,same_SFT2752_conditions=True))
    av=ROOT/'experiments/p2_current500h_av6000_sft2752_dual200_20261107'
    save(work/'COMPARISON.json',dict(current=results,av_only6000=read(av/'RESULTS.json'),
         joint6000=read(CONDITIONS/'RESULTS.json'),note='Same SFT2752 conditions; current500h same training schedule; '+name+' single-component increment versus pure AV'))
    save(work/'HANDOFF_COMPLETE.json',dict(outputs=400,step=6000,planner_step=2752,at=time.time(),world=32))
    save(work/'status.json',dict(stage='complete',outputs=400,at=time.time()))
