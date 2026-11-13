"""Pinned all-attention/FFN LoRA6000 inference on the existing Seamless200."""
import copy
import fcntl
import json
import os
from pathlib import Path
import socket
import time
from unittest.mock import patch

from p2_testsets_eval.protocol import ROOT,read,save,sha,partition

TRAIN=ROOT/'experiments/p2_ltx_all_attention_ffn_lora31_20261108'
PARENT=ROOT/'experiments/p2_ltx6000_latest_sft_dual200_20261107'
WORK=ROOT/'experiments/p2_ltx_all_lora6000_seamless200_sft2752_eval32_20261109'
CLI=ROOT/'scripts/p2_all_lora6000_seamless_eval32_20261109.py'
ARM='ltx_all_lora_event_feature_whisper_6000_sft2752'


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
    original=TRAIN/'step_006000.pt';marker=read(original.with_suffix('.complete.json'))
    f=work/'export/RECEIPT.json'
    if f.exists():
        rec=read(f)
        assert rec['original_sha256']==marker['sha256']
        assert stamp(rec['checkpoint'])==rec['stamp']
        return rec
    v=checked(original);assert v['step']==6000 and len(v['rank_rng'])==31
    assert v['identity']==read(TRAIN/'run.json') and v['identity']['source']==source_identity()
    c=validate(v['identity']['experiment'])
    assert c['loss_weights']==dict(av=1.,event=.05,video_feature=.025,audio_feature=.025)
    assert v['identity']['ltx_lora_report']['projections']==1632
    assert v['identity']['ltx_lora_report']['parameters']==84221952
    assert all(torch.isfinite(t).all() for s in v['optimizer']['state'].values() for t in s.values() if torch.is_tensor(t))
    assert {int(s['step']) for s in v['optimizer']['state'].values() if 'step' in s}=={6000}
    target=work/'export/step_006000.model.pt'
    atomic_torch(target,{k:v[k] for k in ['schema','step','identity','model']})
    exported=torch.load(target,map_location='cpu',weights_only=True,mmap=True)
    assert exported['identity']==v['identity'] and set(exported['model'])==set(v['model'])
    assert all(torch.equal(t,exported['model'][k]) for k,t in v['model'].items())
    rec=dict(success=True,at=time.time(),original=str(original),original_sha256=marker['sha256'],
        checkpoint=str(target),checkpoint_sha256=sha(target),stamp=stamp(target),
        identity=v['identity'],schema=v['schema'],step=6000,all_model_tensors_bitwise_equal=True,
        full_model_adam_and_31_rng_preserved=True)
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
    for n in cluster['nodes']:
        n['gpus']=list(range(8))
        if n['name']=='node11':n['ssh_key']='/private/ssh_keys/root_dedicated-developjob-js-infer-machine-ws3qe_ed25519'
    assert [n['name'] for n in cluster['nodes']]==['node12','node8','node4','node11']
    cluster['training_script']=str(CLI);save(work/'cluster.json',cluster)
    from p2_ltx_all_lora.data import data_config
    save(work/'DATA_CONFIG.json',data_config(rec['identity']['experiment']))
    save(work/'shared_items.json',items)
    for rank,rows in enumerate(partition(items,32)):save(work/'shards'/f'{rank:02d}.json',rows)
    p=copy.deepcopy(parent);p.update(schema='all-ltx-lora6000-seamless200-eval32-v1',created_at=time.time(),
        arms=[dict(name=ARM,kind='ltx_all_lora_full',step=6000,checkpoint=rec['checkpoint'],
            checkpoint_sha256=rec['checkpoint_sha256'],original_checkpoint=rec['original'],
            original_sha256=rec['original_sha256'],planner=parent['planner'])],
        checkpoint_identity=rec['identity'],datasets={'seamless':spec},workers=32,total_samples=200,total_outputs=200,
        cluster=str(work/'cluster.json'),metrics_cluster=str(work/'cluster.json'),worker_cli=str(CLI),
        shared_items_manifest=str(work/'shared_items.json'),shared_items_sha256=sha(work/'shared_items.json'),
        metrics_workers_per_gpu=2,metrics_inference_workers_per_gpu=2,metrics_use_released_inference_nodes=False,
        execution_code=execution_code(),authorized_scope='Four nodes32 GPUs; preserve unrelated vLLM',
        conditioning='Unchanged SFT2752 plans/H/Gemma, speaker Whisper and listener first frame',
        execution_order='Shared32 render; released GPUs decode and score ready shards; native full200 merge')
    save(work/'protocol.json',p)
    save(work/'CPU_ACCEPTANCE.json',dict(success=True,at=time.time(),outputs=200,seed=137,planner_step=2752,
         model_export_bitwise_equal=True,all_282_training_sources_exact=True,fixed200=True,no_cache_increment=True,
         rank00_negative_equals_all_original8=True))
    return p


def pinned(work):
    import torch
    r=read(work/'export/RECEIPT.json');assert stamp(r['checkpoint'])==r['stamp']
    v=torch.load(r['checkpoint'],map_location='cpu',weights_only=True,mmap=True)
    assert stamp(r['checkpoint'])==r['stamp'] and v['identity']==r['identity']
    assert v['schema']==r['schema'] and v['step']==6000
    return v


def load_model(work,path,device='cuda:0'):
    from p2_ltx_all_lora import inference,checkpoint
    p=read(work/'protocol.json');assert str(path)==p['arms'][0]['checkpoint']
    with patch.object(checkpoint,'checked',lambda _:pinned(work)):
        model,strategy,sampler=inference.load_model(path,device)
    return model,strategy,sampler


def worker(work,stage,rank,slot=0):
    assert 0<=rank<32 and 0<=slot<4 and os.environ.get('CUDA_VISIBLE_DEVICES')==str(rank%8)
    p=read(work/'protocol.json');assert p['execution_code']==execution_code()
    node=read(work/'cluster.json')['nodes'][rank//8]
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
             host=socket.gethostname(),rank=rank,physical_gpu=rank%8,stage=stage))
        old_progress=rt.progress
        def progress(w,s,r,**kw):return old_progress(w,s,rank+32*slot,**kw)
        with patch.object(rt,'data_config',lambda:(d,cfg)),patch.object(rt,'progress',progress):
            if stage=='render':
                import p2_ltx6000_latest_sft_dual200_20261107 as native
                import p2_ltx_events.inference as previous
                def loading(path,device='cuda:0'):
                    result=load_model(work,path,device)
                    save(work/'model_acceptance'/f'{rank:02d}.json',dict(success=True,at=time.time(),rank=rank,
                         physical_gpu=rank%8,strict_all_lora_load=True,targets=len(result[0].text_targets)))
                    return result
                with patch.object(native,'ARM',ARM),patch.object(previous,'load_model',loading):
                    native.render(work,all_items(work,[],p),0,p)
            else:rt.decode(work,all_items(work,[],p),rank,p,p['arms'])
        progress(work,stage,0,status='complete',slot=slot)


def final(work):
    from p2_testsets_eval.runtime import ready
    p=read(work/'protocol.json')
    for x in read(work/'shared_items.json'):
        f=work/'generated'/ARM/x['key'];assert ready(f);m=read(f/'metadata.json')
        assert m['step']==6000 and m['seed']==137 and m['checkpoint_sha256']==p['arms'][0]['original_sha256']
        assert m['planning_adapter']==p['planner_adapter_identity'] and m['generation']==p['generation']
        assert not m['target_description_access'] and not m['listener_future_input']
    r=read(work/'RESULTS.json')[ARM+'/seamless']
    assert r['requested_pairs']==r['audio']['requested_pairs']==200 and r['audio']['pairs_with_errors']==0
    assert all(r['coverage'][k]['pairs']==200 for k in ['psnr','ssim','lpips','fid','fvd'])
    save(work/'FINAL_AUDIT.json',dict(success=True,at=time.time(),step=6000,outputs=200,world=32,
         planner_step=2752,seed=137,all_attention_ffn_lora=True))
    save(work/'COMPARISON.json',dict(current=r,previous_text_lora=read(PARENT/'RESULTS.json')[
         'ltx_hidden_event_feature_whisper_6000_sft2752/seamless']))
    save(work/'HANDOFF_COMPLETE.json',dict(success=True,at=time.time(),outputs=200))
