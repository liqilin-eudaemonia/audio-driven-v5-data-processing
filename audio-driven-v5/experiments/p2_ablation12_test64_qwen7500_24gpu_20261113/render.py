"""Native renderer loop; disabled Whisper is not read or supplied."""
import time
from p2_testsets_eval.protocol import read,save,sha
def render(work,items,rank,p):
    import torch
    from p2_testsets_eval import runtime as rt
    from p2_testsets_eval.whisper_runtime import matched_sample
    from p2_testsets_eval.shared_items import claim_items,all_items
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
            use_whisper=p['component']!='no_whisper'
            meta=read(work/'whisper'/item['key']/'metadata.json') if use_whisper else None
            plan=read(folder/'qwen/plan.json')
            if use_whisper:assert meta['encoder']==p['whisper_encoder'] and meta['feature_code']==p['feature_code']
            if use_whisper:assert meta['source']['input_audit_sha256']==sha(folder/'input_audit.json')
            assert plan['adapter']==p['planner_adapter_identity'] and plan['target_description_access'] is False
            payload=torch.load(folder/'qwen/hidden.pt',map_location='cpu',weights_only=True);text=torch.load(folder/'text/payload.pt',map_location='cpu',weights_only=True)
            assert text['prompt']==payload['prompt']==plan['prompt']
            inputs=ConditionInputs(**tree_to(payload['condition'],device))
            with torch.autocast('cuda',dtype=torch.bfloat16):u=model.connector(inputs)
            speaker=SpeakerFeatures(**torch.load(work/'whisper'/item['key']/'payload.pt',map_location='cpu',weights_only=True)).validate().to(device) if use_whisper else None
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

