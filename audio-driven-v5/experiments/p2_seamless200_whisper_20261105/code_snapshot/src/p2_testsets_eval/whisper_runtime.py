"""Matched Seamless200 evaluation: original Qwen conditions plus speaker Whisper.

This module does not change the frozen Whisper training implementation. Native
sampling is deliberately imported from the original ten-arm evaluator.
"""
import fcntl
import hashlib
import math
import os
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from .protocol import ROOT, read, save, sha
from .runtime import ready, complete, progress, cached_loaders, observation, data_config


def speaker_wave(item, folder):
    """Read the exact speaker slice used for the sealed Qwen input, at pair t=0."""
    import soundfile as sf
    from scipy.signal import resample_poly
    audit = read(folder/'input_audit.json')
    if not (audit['source_only'] is True and audit['target_description_in_model_input'] is False
            and audit['listener_future_media_in_model_input'] is False):
        raise ValueError('Source-only conditioning audit failed')
    path = Path(audit['speaker_audio_path'])
    if sha(path) != audit['speaker_audio_sha256']:
        raise ValueError('Speaker source changed')
    with sf.SoundFile(path) as audio:
        rate = audio.samplerate
        first, last = audit['speaker_first_sample'], audit['speaker_last_sample']
        if not 0 <= first < last <= len(audio):
            raise ValueError('Speaker range outside source')
        audio.seek(first)
        wave = audio.read(last-first, dtype='float32', always_2d=True).mean(1)
    if rate != 16000:
        common = math.gcd(rate,16000)
        wave = resample_poly(wave,16000//common,rate//common).astype(np.float32)
    count = round(item['geometry']['duration_s']*16000)
    delta = count-len(wave)
    if abs(delta)>1 or not np.isfinite(wave).all():
        raise ValueError('Whisper/Qwen speaker duration mismatch')
    wave = np.pad(wave,(0,max(0,delta)))[:count]
    return wave,dict(input_audit_sha256=sha(folder/'input_audit.json'),
                     waveform_sha256=hashlib.sha256(wave.astype('<f4').tobytes()).hexdigest(),
                     source_sha256=audit['speaker_audio_sha256'],samples=count,rate=16000,
                     origin_s=0.,silence_removed=False,source_only=True)


def cache(work,items,rank,p):
    from dataclasses import asdict
    from p2_listen.cache import atomic_torch
    from p2_whisper_av.features import FrozenWhisper, validate_encoder_batching
    from p2_listen.features import unload
    todo=[x for x in items if not ready(work/'whisper'/x['key'])]
    if not todo:return
    encoder=FrozenWhisper(p['whisper_encoder']['path'],'cuda:0')
    # Test serial/batch equivalence on the actual encoder once on rank 0.
    if rank==0 and not (work/'ENCODER_BATCH_AUDIT.json').exists():
        report=validate_encoder_batching(encoder,batch_size=p['whisper_batch'])
        save(work/'ENCODER_BATCH_AUDIT.json',report)
    for start in range(0,len(todo),p['whisper_batch']):
        batch=todo[start:start+p['whisper_batch']]
        with ThreadPoolExecutor(max_workers=4) as pool:
            sources=list(pool.map(lambda x:speaker_wave(x,work/'conditions'/x['key']),batch))
        features=encoder.encode_batch([x[0] for x in sources],batch_size=p['whisper_batch'])
        for item,feature,(_,source) in zip(batch,features,sources):
            folder=work/'whisper'/item['key'];folder.mkdir(parents=True,exist_ok=True)
            feature.validate()
            if abs(float(feature.bounds[0,0,0,0]))>1e-7:
                raise ValueError('Whisper starts outside pair origin')
            atomic_torch(folder/'payload.pt',asdict(feature))
            save(folder/'metadata.json',dict(source=source,encoder=p['whisper_encoder'],
                feature_code=p['feature_code'],tokens=int(feature.valid.sum())))
            complete(folder,[folder/'payload.pt',folder/'metadata.json'])
        progress(work,'cache',rank,done=min(start+len(batch),len(todo)),total=len(todo))
    del encoder;unload()


def matched_sample(model,prepared,u,speaker,negative,**kwargs):
    from p2_listen.runtime import guided_sample,append_negative
    return guided_sample(model.renderer,model.prepare(prepared,u,speaker),
                         append_negative(negative,u),**kwargs)


def render(work,items,rank,p,arms):
    import torch
    from p2_listen.cache import atomic_torch,cpu,tree_to
    from p2_listen.features import unload,make_observation
    from p2_whisper_av.inference import load_model
    from p2_whisper_av.checkpoint import checked
    from p2_whisper_av.config import source_identity
    from p2_whisper_av.features import SpeakerFeatures
    from omni_av_p2.connector import ConditionInputs
    from omni_av_p2.inference import reference_latents
    from omni_av_p2.ltx import unpack_latents
    from omni_av.ltx_runtime import with_text
    device='cuda:0';_,cfg=data_config()
    if not any(not ready(work/'latents'/a['name']/x['key']) for a in arms for x in items):return
    for arm in arms:
        if read(Path(arm['checkpoint']).with_suffix('.complete.json'))['sha256']!=arm['checkpoint_sha256']:
            raise ValueError('Inference checkpoint marker changed after preparation')
    model,strategy,sampler=load_model(arms[0]['checkpoint'],device)
    model.eval().requires_grad_(False);renderer=model.renderer
    renderer.eval().requires_grad_(False);renderer.set_gradient_checkpointing(False)
    negative=tree_to(torch.load(work/'negative'/f'{rank:02d}'/'payload.pt',map_location='cpu',weights_only=True),device)
    with cached_loaders(['load_video_vae_encoder']),torch.inference_mode():
        for arm_index,arm in enumerate(arms):
            # load_model already checked the first package SHA, identity, assets,
            # all tensors and strict state load. Do not read its 11GB twice.
            if arm_index:
                saved=checked(arm['checkpoint'])
                if saved['identity']['source']!=source_identity() or saved['step']!=arm['step']:
                    raise ValueError('Whisper checkpoint source/step mismatch')
                if saved['identity']!=p['checkpoint_identity']:
                    raise ValueError('Unexpected Whisper architecture or training identity')
                model.load_state_dict(saved['model'],strict=True);del saved
            for index,item in enumerate(items):
                out=work/'latents'/arm['name']/item['key']
                if ready(out):continue
                start=time.monotonic();folder=work/'conditions'/item['key'];g=item['geometry']
                for part in ['qwen','text']:
                    if not ready(folder/part):raise ValueError('Incomplete baseline condition')
                feature_folder=work/'whisper'/item['key']
                if not ready(feature_folder):raise ValueError('Whisper cache missing')
                meta=read(feature_folder/'metadata.json')
                if (meta['encoder']!=p['whisper_encoder'] or meta['feature_code']!=p['feature_code']
                        or meta['source']['input_audit_sha256']!=sha(folder/'input_audit.json')):
                    raise ValueError('Whisper conditioning identity mismatch')
                payload=torch.load(folder/'qwen/hidden.pt',map_location='cpu',weights_only=True)
                text=torch.load(folder/'text/payload.pt',map_location='cpu',weights_only=True)
                plan=read(folder/'qwen/plan.json')
                if not (text['prompt']==payload['prompt']==plan['prompt'] and plan['adapter'] is None):
                    raise ValueError('Expected unchanged base Qwen conditions')
                inputs=ConditionInputs(**tree_to(payload['condition'],device))
                with torch.autocast('cuda',dtype=torch.bfloat16):u=model.connector(inputs)
                speaker=SpeakerFeatures(**torch.load(feature_folder/'payload.pt',map_location='cpu',weights_only=True)).validate().to(device)
                native=dict(fps=g['fps'],height=g['height'],width=g['width'],frames=g['native_frames'],duration_s=g['native_duration_s'])
                av=reference_latents(cfg,make_observation(observation(item,folder)),native,device)
                torch.manual_seed(p['seed'])
                prepared=strategy.prepare_training_inputs(with_text(av,tree_to(text,device)),sampler)
                torch.manual_seed(p['seed'])
                sampled=matched_sample(model,prepared,u,speaker,negative,**p['generation'])
                video,audio=unpack_latents(sampled,av)
                atomic_torch(out/'payload.pt',cpu(dict(video=video,audio=audio)))
                save(out/'metadata.json',dict(key=item['key'],arm=arm['name'],checkpoint=arm['original_checkpoint'],
                    checkpoint_sha256=arm['original_sha256'],inference_checkpoint=arm['checkpoint'],
                    step=arm['step'],seed=p['seed'],geometry=g,planning_sha256=sha(folder/'qwen/plan.json'),
                    target_description_access=False,listener_future_input=False,generation=p['generation'],
                    whisper_source=meta,sampling=p['sampling'],seconds=time.monotonic()-start))
                complete(out,[out/'payload.pt',out/'metadata.json'])
                progress(work,'render',rank,arm=arm['name'],done=index+1,total=len(items),last_seconds=time.monotonic()-start)
                del payload,text,inputs,u,speaker,av,prepared,sampled,video,audio
    del model,renderer,strategy,sampler,negative;unload()


def worker(work,stage,rank,limit=None,arm_names=None):
    from p2_whisper_av.bootstrap import activate_ltx
    activate_ltx()
    import torch
    torch.set_num_threads(4);torch.cuda.set_device(0)
    work=Path(work);(work/'locks').mkdir(exist_ok=True)
    with (work/'locks'/f'{stage}_{rank:02d}.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        p=read(work/'protocol.json')
        for name,expected in p['execution_code'].items():
            if sha(name)!=expected:raise ValueError('Queued execution code changed: '+name)
        items=read(work/'shards'/f'{rank:02d}.json')
        if limit:items=items[:limit]
        arms=[a for a in p['arms'] if arm_names is None or a['name'] in arm_names]
        progress(work,stage,rank,status='starting',samples=len(items))
        try:
            data_config()
            if stage=='cache':cache(work,items,rank,p)
            elif stage=='render':render(work,items,rank,p,arms)
            elif stage=='decode':
                from .runtime import decode
                decode(work,items,rank,p,arms)
            else:raise ValueError(stage)
            progress(work,stage,rank,status='complete',samples=len(items))
        except Exception as e:
            progress(work,stage,rank,status='failed',error=repr(e));raise
