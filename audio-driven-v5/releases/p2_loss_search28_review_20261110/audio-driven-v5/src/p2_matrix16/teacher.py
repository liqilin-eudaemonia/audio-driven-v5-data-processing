"""Frozen teacher targets from original video and recipe-processed listener audio."""
from pathlib import Path
import math
import torch
from p2_listen.contracts import read,sha,digest,atomic_json
from p2_listen.cache import load_part,save_part,cpu
from p2_plan_cache.protocol import lock


def identity(data,obs,target,root):
    from p2_listen.features import identity as native
    from listen_audio_postprocess.recipes import implementation_identity
    import scipy
    recipe=read(Path(data['output'])/'audio_postprocess/recipes'/(obs['sample_id']+'.json'))
    if recipe['source']['sha256']!=target['media_hashes']['target_audio']:raise ValueError('Teacher audio recipe mismatch')
    return dict(schema='p2-postprocessed-feature-teacher-v1',base=native(data,'teacher',obs,target),recipe=recipe,
                code={**implementation_identity(),str(Path(__file__)):sha(__file__)},
                resample='scipy.signal.resample_poly; mono channel mean; 16000Hz',scipy=scipy.__version__,
                frozen_base_teacher=True,mask_policy='full_temporal_support; silence_is_valid')


@torch.no_grad()
def extract(data,obs,target,layout,runtime,processor,recipe):
    import numpy as np
    from scipy.signal import resample_poly
    from p2_listen.features import image_grid,sample_frames,video_inputs
    from p2_listen.layout import aggregate_audio
    from omni_av.omni import move_inputs
    from listen_audio_postprocess.recipes import read_listener_audio
    model=runtime.thinker;device=next(model.parameters()).device
    if any(p.requires_grad for p in model.parameters()):raise ValueError('Teacher must be independent frozen base Qwen')
    _,size,_=image_grid(obs,processor,runtime.config)
    count=layout['video_intervals'].shape[1]*2
    frames=sample_frames(target['target_video'],[i/8 for i in range(count)],size)
    vi=move_inputs(video_inputs(processor,frames),device)
    video=model.get_video_features(vi['pixel_values_videos'],vi['video_grid_thw'])[None]
    wave,rate,meta=read_listener_audio(recipe)
    mono=wave.mean(axis=1)
    if rate!=16000:
        divisor=math.gcd(rate,16000)
        mono=resample_poly(mono,16000//divisor,rate//divisor).astype(np.float32)
    # One rounding sample at most; no time cropping or inferred silence mask.
    expected=round(obs['requested_duration_s']*16000)
    if abs(len(mono)-expected)>1:raise ValueError('Resampled audio duration differs')
    mono=np.pad(mono[:expected],(0,max(0,expected-len(mono))))
    ai=processor.feature_extractor(mono,sampling_rate=16000,return_tensors='pt',return_attention_mask=True,padding='max_length')
    audio=model.get_audio_features(ai['input_features'].to(device,torch.bfloat16),ai['attention_mask'].to(device))
    if audio.ndim!=2:raise ValueError('Unexpected audio tower shape')
    audio,audio_valid=aggregate_audio(audio[None],layout['audio_intervals'])
    result=dict(video=video,audio=audio,video_valid=layout['video_target_valid'].to(device),
                audio_valid=layout['audio_target_valid'].to(device)&audio_valid,waveform=meta)
    for mod in ('video','audio'):
        if result[mod].shape!=(*layout[mod+'_valid'].shape,runtime.config.target_dim) or not torch.isfinite(result[mod]).all():
            raise ValueError('Teacher geometry/nonfinite mismatch')
    return cpu(result)


def cache_teacher(config,root,ids=None,rank=0,shards=1,device='cuda:0'):
    from .config import data_config
    from p2_listen.config import extractor_config
    from p2_listen.features import image_grid,unload
    from p2_listen.layout import make_layout
    from .data import metadata_targets
    from p2_listen.contracts import observation
    data=data_config(config);cfg=extractor_config(data)
    from omni_av_p2.data import load_omni
    root=Path(root)
    obs=read(Path(data['output'])/'manifests/observations.json')
    if ids is not None:
        if set(ids)-{o['sample_id'] for o in obs}:raise ValueError('Unknown teacher sample')
        obs=[o for o in obs if o['sample_id'] in ids]
    if not 0<=rank<shards:raise ValueError('Bad shard')
    obs=sorted(obs,key=lambda o:o['sample_id'])[rank::shards]
    targets=metadata_targets(data,obs);runtime=processor=None;rows=[]
    try:
        for o in obs:
            sid=o['sample_id'];target=targets[sid]
            try:
                with lock(root/'locks'/(sid+'.lock')):
                    ident=identity(data,o,target,root)
                    existing=load_part(root/'cache'/sid,ident,missing_ok=True)
                    if existing is None:
                        if runtime is None:runtime,processor=load_omni(cfg,device)
                        observation(o,verify=True)
                        if sha(target['target_video'])!=target['media_hashes']['target_video']:raise ValueError('Video source changed')
                        _,_,grid=image_grid(o,processor,runtime.config)
                        layout=make_layout(o['requested_duration_s'],grid,8,8)
                        value=extract(data,o,target,layout,runtime,processor,ident['recipe'])
                        save_part(root/'cache'/sid,ident,value)
                    rows.append(dict(sample_id=sid,success=True))
            except Exception as exc:rows.append(dict(sample_id=sid,success=False,error=str(exc)))
    finally:
        del runtime,processor;unload()
    report=dict(success=all(r['success'] for r in rows),samples=len(rows),results=rows)
    atomic_json(root/f'report-{rank}-of-{shards}.json',report)
    return report
