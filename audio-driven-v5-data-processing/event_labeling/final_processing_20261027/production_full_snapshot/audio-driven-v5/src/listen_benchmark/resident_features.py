"""Resident native encoders. Tensor definitions match p2_listen.features/runtime."""
import math
from pathlib import Path
import time
import torch
from contextlib import contextmanager


@contextmanager
def workspace_lease(folder,slots):
    if folder is None or slots<=0:
        yield;return
    import fcntl
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);held=None
    try:
        while held is None:
            for i in range(slots):
                handle=(folder/f'{i}.lock').open('a')
                try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:handle.close();continue
                held=handle;break
            if held is None:time.sleep(.01)
        yield
    finally:
        if held is not None:held.close()


def prepare_vae(c,obs,target):
    import numpy as np
    import soundfile as sf
    from PIL import Image
    from p2_listen.layout import geometry
    from p2_listen.features import sample_frames
    g=geometry(obs)
    pixels=sample_frames(target['target_video'],[i/g['fps'] for i in range(g['frames'])],(g['width'],g['height']))
    with Image.open(obs['listener_image']) as im:
        if im.size!=(g['width'],g['height']):raise ValueError('Reference/crop geometry mismatch')
        first=np.asarray(im.convert('RGB')).copy()
    wave,rate=sf.read(target['target_audio'],dtype='float32',always_2d=True)
    count=round(g['duration_s']*rate)
    wave=np.pad(wave,((0,max(0,count-len(wave))),(0,0)))[:count]
    return dict(geometry=g,pixels=pixels,first=first,wave=wave,rate=rate)


class ResidentVAE:
    def __init__(self,c,device):
        from p2_listen.config import extractor_config
        from ltx_core.model.audio_vae import AudioProcessor
        from ltx_trainer.model_loader import load_video_vae_encoder,load_audio_vae_encoder
        cfg=extractor_config(c);self.device=device
        self.video=load_video_vae_encoder(cfg.ltx_path,device=device,dtype=torch.bfloat16).requires_grad_(False).eval()
        self.audio=load_audio_vae_encoder(cfg.ltx_path,device=device,dtype=torch.float32).requires_grad_(False).eval()
        ae=self.audio
        if not getattr(ae,'is_causal',False) or ae.sample_rate!=16000 or ae.mel_hop_length!=160:raise ValueError('Unsupported audio VAE time contract')
        self.processor=AudioProcessor(target_sample_rate=ae.sample_rate,mel_bins=ae.mel_bins,mel_hop_length=ae.mel_hop_length,n_fft=ae.n_fft).to(device)

    @torch.no_grad()
    def encode(self,prepared):
        from ltx_core.types import Audio,AudioLatentShape
        from p2_listen.runtime import align_audio_latents,support_masks
        from p2_listen.cache import cpu
        g=prepared['geometry'];device=self.device
        value=torch.from_numpy(prepared['pixels']).permute(3,0,1,2)[None].to(device,torch.bfloat16)/127.5-1
        ref=torch.from_numpy(prepared['first']).permute(2,0,1)[None,:,None].to(device,torch.bfloat16)/127.5-1
        with torch.autocast('cuda',dtype=torch.bfloat16):
            z=self.video(value);z[:,:,:1]=self.video(ref)
        if z.shape!=(1,128,(g['frames']-1)//8+1,g['height']//32,g['width']//32):raise ValueError('Video VAE geometry mismatch')
        del value,ref
        value=torch.from_numpy(prepared['wave'].T.copy())[None].to(device)
        if value.shape[1]==1:value=value.repeat(1,2,1)
        az=self.audio(self.processor.waveform_to_mel(Audio(waveform=value,sampling_rate=prepared['rate'])).float())
        expected=tuple(AudioLatentShape.from_duration(1,g['duration_s']).to_torch_shape())
        az,endpoint=align_audio_latents(az,expected)
        ae=self.audio;margin=ae.n_fft/(2*ae.sample_rate)+ae.mel_hop_length/ae.sample_rate
        support=support_masks(g,z.shape,az.shape,margin);scalar=lambda x:torch.tensor([x])
        if not torch.isfinite(z).all() or not torch.isfinite(az).all():raise ValueError('Nonfinite VAE targets')
        return cpu(dict(av=dict(latents=dict(latents=z,num_frames=scalar(z.shape[2]),height=scalar(z.shape[3]),width=scalar(z.shape[4]),fps=scalar(g['fps'])),
            audio_latents=dict(latents=az.to(torch.bfloat16),num_time_steps=scalar(az.shape[2]),duration=scalar(g['duration_s']))),
            support=support,geometry=g,audio_support_margin_s=margin,audio_endpoint_tokens_removed=endpoint))


class ResidentFeatures:
    def __init__(self,c,device='cuda:0',native_prefetch=False,workspace_slots=0,workspace_lock_dir=None):
        from p2_listen.config import extractor_config
        from omni_av_p2.data import load_omni
        self.c=c;self.device=device;self.native_prefetch=native_prefetch
        self.runtime,self.processor=load_omni(extractor_config(c),device)
        self.vae=ResidentVAE(c,device)
        self.workspace_slots=workspace_slots;self.workspace_lock_dir=workspace_lock_dir
        self.resident_allocated_mib=torch.cuda.memory_allocated()/2**20

    def encode_vae(self,prepared):
        if self.workspace_slots:torch.cuda.empty_cache()
        with workspace_lease(self.workspace_lock_dir,self.workspace_slots):
            try:return self.vae.encode(prepared)
            finally:
                if self.workspace_slots:torch.cuda.empty_cache()

    def prepare(self,obs,target):
        value=prepare_vae(self.c,obs,target)
        if not self.native_prefetch:return value
        from PIL import Image
        from omni_av.data import read_audio
        from p2_listen.features import image_grid,sample_frames,video_inputs
        from p2_listen.layout import make_layout
        from p3_listen.prompts import OMNI
        processor=self.processor;cfg=self.runtime.config
        wave=read_audio(obs['speaker_audio'],end_s=obs['requested_duration_s'])
        if len(wave)!=round(obs['requested_duration_s']*16000):raise ValueError('Speaker audio must cover exact window')
        with Image.open(obs['listener_image']) as im:image=im.convert('RGB')
        messages=[dict(role='user',content=[dict(type='image'),dict(type='audio'),dict(type='text',text=OMNI)])]
        prompt=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        inputs=dict(processor(text=[prompt],images=[image],audio=[wave],sampling_rate=16000,return_tensors='pt',padding=True,truncation=False,images_kwargs={'max_pixels':cfg.vision_max_pixels,'min_pixels':28*28*4}))
        _,size,grid=image_grid(obs,processor,cfg);layout=make_layout(obs['requested_duration_s'],grid,cfg.vision_fps,cfg.audio_hz)
        frames_n=layout['video_intervals'].shape[1]*2
        frames=sample_frames(target['target_video'],[i/8 for i in range(frames_n)],size)
        vi=video_inputs(processor,frames)
        wave=read_audio(target['target_audio'],end_s=obs['requested_duration_s'])
        ai=processor.feature_extractor(wave,sampling_rate=16000,return_tensors='pt',return_attention_mask=True,padding='max_length')
        value['native']=dict(inputs=inputs,layout=layout,teacher_size=list(size),video=vi,audio=ai)
        return value

    @torch.no_grad()
    def input_prepared(self,value):
        from omni_av.omni import move_inputs
        from omni_av_p2.data import select_hidden_layers
        from p2_listen.cache import cpu
        model=self.runtime.thinker;inputs=move_inputs(value['inputs'],self.device);ids=inputs['input_ids']
        embeds=model.get_input_embeddings()(ids)
        audio=model.get_audio_features(inputs['input_features'],inputs['feature_attention_mask'])
        image=model.get_image_features(inputs['pixel_values'],inputs['image_grid_thw'])
        image_mask,_,audio_mask=model.get_placeholder_mask(ids,inputs_embeds=embeds,image_features=image)
        if audio_mask.sum()!=audio.numel() or image_mask.sum()!=image.numel():raise ValueError('Multimodal placeholder mismatch')
        embeds=embeds.masked_scatter(audio_mask,audio.to(embeds)).masked_scatter(image_mask,image.to(embeds))
        positions,_=model.get_rope_index(ids,image_grid_thw=inputs['image_grid_thw'],attention_mask=inputs['attention_mask'],use_audio_in_video=False,audio_seqlens=inputs['feature_attention_mask'].sum(-1))
        valid=inputs['attention_mask'].bool()
        outputs=model.model(inputs_embeds=embeds.detach(),position_ids=positions,attention_mask=valid.long(),output_hidden_states=True,use_cache=False,return_dict=True)
        h=select_hidden_layers(outputs.hidden_states,self.runtime.config.hidden_layers,model.config.text_config.num_hidden_layers)
        condition=dict(hidden=h,hidden_valid=valid,layer_ids=torch.tensor(self.runtime.config.hidden_layers,device=h.device))
        for mod in ('video','audio'):
            for key in ('_coordinates','_valid'):condition[mod+key]=value['layout'][mod+key]
        return cpu(dict(condition=condition,layout=value['layout'],teacher_size=value['teacher_size']))

    @torch.no_grad()
    def teacher_prepared(self,inputs,value):
        from omni_av.omni import move_inputs
        from p2_listen.layout import aggregate_audio
        from p2_listen.cache import cpu
        model=self.runtime.thinker;layout=inputs['layout'];vi=move_inputs(value['video'],self.device)
        video=model.get_video_features(vi['pixel_values_videos'],vi['video_grid_thw'])[None]
        ai=value['audio'];audio=model.get_audio_features(ai['input_features'].to(self.device,torch.bfloat16),ai['attention_mask'].to(self.device))
        if audio.ndim!=2 or not math.isclose(self.processor.feature_extractor.hop_length/self.processor.feature_extractor.sampling_rate,.01):raise ValueError('Native audio geometry differs')
        audio,valid=aggregate_audio(audio[None],layout['audio_intervals'])
        result=dict(video=video,audio=audio,video_valid=layout['video_target_valid'].to(self.device),audio_valid=layout['audio_target_valid'].to(self.device)&valid)
        for mod in ('video','audio'):
            if result[mod].shape!=(*inputs['condition'][mod+'_valid'].shape,self.runtime.config.target_dim):raise ValueError('Teacher/query shape mismatch')
        return cpu(result)

    @torch.no_grad()
    def process(self,obs,target,prepared=None):
        from p2_listen.features import identity,folder,extract_input,extract_teacher
        from p2_listen.cache import load_part,save_part
        from p2_listen.contracts import read,path_at
        from p2_listen.annotations import validate,validate_contract,rasterize
        c=self.c;sid=obs['sample_id'];times={};start=time.time()
        ident=identity(c,'input',obs);data=load_part(folder(c,'input',sid),ident,missing_ok=True)
        if data is None:
            data=self.input_prepared(prepared['native']) if prepared and 'native' in prepared else extract_input(c,obs,self.runtime,self.processor)
            save_part(folder(c,'input',sid),ident,data)
        times['input']=time.time()-start;start=time.time()
        ident=identity(c,'teacher',obs,target)
        if load_part(folder(c,'teacher',sid),ident,missing_ok=True) is None:
            teacher=self.teacher_prepared(data,prepared['native']) if prepared and 'native' in prepared else extract_teacher(c,obs,target,data,self.runtime,self.processor)
            save_part(folder(c,'teacher',sid),ident,teacher)
        times['teacher']=time.time()-start;start=time.time()
        ident=identity(c,'vae',obs,target)
        if load_part(folder(c,'vae',sid),ident,missing_ok=True) is None:
            value=self.encode_vae(prepared or prepare_vae(c,obs,target));save_part(folder(c,'vae',sid),ident,value)
        times['vae']=time.time()-start;start=time.time()
        label=read(path_at(c['output'])/'labels/canonical'/f'{sid}.json')
        validate(label,obs,require_admitted=False);validate_contract(label,c)
        ident=identity(c,'events',obs,label=label)
        if load_part(folder(c,'events',sid),ident,missing_ok=True) is None:
            save_part(folder(c,'events',sid),ident,rasterize(label,data['layout'],c['events']['K_train']))
        times['events']=time.time()-start
        return dict(sample_id=sid,seconds=times,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                    peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)


def compare_trees(a,b,path='root'):
    if isinstance(a,torch.Tensor):
        if not isinstance(b,torch.Tensor) or a.dtype!=b.dtype or a.shape!=b.shape or not torch.equal(a,b):
            diff=float((a.float()-b.float()).abs().max()) if isinstance(b,torch.Tensor) and a.shape==b.shape else None
            raise AssertionError(f'Resident/reference tensor differs: {path}; max_abs={diff}')
    elif isinstance(a,dict):
        if set(a)!=set(b):raise AssertionError(f'Keys differ: {path}')
        for k in a:compare_trees(a[k],b[k],path+'/'+k)
    elif isinstance(a,(list,tuple)):
        if len(a)!=len(b):raise AssertionError(f'Length differs: {path}')
        for i,(x,y) in enumerate(zip(a,b)):compare_trees(x,y,path+f'/{i}')
    elif a!=b:raise AssertionError(f'Metadata differs: {path}')


def canary(run,device='cuda:0'):
    from .full_fusion import read,atomic
    from p2_listen.config import activate
    from p2_listen.runtime import encode_vae
    from p2_listen.features import unload,cache_parts
    run=Path(run);cfg=read(run/'config.json');c=read(run/'p2_config.json');activate(c)
    obs=sorted(read(cfg['canary_observations']),key=lambda x:x['requested_duration_s'])
    selected=[obs[0],obs[-1]];targets={x['sample_id']:x for x in read(cfg['canary_targets'])}
    reference=[];started=time.time()
    for o in selected:reference.append(encode_vae(c,o,targets[o['sample_id']],device))
    encoder=ResidentVAE(c,device)
    for o,ref in zip(selected,reference):compare_trees(ref,encoder.encode(prepare_vae(c,o,targets[o['sample_id']])))
    del encoder,reference;unload()
    # One writer creates the shared fixed positive/negative text cache.
    cache_parts(c,[],{},['text'],device)
    torch.cuda.reset_peak_memory_stats()
    # Measure full resident model peak on the longest supported production geometry.
    session=ResidentFeatures(c,device,native_prefetch=True);o=dict(obs[-1]);t=targets[o['sample_id']]
    from p2_listen.features import extract_input,extract_teacher
    for checked in selected:
        checked_target=targets[checked['sample_id']];prepared_native=session.prepare(checked,checked_target)
        native_input=extract_input(c,checked,session.runtime,session.processor);prefetched_input=session.input_prepared(prepared_native['native']);compare_trees(native_input,prefetched_input)
        native_teacher=extract_teacher(c,checked,checked_target,native_input,session.runtime,session.processor);compare_trees(native_teacher,session.teacher_prepared(prefetched_input,prepared_native['native']))
    del prepared_native,native_input,prefetched_input,native_teacher
    scope=read(run/'scope_audit.json')
    prepared=prepare_vae(c,o,t);original_g=prepared['geometry'];stress_frames=scope['max_aligned_frames']
    import numpy as np
    prepared['pixels']=np.concatenate([prepared['pixels'],np.repeat(prepared['pixels'][-1:],max(0,stress_frames-len(prepared['pixels'])),axis=0)])[:stress_frames]
    prepared['geometry']=dict(original_g,frames=stress_frames,duration_s=stress_frames/25,real_duration_s=scope['max_duration_s'],padding_s=stress_frames/25-scope['max_duration_s'])
    count=round(stress_frames/25*prepared['rate']);prepared['wave']=np.pad(prepared['wave'],((0,max(0,count-len(prepared['wave']))),(0,0)))[:count]
    session.vae.encode(prepared)
    result=dict(passed=True,utc=time.time(),samples=[o['sample_id'] for o in selected],bitwise_vae_parity=True,
        native_input_teacher_bitwise_parity=True,stress_frames=stress_frames,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
        peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,seconds=time.time()-started,text_complete=True)
    atomic(run/'CANARY.json',result);return result
