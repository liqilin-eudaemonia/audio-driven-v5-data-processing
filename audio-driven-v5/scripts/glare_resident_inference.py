"""Resident official GLARE inference with bounded input caches and decoder batches."""
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import shutil
import tempfile
import time
from types import MethodType

import cv2
import numpy as np
import torch
from infer_glare_checkpoint import (ROOT, sha, official_module, make_analyzer, extract_intensity,
                                    load_checkpoint_model, align_empty_cfg_history, configure_window, write_video)
from baseline_training.training import seed


def batched_decode(model, appearance, features, motion, batch):
    if motion.shape[0]!=1:
        raise ValueError('This interface accepts one reference at a time')
    decoded=[]
    for start in range(0,motion.shape[1],batch):
        target=appearance+motion[0,start:start+batch]
        count=len(target)
        repeated=[f.expand(count,*f.shape[1:]) for f in features]
        value,_=model.motion_autoencoder.dec(target,alpha=None,feats=repeated)
        decoded.append(value)
    return {'d_hat':torch.cat(decoded,dim=0)}


def remember(cache,key,value,capacity):
    cache[key]=value
    cache.move_to_end(key)
    while len(cache)>capacity:cache.popitem(last=False)
    return value


class ResidentGLARE:
    def __init__(self,checkpoint,cache_dir):
        started=time.perf_counter()
        self.path=Path(checkpoint)
        with tempfile.TemporaryDirectory(prefix='glare-weights-') as folder:
            snapshot=Path(folder)/'checkpoint.pt'
            shutil.copyfile(checkpoint,snapshot)
            self.checkpoint_sha=sha(snapshot)
            weights=torch.load(snapshot,map_location='cpu',weights_only=False)
        self.step=weights['step']
        self.model=load_checkpoint_model(weights)
        del weights
        self.protocol=json.loads((ROOT/'experiments/glare_current500h_prosody_subset_20261110/full/protocol.json').read_text())
        self.native=official_module(self.protocol['extractor']['official_script'])
        self.contract=self.protocol['extractor']
        self.analyzer=None
        self.audio_cache=OrderedDict()
        self.reference_cache=OrderedDict()
        self.wave_features=OrderedDict()
        self.cache_dir=Path(cache_dir)
        self.cache_dir.mkdir(parents=True,exist_ok=True)
        self.prosody_id=__import__('hashlib').sha256(json.dumps(dict(contract=self.contract,code=sha(Path(__file__)),extract=sha(ROOT/'scripts/infer_glare_checkpoint.py')),sort_keys=True).encode()).hexdigest()
        self.native_decode=self.model.decode_latent_into_image
        align_empty_cfg_history(self.model)
        from transformers import Wav2Vec2FeatureExtractor
        self.processor=Wav2Vec2FeatureExtractor.from_pretrained(self.model.opt.wav2vec_model_path,local_files_only=True)
        self.current_audio=None
        native_audio=self.model.audio_encoder.inference
        native_emotion=self.model.emotion_encoder.predict_emotion
        def audio_feature(encoder,audio,seq_len):
            key=(self.current_audio,'wav2vec',seq_len)
            if key not in self.wave_features:
                remember(self.wave_features,key,native_audio(audio,seq_len).detach(),16)
            return self.wave_features[key]
        def emotion_feature(encoder,audio):
            key=(self.current_audio,'emotion')
            if key not in self.wave_features:
                remember(self.wave_features,key,native_emotion(audio).detach(),16)
            return self.wave_features[key]
        self.model.audio_encoder.inference=MethodType(audio_feature,self.model.audio_encoder)
        self.model.emotion_encoder.predict_emotion=MethodType(emotion_feature,self.model.emotion_encoder)
        self.load_seconds=time.perf_counter()-started

    def audio(self,path):
        digest=sha(path)
        if digest in self.audio_cache:
            return self.audio_cache[digest],True
        wave=self.native.load_waveform(Path(path),16000)
        if not len(wave) or not np.isfinite(wave).all():raise ValueError('Empty/nonfinite waveform')
        frames=max(2,math.ceil(len(wave)/640))
        key=__import__('hashlib').sha256((digest+self.prosody_id).encode()).hexdigest()
        saved=self.cache_dir/(key+'.npz')
        disk_hit=saved.exists()
        if disk_hit:
            with np.load(saved,allow_pickle=False) as f:
                intensity=torch.from_numpy(f['intensity'].copy())
                metadata=json.loads(str(f['metadata']))
        else:
            if self.analyzer is None:
                self.analyzer=make_analyzer('/private/ckpts/Qwen2-Audio-7B-Instruct',self.native,self.contract)
            intensity,metadata=extract_intensity(wave,frames,'/private/ckpts/Qwen2-Audio-7B-Instruct',self.analyzer,True)
            with tempfile.NamedTemporaryFile(dir=self.cache_dir,suffix='.npz',delete=False) as tmp:
                np.savez_compressed(tmp,intensity=intensity.numpy(),metadata=json.dumps(metadata))
                temp=Path(tmp.name)
            temp.replace(saved)
        assert intensity.shape==(frames,1) and torch.isfinite(intensity).all()
        padded=np.pad(wave,(0,frames*640-len(wave)))
        normalized=torch.from_numpy(np.asarray(self.processor(padded,sampling_rate=16000).input_values[0],np.float32))
        return remember(self.audio_cache,digest,dict(digest=digest,frames=frames,padded=padded,audio=normalized[640:][None],intensity=intensity[1:][None],prosody=metadata),8),disk_hit

    @torch.inference_mode()
    def reference(self,path):
        digest=sha(path)
        if digest in self.reference_cache:return self.reference_cache[digest],True
        image=cv2.imread(str(path))
        if image is None:raise ValueError('Invalid reference image')
        image=cv2.cvtColor(cv2.resize(image,(512,512),interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2RGB)
        tensor=torch.from_numpy(image.copy()).permute(2,0,1).float()[None].cuda()/127.5-1
        appearance,alpha,features=self.model.encode_image_into_latent(tensor)
        motion=self.model.motion_autoencoder.dec.direction(alpha)
        return remember(self.reference_cache,digest,dict(image=image,appearance=appearance,features=features,motion=motion),2),False

    @torch.inference_mode()
    def generate(self,reference,audio,seed_value=137,nfe=10,decode_batch=1):
        if nfe<2 or decode_batch<1:raise ValueError('Require nfe>=2 and positive decode batch')
        started=time.perf_counter()
        value,audio_hit=self.audio(audio)
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        ref,ref_hit=self.reference(reference)
        seed(seed_value)
        model=self.model
        model.opt.nfe=nfe
        model.opt.fix_noise_seed=True
        model.opt.seed=seed_value
        configure_window(model,value['frames'],'cuda:0')
        self.current_audio=value['digest']
        if decode_batch==1:model.decode_latent_into_image=self.native_decode
        else:
            def decode(instance,s_r,s_r_feats,r_d):return batched_decode(instance,s_r,s_r_feats,r_d,decode_batch)
            model.decode_latent_into_image=MethodType(decode,model)
        data=dict(s_r=ref['appearance'],s_r_feats=ref['features'],r_s=ref['motion'],ref_r_s=ref['motion'][:,None],a=value['audio'],intensity=value['intensity'])
        output=model.inference_from_latent(data,a_cfg_scale=2.,nfe=nfe,seed=seed_value,alpha_dict=None)
        video=output['d_hat']
        if video.ndim==3:video=video[None]
        assert len(video)==value['frames']-1 and torch.isfinite(video).all()
        raw=video.cpu()
        rgb=((video.clamp(-1,1)+1)*127.5).byte().permute(0,2,3,1).cpu().numpy()
        return ref['image'],rgb,value['padded'],raw,dict(generation_seconds=time.perf_counter()-started,audio_cache_hit=audio_hit,reference_cache_hit=ref_hit,decode_batch=decode_batch,step=self.step,checkpoint_sha256=self.checkpoint_sha,reference=str(reference),reference_sha256=sha(reference),audio=str(audio),audio_sha256=value['digest'],seed=seed_value,nfe=nfe)

    def run_jobs(self,jobs,decode_batch=1,overlap_export=True):
        outputs=[str(Path(j['output']).resolve()) for j in jobs]
        if len(outputs)!=len(set(outputs)):raise ValueError('Batch output paths must be unique')
        def publish(image,rgb,padded,output,info):
            write_video(image,rgb,padded,output)
            output.with_suffix('.json').write_text(json.dumps(info,indent=2)+'\n')
        results=[]
        with ThreadPoolExecutor(max_workers=1) as writer:
            pending=None
            for job in jobs:
                output=Path(job['output'])
                if output.exists():raise FileExistsError(output)
                output.parent.mkdir(parents=True,exist_ok=True)
                image,rgb,padded,raw,info=self.generate(job['reference'],job['audio'],job.get('seed',137),job.get('nfe',10),decode_batch)
                del raw
                if pending is not None:pending.result()
                info['output']=str(output)
                if overlap_export:pending=writer.submit(publish,image,rgb,padded,output,info)
                else:publish(image,rgb,padded,output,info)
                results.append(info)
            if pending is not None:pending.result()
        return results
