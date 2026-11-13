"""Source-only Avatar Forcing adaptation for the frozen ViCo test43.

Vendor network is unchanged. No target video/audio is read during generation.
All artifacts live separately from the previously sealed evaluation.
"""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import types

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / 'experiments/p2_vico43_unified_20261107'
REPO = ROOT / 'compare/AvatarForcing'
DY = ROOT / 'experiments/p2_compare_vico43_20261106/dystream'
OLD = ROOT / 'experiments/p2_vico256_combined8000_20261106'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.tmp.{os.getpid()}')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def prepare():
    """Freeze inputs before generation; verify speaker samples against old audit."""
    import numpy as np
    import soundfile as sf
    p = read(DY/'protocol.json')
    ids = [s['id'] for s in p['samples']]
    assert len(ids) == len(set(ids)) == 43
    for s in p['samples']:
        assert sha(s['image']) == s['image_sha256']
        x, rate = sf.read(s['speaker'], dtype='float32')
        z, zr = sf.read(s['silent'], dtype='float32')
        assert rate == zr == 16000 and x.shape == z.shape
        assert np.count_nonzero(z) == 0
        assert hashlib.sha256(x.astype('<f4').tobytes()).hexdigest() == s['speaker_waveform_sha256']
    assets = {str(REPO/'pretrained_dir'/n): sha(REPO/'pretrained_dir'/n)
              for n in ['flow_transformer.pth', 'motion_autoencoder.pth']}
    wav = REPO/'pretrained_dir/wav2vec2-base-960h'
    assets.update({str(f): sha(f) for f in sorted(wav.iterdir()) if f.is_file()})
    assert (wav/'pytorch_model.bin').exists() or (wav/'model.safetensors').exists()
    sources = [Path(__file__), REPO/'configs/inference.yaml', *sorted((REPO/'models').rglob('*.py'))]
    protocol = dict(schema='vico43-avatarforcing-source-only-v1', samples=p['samples'],
        vendor_commit=subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'], text=True).strip(),
        assets=assets, code={str(f): sha(f) for f in sources},
        source_protocol_sha256=sha(DY/'protocol.json'),
        old_results={str(q):sha(q) for q in [OLD/'RESULTS.json', DY/'IMAGE_RESULTS.json',
            DY/'MOTION_RESULTS.json', ROOT/'experiments/common_method_comparison_20261106/vico_deca/RESULTS.json']},
        parameters=dict(seed=137, nfe=10, a_cfg_scale=2., u_cfg_scale=1., use_kv_cache=True),
        input=dict(avatar_ref='identical cropped listener first frame, no second crop',
            avatar_audio='all-zero waveform, passed through official wav2vec preprocessor',
            user_audio='same real speaker waveform', user_visual='zero latent tensor B,T,512; no video read',
            target_future_access=False, target_audio_access=False),
        adaptation=['Override encode_user_motion only: zero conditional latent, as author issue5.',
            'Direct complete-waveform model call; do not apply CLI 30-second truncation.',
            'Official reference Resize/Normalize, with already-cropped common first frame.',
            'Per-clip global minmax after clamp follows official save_video; native raw RGB used for score resize.'],
        native=dict(width=512,height=512,fps=25), scoring=dict(width=256,height=256,fps=30,
            rule='bicubic antialias rawRGB; nearest timestamp round-half-up, clamp tail, no speed change'),
        precision='official bfloat16 autocast; PyTorch2.10 CUDA12.8 compatibility environment; not original Torch2.0',
        sampling='one candidate per UUID seed137; no GT-based selection',
        audio_output='none; all-zero avatar waveform is a condition, not generated audio',
        published_table_reproduction=False,
        sources=['https://github.com/TaekyungKi/AvatarForcing',
                 'https://github.com/TaekyungKi/AvatarForcing/issues/5'])
    target=WORK/'avatarforcing/protocol.json'
    if target.exists():
        assert read(target)==protocol, 'Protocol changed; preserve old run and use a new directory'
    else:
        save(target,protocol)
    save(WORK/'INPUT_AUDIT.json',dict(passed=True,clips=43,source_only=True,
        same_waveform_and_first_frame=True,all_zero_self_audio=True,
        original_result_sha256=protocol['old_results']))
    return protocol


def timestamp_map(frames, native_frames):
    import numpy as np
    if frames <= 0 or native_frames <= 0:
        raise ValueError('Empty video')
    return np.minimum(np.floor(np.arange(frames)*25/30+.5).astype(int),native_frames-1)


def export_rgb(path, chunks, size, fps, crf):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.stem+f'.{os.getpid()}.tmp.mp4')
    proc=subprocess.Popen(['ffmpeg','-nostdin','-v','error','-y','-f','rawvideo','-pix_fmt','rgb24',
        '-s',f'{size}x{size}','-r',str(fps),'-i','pipe:0','-an','-c:v','libx264','-threads','1',
        '-preset','fast','-crf',str(crf),'-pix_fmt','yuv420p','-movflags','+faststart',str(tmp)],stdin=subprocess.PIPE)
    try:
        for x in chunks: proc.stdin.write(x.tobytes())
    finally:
        proc.stdin.close()
    if proc.wait():raise RuntimeError('ffmpeg export failed')
    tmp.replace(path)


def build_model():
    import torch
    from omegaconf import OmegaConf
    sys.path.insert(0,str(REPO))
    from models.avatarforcing.AvatarForcing import AvatarForcing
    opt=OmegaConf.load(REPO/'configs/inference.yaml')
    opt.rank=0;opt.wav2vec_model_path=str(REPO/'pretrained_dir/wav2vec2-base-960h')
    model=AvatarForcing(opt).cuda().eval()
    # transformers4.30 predates torch's parametrized weight_norm naming. Its
    # from_pretrained silently leaves these two tensors random on torch2.10.
    # Explicitly restore every inference-used encoder tensor from the official
    # CTC checkpoint; only the unused training-mask embedding is absent there.
    wav_state=torch.load(REPO/'pretrained_dir/wav2vec2-base-960h/pytorch_model.bin',map_location='cpu',weights_only=True)
    renamed={}
    wav={}
    for k,v in wav_state.items():
        if not k.startswith('wav2vec2.'):continue
        key=k[len('wav2vec2.'):]
        if key.endswith('pos_conv_embed.conv.weight_g'):
            key=key.replace('.weight_g','.parametrizations.weight.original0');renamed[k]=key
        elif key.endswith('pos_conv_embed.conv.weight_v'):
            key=key.replace('.weight_v','.parametrizations.weight.original1');renamed[k]=key
        wav[key]=v
    missing_wav,unexpected_wav=model.audio_encoder.wav2vec2.load_state_dict(wav,strict=False)
    assert set(missing_wav)<= {'masked_spec_embed'} and not unexpected_wav,(missing_wav,unexpected_wav)
    # Compare actual loaded tensors, rather than only trusting missing-key logs.
    actual=model.audio_encoder.wav2vec2.state_dict()
    assert all(torch.equal(actual[k].cpu(),v) for k,v in wav.items())
    assert not model.audio_encoder.wav2vec2.training
    del wav_state,wav,actual
    loaded={};states=[]
    for name in ['motion_autoencoder.pth','flow_transformer.pth']:
        state=torch.load(REPO/'pretrained_dir'/name,map_location='cpu',weights_only=True)
        states.append(state);count=0
        with torch.no_grad():
            for key,param in model.named_parameters():
                if key in state:
                    assert param.shape==state[key].shape, (key,param.shape,state[key].shape)
                    param.copy_(state[key].to(param.device));loaded[key]=name;count+=1
        print('loaded',name,count,flush=True)
    missing=[k for k,_ in model.named_parameters() if k not in loaded and not k.startswith('audio_encoder.wav2vec2.')]
    assert not missing, f'Uninitialized core parameters: {missing}'
    save(WORK/'avatarforcing/WEIGHTS_LOADED.json',dict(passed=True,parameters_from_core=len(loaded),
        all_non_wav2vec_parameters_loaded=True,missing=missing,parameter_sources=loaded,
        wav2vec_exact_tensor_equality=True,wav2vec_renamed=renamed,
        wav2vec_missing_unused_training_mask=missing_wav))
    del states

    # Model expects a list only to determine temporal length. No image/video is encoded.
    def zero_visual(self, frames):
        return torch.zeros(1,len(frames),self.opt.dim_w,device=self.rank,dtype=torch.float32)
    model.encode_user_motion=types.MethodType(zero_visual,model)
    return model


def generate(smoke=False):
    import numpy as np
    import soundfile as sf
    import torch
    import cv2
    from transformers import Wav2Vec2FeatureExtractor
    p=read(WORK/'avatarforcing/protocol.json')
    for f,h in {**p['assets'],**p['code']}.items():assert sha(f)==h,f
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='7', 'Use the explicitly selected physical GPU7'
    torch.set_num_threads(2);cv2.setNumThreads(1)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    # Bound this additional worker's allocator; never evict or stop other jobs.
    torch.cuda.set_per_process_memory_fraction(.10,0)
    model=build_model()
    pp=Wav2Vec2FeatureExtractor.from_pretrained(str(REPO/'pretrained_dir/wav2vec2-base-960h'),local_files_only=True)
    samples=p['samples']
    if smoke:
        samples=[min(samples,key=lambda s:s['duration']),max(samples,key=lambda s:s['duration'])]
    with (WORK/'avatarforcing/generation.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for s in samples:
            folder=WORK/'avatarforcing'/('smoke' if smoke else 'generated')/s['id']
            dest=folder/'complete.json'
            if dest.exists():
                old=read(dest);assert old['protocol_sha256']==sha(WORK/'avatarforcing/protocol.json')
                assert sha(folder/'generated.mp4')==old['native_sha256']
                assert sha(old['score_path'])==old['score_sha256']
                continue
            start=time.time();torch.manual_seed(137);np.random.seed(137)
            assert sha(s['image'])==s['image_sha256']
            wave,rate=sf.read(s['speaker'],dtype='float32');assert rate==16000
            assert hashlib.sha256(wave.astype('<f4').tobytes()).hexdigest()==s['speaker_waveform_sha256']
            user=pp(wave,sampling_rate=16000,return_tensors='pt').input_values
            silent=pp(np.zeros_like(wave),sampling_rate=16000,return_tensors='pt').input_values
            assert torch.count_nonzero(silent)==0
            frame=cv2.cvtColor(cv2.imread(s['image']),cv2.COLOR_BGR2RGB)
            frame=cv2.resize(frame,(512,512),interpolation=cv2.INTER_AREA)
            # Same numerical operation as albumentations Normalize(mean=.5,std=.5).
            ref=torch.from_numpy(((frame.astype(np.float32)-127.5)/127.5).transpose(2,0,1).copy()).unsqueeze(0)
            expected=math.ceil(len(wave)*25/16000)
            data=dict(avatar_ref=ref,avatar_a=silent,user_a=user,user_frame=[None]*expected)
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                output=model.inference(data,a_cfg_scale=2.,u_cfg_scale=1.,nfe=10,seed=137,use_kv_cache=True)['d_hat']
            assert output.shape==(expected,3,512,512),output.shape
            assert torch.isfinite(output).all()
            output=output.float().clamp(-1,1).cpu()
            lo,hi=output.min(),output.max();assert hi>lo
            rgb=((output-lo)/(hi-lo)*255).byte().permute(0,2,3,1).numpy()
            export_rgb(folder/'generated.mp4',[rgb],512,25,18)
            mapping=timestamp_map(s['frames'],len(rgb))
            score=WORK/'avatarforcing'/('smoke_scoring' if smoke else 'scoring')/'avatarforcing'/(s['id']+'.listener.mp4')
            def chunks():
                for i in range(0,len(mapping),32):
                    x=torch.from_numpy(rgb[mapping[i:i+32]].copy()).permute(0,3,1,2).float()/255
                    x=torch.nn.functional.interpolate(x,size=(256,256),mode='bicubic',align_corners=False,antialias=True).clamp(0,1)
                    yield (x.permute(0,2,3,1).numpy()*255).round().astype(np.uint8)
            export_rgb(score,chunks(),256,30,18)
            record=dict(id=s['id'],native_frames=len(rgb),score_frames=s['frames'],score_path=str(score),
                native_sha256=sha(folder/'generated.mp4'),score_sha256=sha(score),
                input_image_sha256=s['image_sha256'],speaker_waveform_sha256=s['speaker_waveform_sha256'],
                source_only=True,listener_gt_audio_read=False,listener_future_video_read=False,
                user_visual_zero=True,mapping=mapping.tolist(),
                max_timestamp_error_s=float(np.abs(mapping/25-np.arange(s['frames'])/30).max()),
                protocol_sha256=sha(WORK/'avatarforcing/protocol.json'),seconds=time.time()-start,
                peak_memory_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
            save(dest,record)
            print('generated',s['id'],len(rgb),round(record['seconds'],2),flush=True)
            del output,rgb,data;torch.cuda.empty_cache()
            done=len(list((WORK/'avatarforcing/generated').glob('*/complete.json')))
            save(WORK/'status.json',dict(stage='avatarforcing_generation',completed=done,total=43,at=time.time()))
    if smoke:save(WORK/'avatarforcing/SMOKE_COMPLETE.json',dict(passed=True,ids=[s['id'] for s in samples]))
    else:
        assert len(list((WORK/'avatarforcing/generated').glob('*/complete.json')))==43
        save(WORK/'avatarforcing/GENERATION_COMPLETE.json',dict(clips=43,at=time.time()))


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('command',choices=['prepare','smoke','generate'])
    args=a.parse_args()
    if args.command=='prepare':prepare()
    else:generate(args.command=='smoke')
