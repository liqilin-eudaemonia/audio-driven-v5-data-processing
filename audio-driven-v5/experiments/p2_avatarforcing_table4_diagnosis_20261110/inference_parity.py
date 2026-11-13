"""One predeclared clip: original preprocessing parity and conditioning audit."""
import importlib.util
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
WORK=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'compare/deployment')]
from p2_avatarforcing_table4_generate import WORK as RUN,REPO,guard,read,save,sha


def main():
    import cv2,numpy as np,soundfile as sf,torch
    from transformers import Wav2Vec2FeatureExtractor
    import vico_avatarforcing as adapter
    device=guard(.15,49152);cv2.setNumThreads(1)
    save(WORK/'INFERENCE_DEVICE.json',device)
    adapter.WORK=WORK
    model=adapter.build_model()
    import types
    model.encode_user_motion=types.MethodType(type(model).encode_user_motion,model)
    policy=read(RUN/'protocol.json');s=min(policy['samples'],key=lambda row:row['duration'])
    folder=RUN/'inputs'/s['id']
    wave,sr=sf.read(s['speaker'],dtype='float32');assert sr==16000
    pp=Wav2Vec2FeatureExtractor.from_pretrained(model.opt.wav2vec_model_path,local_files_only=True)
    user=[]
    for path in sorted((folder/'speaker25').glob('*.jpg')):
        rgb=cv2.cvtColor(cv2.imread(str(path)),cv2.COLOR_BGR2RGB)
        user.append(torch.from_numpy(((rgb.astype(np.float32)-127.5)/127.5).transpose(2,0,1).copy()).unsqueeze(0))
    ref=cv2.cvtColor(cv2.imread(str(folder/'reference.png')),cv2.COLOR_BGR2RGB)
    deployed=dict(avatar_ref=torch.from_numpy(((ref.astype(np.float32)-127.5)/127.5).transpose(2,0,1).copy()).unsqueeze(0),
        avatar_a=pp(np.zeros_like(wave),sampling_rate=sr,return_tensors='pt').input_values,
        user_a=pp(wave,sampling_rate=sr,return_tensors='pt').input_values,user_frame=user)
    spec=importlib.util.spec_from_file_location('official_af_inference',REPO/'inference.py')
    native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native)
    processor=native.DataProcessor(model.opt)
    official=processor.preprocess(avatar_ref_path=s['image'],avatar_audio_path=s['silent'],
        user_audio_path=s['speaker'],user_video_path=str(folder/'speaker25'))
    errors={k:float((deployed[k]-official[k]).abs().max()) for k in ['avatar_ref','avatar_a','user_a']}
    assert len(deployed['user_frame'])==len(official['user_frame'])
    errors['user_frame']=max(float((a-b).abs().max()) for a,b in zip(deployed['user_frame'],official['user_frame']))
    print('PREPROCESS',errors,flush=True)

    def generate(data,a=2,seed=137):
        torch.manual_seed(seed);np.random.seed(seed)
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            return model.inference(dict(data),a_cfg_scale=a,u_cfg_scale=1,nfe=10,seed=seed,use_kv_cache=True)['d_hat'].float().cpu()

    started=time.monotonic();ours=generate(deployed);theirs=generate(official)
    output_error=float((ours-theirs).abs().max())
    print('INFERENCE_MAX_ABS',output_error,flush=True)
    def export(name,value):
        value=value.clamp(-1,1);rgb=((value-value.min())/(value.max()-value.min())*255).byte().permute(0,2,3,1).numpy()
        adapter.export_rgb(WORK/name,[rgb],512,25,18)
    export('official_preprocessor_seed137.mp4',theirs)
    a0=generate(official,a=0);export('counterfactual_avatar_cfg0.mp4',a0)
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        silence=model.audio_encoder.inference(official['avatar_a'].cuda(),seq_len=len(ours))
        style,lam,feats=model.encode_image_into_latent(official['avatar_ref'].cuda())
        direction=model.motion_autoencoder.dec.direction(lam)
        reconstructed,_=model.motion_autoencoder.dec(style+direction,alpha=None,feats=feats)
    rec=reconstructed.float().clamp(-1,1).cpu()
    mae_error=float((rec-official['avatar_ref']).abs().mean())
    image=((rec[0].permute(1,2,0).numpy()+1)*127.5).clip(0,255).round().astype('uint8')
    cv2.imwrite(str(WORK/'reference_autoencoder_reconstruction.png'),cv2.cvtColor(image,cv2.COLOR_RGB2BGR))
    old_video=RUN/'generated'/s['id']/'generated.mp4'
    def decode(path):
        cap=cv2.VideoCapture(str(path));values=[]
        while True:
            okay,frame=cap.read()
            if not okay:break
            values.append(frame)
        cap.release();return np.stack(values)
    old=decode(old_video);new=decode(WORK/'official_preprocessor_seed137.mp4')
    replay_delta=int(np.abs(old.astype('int16')-new.astype('int16')).max())
    info=dict(sample=s['id'],frames=len(ours),preprocess_max_abs=errors,official_vs_adapter_raw_output_max_abs=output_error,
        official_vs_sealed_video_decoded_max_abs=replay_delta,official_vs_sealed_video_sha_equal=sha(old_video)==sha(WORK/'official_preprocessor_seed137.mp4'),
        silence_wave_samples_nonzero=int(torch.count_nonzero(official['avatar_a'])),
        silence_encoded_feature_nonzero=int(torch.count_nonzero(silence)),silence_encoded_feature_shape=list(silence.shape),
        silence_encoded_feature_rms=float(silence.float().square().mean().sqrt()),
        avatar_cfg0_vs2_raw_mean_abs=float((a0-theirs).abs().mean()),
        reference_autoencoder_raw_L1=mae_error,elapsed_s=time.monotonic()-started,
        production_sources={str(path):sha(path) for path in [REPO/'inference.py',ROOT/'compare/deployment/vico_avatarforcing.py',ROOT/'scripts/p2_avatarforcing_table4_generate.py']},
        scientific_scope='one shortest clip; no aggregate claim, no tuning, official listening-only guidance not disclosed',
        gpu=device,script_sha256=sha(__file__))
    save(WORK/'INFERENCE_PARITY.json',info);print(json.dumps(info),flush=True)


if __name__=='__main__':main()
