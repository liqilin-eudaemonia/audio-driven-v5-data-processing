"""Isolate cuDNN benchmarking and whole-video min/max export on one outlier."""
import hashlib
from pathlib import Path
import sys
import types

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
WORK=Path(__file__).resolve().parent
RUN=ROOT/'experiments/p2_avatarforcing_table4_recheck_20261110'
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'compare/deployment')]
from p2_avatarforcing_table4_generate import guard,read,save,sha


def main():
    import cv2,soundfile as sf,torch
    from transformers import Wav2Vec2FeatureExtractor
    import vico_avatarforcing as adapter
    guard(.15,49152);cv2.setNumThreads(1);adapter.WORK=WORK/'numerical_probe'
    model=adapter.build_model();model.encode_user_motion=types.MethodType(type(model).encode_user_motion,model)
    uid='UCA1A5GqCdQ_001213_001223.1';sample=next(s for s in read(RUN/'protocol.json')['samples'] if s['id']==uid)
    folder=RUN/'inputs'/uid;wave,sr=sf.read(sample['speaker'],dtype='float32');pp=Wav2Vec2FeatureExtractor.from_pretrained(model.opt.wav2vec_model_path,local_files_only=True)
    def tensor(path):
        a=cv2.cvtColor(cv2.imread(str(path)),cv2.COLOR_BGR2RGB)
        return torch.from_numpy(((a.astype('float32')-127.5)/127.5).transpose(2,0,1).copy()).unsqueeze(0)
    paths=sorted((folder/'speaker25').glob('*.jpg'));n=int(np.ceil(len(wave)/640));paths=(paths+[paths[-1]])[:n]
    data=dict(avatar_ref=tensor(folder/'reference.png'),user_frame=[tensor(path) for path in paths],
        avatar_a=pp(np.zeros_like(wave),sampling_rate=sr,return_tensors='pt').input_values,user_a=pp(wave,sampling_rate=sr,return_tensors='pt').input_values)
    states=[];subsamples=[];summaries=[]
    for benchmark in [False,True]:
        torch.backends.cudnn.benchmark=benchmark;torch.manual_seed(137);np.random.seed(137)
        hashes=[];original=torch.randn
        def captured(*args,**kwargs):
            a=original(*args,**kwargs);hashes.append(hashlib.sha256(a.float().cpu().numpy().tobytes()).hexdigest());return a
        torch.randn=captured
        try:
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                out=model.inference(dict(data),a_cfg_scale=2,u_cfg_scale=1,nfe=10,seed=137,use_kv_cache=True)['d_hat'].float().clamp(-1,1).cpu()
        finally:torch.randn=original
        lo=float(out.min());hi=float(out.max());small=out[:,:,::8,::8].numpy().copy()
        summary=dict(cudnn_benchmark=benchmark,minimum=lo,maximum=hi,mean=float(out.mean()),noise_sha256=hashes)
        summaries.append(summary);subsamples.append(small);states.append(((small-lo)/(hi-lo)*255).clip(0,255))
        first=((out[0].permute(1,2,0).numpy()-lo)/(hi-lo)*255).clip(0,255).astype('uint8')
        cv2.imwrite(str(WORK/f'cudnn_benchmark{benchmark}_first.png'),cv2.cvtColor(first,cv2.COLOR_RGB2BGR))
        print(summary,flush=True);del out;torch.cuda.empty_cache()
    raw=float(np.abs(subsamples[0]-subsamples[1]).mean());rgb=float(np.abs(states[0]-states[1]).mean())
    result=dict(sample=uid,variants=summaries,raw_clamped_subsample_mean_abs=raw,
        fixed_minus1_plus1_mapping_RGB_mean_abs=raw*127.5,minmax_mapping_RGB_mean_abs=rgb,
        same_sampled_noise=summaries[0]['noise_sha256']==summaries[1]['noise_sha256'],
        export_amplification_over_fixed_mapping=rgb/(raw*127.5) if raw else None,
        script_sha256=sha(__file__),scope='one flagged outlier; controlled backend switch, no score fitting')
    save(WORK/'NUMERICAL_PROBE.json',result);print(result,flush=True)


if __name__=='__main__':main()
