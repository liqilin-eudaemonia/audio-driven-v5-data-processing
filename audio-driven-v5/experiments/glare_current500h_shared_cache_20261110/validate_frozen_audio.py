"""Compare caches to the actual native training encoders and test projection grad."""
import os
from pathlib import Path
import sys
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import glare_frozen_audio_cache as audio
from baseline_training.training import build

assert os.environ.get('CUDA_VISIBLE_DEVICES')=='5'
torch.set_num_threads(2)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
torch.backends.cudnn.benchmark=False
torch.backends.cudnn.deterministic=True
torch.use_deterministic_algorithms(True)
model,_=build('glare','cuda:0')
model.eval()
old=ROOT/'experiments/baseline_training_realdata_20261109'
d=audio.read(old/'DATA_PROTOCOL.json')
rows=[]
for sid in d['diagnostic_train']+d['diagnostic_validation']:
    value=torch.load(old/'features/glare'/(sid+'.pt'),map_location='cpu',weights_only=True)
    cached=torch.load(audio.paths(sid)[0],map_location='cpu',weights_only=True)
    wave=value['audio'][640:][None].cuda()
    n=len(value['motion'])-1
    with torch.no_grad():
        hidden=model.audio_encoder.get_wav2vec2_feature(wave,seq_len=n)[0].cpu()
        emotion=model.emotion_encoder.predict_emotion(wave)[0].cpu()
    torch.testing.assert_close(hidden,cached['wav2vec'],rtol=1e-5,atol=1e-5)
    torch.testing.assert_close(emotion,cached['emotion'],rtol=1e-5,atol=1e-5)
    rows.append(dict(sample_id=sid,shape=list(hidden.shape),
                wav2vec_max_error=float((hidden-cached['wav2vec']).abs().max()),
                emotion_max_error=float((emotion-cached['emotion']).abs().max())))
    assert not cached['wav2vec'].requires_grad and not torch.is_inference(cached['wav2vec'])
    model.audio_encoder.audio_projection.zero_grad(set_to_none=True)
    model.audio_encoder.audio_projection(cached['wav2vec'][None].cuda()).square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.audio_encoder.audio_projection.parameters())
    assert sum(float(p.grad.square().sum()) for p in model.audio_encoder.audio_projection.parameters())>0
audio.save(audio.WORK/'CANARY_ACCEPTED.json',dict(passed=True,samples=rows,
    native_training_encoder_parity=True,trainable_projection_gradient_preserved=True,
    view='normalized whole-pair audio[640:], N-1 frames; no history'))
print(rows,flush=True)
