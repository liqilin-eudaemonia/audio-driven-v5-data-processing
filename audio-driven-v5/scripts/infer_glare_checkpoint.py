#!/usr/bin/env python3
"""Generate a listening portrait using a project GLARE checkpoint and native FLOAT."""
import argparse
import gc
import inspect
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace, MethodType

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
import numpy as np
import torch
from baseline_training.training import build, seed
from baseline_training.glare_cached import configure_window, sha
from glare_prosody_source_runtime import official_module


def make_analyzer(model_path, native, contract):
    analyzer = native.Qwen2AudioAnalyzer(Path(model_path), contract['prompt'], 64, 'cuda:0')
    processor = analyzer.processor
    keyword = 'audios' if 'audios' in inspect.signature(processor.__call__).parameters else 'audio'
    class CompatibleProcessor:
        def __call__(self, **kwargs):
            audio = kwargs.pop('audio')
            return processor(**kwargs, **{keyword: audio})
        def __getattr__(self, name): return getattr(processor, name)
    analyzer.processor = CompatibleProcessor()
    return analyzer


def extract_intensity(wave, frames, model_path, analyzer=None, retain=False):
    """Reuse the same official offline analyzer, normalizer and frame encoder as training."""
    protocol = json.loads((ROOT/'experiments/glare_current500h_prosody_subset_20261110/full/protocol.json').read_text())
    contract = protocol['extractor']
    native = official_module(contract['official_script'])
    chunks = native.split_chunks(native.normalize_waveform_loudness(wave, -20., .98), 16000, 2.4)
    if not chunks:
        return torch.zeros(frames, 1), dict(chunks=[], policy='official full 2.4s blocks, tail defaults to zero')
    if analyzer is None:
        analyzer = make_analyzer(model_path, native, contract)
    texts = []
    for start in range(0, len(chunks), 8):
        texts.extend(analyzer.infer_chunks(chunks[start:start+8], 16000, 2.4))
    from float.models.intensity_encoder import IntensityEncoder
    holder = SimpleNamespace(default_fps=25.,
        _time_to_start_frame_1_based=IntensityEncoder._time_to_start_frame_1_based,
        _time_to_end_frame_1_based=IntensityEncoder._time_to_end_frame_1_based)
    result = torch.zeros(frames, 1)
    for i, text in enumerate(texts):
        begin = i*60
        count = min(60, frames-begin)
        if count > 0:
            result[begin:begin+count] = IntensityEncoder.text_to_frame_vector(holder, text, count, fps=25.)
    if not retain:
        del analyzer
        gc.collect()
        torch.cuda.empty_cache()
    return result, dict(chunks=texts, policy='official full 2.4s blocks, tail defaults to zero',
                        normalizer_sha256=sha(contract['official_script']))



def align_empty_cfg_history(model):
    """Official stats-CFG expands x to5 branches but prev_x/prev_wa to4.

    With this model's zero history the tensors have no elements. Correct only
    their batch dimension before invoking the unchanged native FMT forward.
    """
    forward = model.fmt.forward
    def compatible(self, t, x, wa, wr, we, alpha_dict, prev_x, prev_wa, **kwargs):
        if self.num_prev_frames == 0:
            if prev_x.shape[0] != x.shape[0]:
                assert prev_x.shape[1] == 0
                prev_x = prev_x.new_zeros(x.shape[0], 0, prev_x.shape[-1])
            if prev_wa.shape[0] != x.shape[0]:
                assert prev_wa.shape[1] == 0
                prev_wa = prev_wa.new_zeros(x.shape[0], 0, prev_wa.shape[-1])
        return forward(t, x, wa, wr, we, alpha_dict, prev_x, prev_wa, **kwargs)
    model.fmt.forward = MethodType(compatible, model.fmt)

def load_checkpoint_model(checkpoint):
    identity=checkpoint['identity']
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    model, implementation = build('glare', 'cuda:0')
    expected = {name for name, value in model.named_parameters() if value.requires_grad}
    if set(checkpoint['trainable']) != expected:
        raise ValueError('Checkpoint trainable parameter set differs from the native training model')
    if identity['implementation']['assets_sha256'] != implementation['assets_sha256']:
        raise ValueError('Frozen model assets differ from training')
    # Only nontrainable frozen encoders are absent from this state dictionary.
    model.load_state_dict(checkpoint['trainable'], strict=False)
    for name, value in model.named_parameters():
        if name in expected:
            assert torch.equal(value.detach().cpu(), checkpoint['trainable'][name]), name
    model.eval().requires_grad_(False)
    return model


def write_video(image,rgb,padded,output):
    frames=len(rgb)+1
    import soundfile as sf
    import av
    with tempfile.TemporaryDirectory(prefix='glare-video-', dir=output.parent) as folder:
        silent = Path(folder)/'silent.mp4'
        with av.open(str(silent), 'w') as container:
            stream = container.add_stream('libx264', rate=25)
            stream.width = stream.height = 512
            stream.pix_fmt = 'yuv420p'
            for frame in [image, *rgb]:
                for packet in stream.encode(av.VideoFrame.from_ndarray(frame, format='rgb24')):
                    container.mux(packet)
            for packet in stream.encode(): container.mux(packet)
        mux_audio = Path(folder)/'audio.wav'
        sf.write(mux_audio, padded, 16000)
        final = Path(folder)/'result.mp4'
        subprocess.run(['ffmpeg','-v','error','-y','-i',str(silent),'-i',str(mux_audio),
                        '-map','0:v:0','-map','1:a:0','-c:v','copy','-c:a','aac','-t',f'{frames/25:.8f}',str(final)],check=True)
        final.replace(output)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--reference', type=Path, required=True, help='Listener face crop matching training framing')
    p.add_argument('--audio', type=Path, required=True, help='Speaker waveform')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--intensity-npz', type=Path, help='Optional speaker-only precomputed intensity[N,1]')
    p.add_argument('--qwen-path', default='/private/ckpts/Qwen2-Audio-7B-Instruct')
    p.add_argument('--seed', type=int, default=137)
    p.add_argument('--nfe', type=int, default=10)
    p.add_argument('--audio-guidance', type=float, default=2.)
    a = p.parse_args()
    if a.nfe < 2: raise ValueError('ODE requires at least two time points')
    if a.output.exists(): raise FileExistsError(a.output)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    torch.cuda.set_device(0)
    torch.set_num_threads(2)
    seed(a.seed)
    # Snapshot atomically replaced live training checkpoints before loading.
    with tempfile.TemporaryDirectory(prefix='glare-inference-') as folder:
        snapshot = Path(folder)/'checkpoint.pt'
        import shutil
        shutil.copyfile(a.checkpoint, snapshot)
        checkpoint_sha = sha(snapshot)
        checkpoint = torch.load(snapshot, map_location='cpu', weights_only=False)
    import cv2
    import soundfile as sf
    image = cv2.imread(str(a.reference))
    if image is None: raise ValueError('Cannot read reference image')
    image = cv2.cvtColor(cv2.resize(image, (512,512), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    protocol=json.loads((ROOT/'experiments/glare_current500h_prosody_subset_20261110/full/protocol.json').read_text())
    offline=official_module(protocol['extractor']['official_script'])
    wave=offline.load_waveform(a.audio,16000)
    if not len(wave) or not np.isfinite(wave).all(): raise ValueError('Empty/nonfinite speaker waveform')
    frames = max(2, math.ceil(len(wave)/640))
    identity = checkpoint['identity']
    if a.intensity_npz:
        with np.load(a.intensity_npz, allow_pickle=False) as data:
            intensity = torch.from_numpy(data['intensity'].copy()).float()
        prosody = dict(source=str(a.intensity_npz), sha256=sha(a.intensity_npz))
    else:
        # Set up only native module import paths before loading the official analyzer.
        sys.path.insert(0,str(ROOT/'related/glare'))
        intensity, prosody = extract_intensity(wave, frames, a.qwen_path)
    assert intensity.shape == (frames,1) and torch.isfinite(intensity).all()
    model=load_checkpoint_model(checkpoint)
    step=checkpoint['step']
    del checkpoint
    model.opt.nfe = a.nfe
    model.opt.fix_noise_seed = True
    model.opt.seed = a.seed
    configure_window(model, frames, 'cuda:0')
    align_empty_cfg_history(model)
    from transformers import Wav2Vec2FeatureExtractor
    processor = Wav2Vec2FeatureExtractor.from_pretrained(model.opt.wav2vec_model_path, local_files_only=True)
    padded = np.pad(wave, (0, frames*640-len(wave)))
    audio = torch.from_numpy(np.asarray(processor(padded, sampling_rate=16000).input_values[0], np.float32))
    reference = torch.from_numpy(image.copy()).permute(2,0,1).float()[None].cuda()/127.5-1
    started = time.monotonic()
    with torch.inference_mode():
        # Native latent inference reuses the encoded reference and native sample/decoder.
        appearance, motion_lambda, features = model.encode_image_into_latent(reference)
        motion = model.motion_autoencoder.dec.direction(motion_lambda)
        data = dict(s_r=appearance, s_r_feats=features, r_s=motion,
                    ref_r_s=motion[:,None], a=audio[640:][None], intensity=intensity[1:][None])
        generated = model.inference_from_latent(data, a_cfg_scale=a.audio_guidance,
                    nfe=a.nfe, seed=a.seed, alpha_dict=None)
        video = generated['d_hat']
        if video.ndim == 3: video = video[None]
        assert video.shape == (frames-1,3,512,512) and torch.isfinite(video).all()
        rgb = ((video.clamp(-1,1)+1)*127.5).byte().permute(0,2,3,1).cpu().numpy()
    write_video(image,rgb,padded,a.output)
    metadata = dict(checkpoint=str(a.checkpoint),checkpoint_sha256=checkpoint_sha,step=step,
                    reference=str(a.reference),reference_sha256=sha(a.reference),audio=str(a.audio),audio_sha256=sha(a.audio),
                    frames=frames,fps=25,seed=a.seed,nfe=a.nfe,prosody=prosody,
                    native_entry='FLOAT.inference_from_latent -> sample -> decode_latent_into_image',
                    view='first reference frame then N-1 generated frames, no history, integer audio[640:] window',
                    statistics='official learned null conditions, no target statistics',
                    cfg_compatibility='zero-history batch dimension aligned to native five-branch stats CFG',
                    generation_seconds=time.monotonic()-started)
    a.output.with_suffix('.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(metadata),flush=True)


if __name__ == '__main__': main()
