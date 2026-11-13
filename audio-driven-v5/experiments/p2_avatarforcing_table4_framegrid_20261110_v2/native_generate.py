"""Official AF native runtime and source methods, with audited ViCo adaptation."""
import argparse
import importlib.util
import math
import os
from pathlib import Path
import subprocess
import sys
import time

WORK = Path(__file__).resolve().parent
ROOT = WORK.parents[1]
REPO = ROOT / 'compare/AvatarForcing'
sys.path[:0] = [str(REPO), str(ROOT / 'scripts')]
from p2_avatarforcing_table4_generate import read, save, sha


def protocol():
    parent = ROOT / 'experiments/p2_avatarforcing_table4_full_reproduction_20261110'
    previous = read(parent / 'NATIVE_GENERATION_PROTOCOL.json')
    result = dict(previous)
    result['schema'] = 'af-native-original-view-framegrid-v2'
    result['code'] = dict(previous['code'])
    result['code'][str(Path(__file__))] = sha(__file__)
    result['parent_protocol_sha256'] = sha(parent / 'NATIVE_GENERATION_PROTOCOL.json')
    result['reused_generation_manifest_sha256'] = sha(WORK / 'REUSED_GENERATION_MANIFEST.json')
    result['frame_grid_correction'] = dict(
        output_frames='ceil(audio_sample_count/640), exactly as official AvatarForcing.sample.T',
        source_video_frames='Existing ffmpeg25fps rounded sequence; unmodified model replicates final motion latent internally',
        gt_alignment='Nearest timestamp and clamp final available frame when the audio grid includes one additional valid timestamp',
        generation_math_changed=False, two_affected_ids=['yTInFHb_o1Q_000516_000551.1','yTInFHb_o1Q_000516_000551.0'])
    dest = WORK / 'NATIVE_GENERATION_PROTOCOL.json'
    if dest.exists():
        assert read(dest) == result
    else:
        save(dest, result)
    return result


def audit_weights(agent):
    import torch
    model = agent.G
    loaded = {}
    states = {n: torch.load(REPO / 'pretrained_dir' / n, map_location='cpu', weights_only=True)
              for n in ['motion_autoencoder.pth', 'flow_transformer.pth']}
    for name, parameter in model.named_parameters():
        if name.startswith('audio_encoder.wav2vec2.'):
            continue
        source = next((n for n in reversed(list(states)) if name in states[n]), None)
        assert source is not None, name
        assert torch.equal(parameter.detach().cpu(), states[source][name]), name
        loaded[name] = source
    wav = torch.load(REPO / 'pretrained_dir/wav2vec2-base-960h/pytorch_model.bin', map_location='cpu', weights_only=True)
    native = model.audio_encoder.wav2vec2.state_dict()
    checked = []
    for key, value in wav.items():
        if not key.startswith('wav2vec2.'):
            continue
        key = key.removeprefix('wav2vec2.')
        assert key in native, key
        assert torch.equal(native[key].cpu(), value), key
        checked.append(key)
    assert set(native)-set(checked) <= {'masked_spec_embed'}, set(native)-set(checked)
    save(WORK / 'NATIVE_WEIGHTS_AUDIT.json', dict(passed=True, core_parameters=len(loaded),
        parameter_sources=loaded, wav2vec_checked_tensors=len(checked),
        native_weight_g_v_names=True, wav2vec_exact_equality=True,
        unused_training_mask=sorted(set(native)-set(checked))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['probe', 'all'])
    args = parser.parse_args()
    p = protocol()
    for path, digest in {**p['assets'], **p['code']}.items():
        assert sha(path) == digest, path
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '7'
    gpu = subprocess.check_output(['nvidia-smi', '-i', '7', '--query-gpu=uuid,memory.free,memory.total',
                                   '--format=csv,noheader,nounits'], text=True).strip().split(',')
    assert gpu[0].strip() == 'GPU-4d5ee808-e829-c824-c506-cad80ff133f5'
    required = math.ceil(int(gpu[2])*.15)+16384
    assert int(gpu[1]) >= required, (gpu, required)
    import cv2
    import numpy as np
    import torch
    import torchvision
    from omegaconf import OmegaConf
    assert torch.__version__ == '2.0.1+cu118'
    assert torchvision.__version__ == '0.15.2+cu118'
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    torch.cuda.set_per_process_memory_fraction(.15)
    spec = importlib.util.spec_from_file_location('official_af_inference_native', REPO / 'inference.py')
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)
    official.seed_everything(20)
    opt = OmegaConf.load(REPO / 'configs/inference.yaml')
    opt.rank, opt.ngpus = 0, 1
    opt.mae_ckpt_path = str(REPO / 'pretrained_dir/motion_autoencoder.pth')
    opt.ckpt_path = str(REPO / 'pretrained_dir/flow_transformer.pth')
    opt.wav2vec_model_path = str(REPO / 'pretrained_dir/wav2vec2-base-960h')
    official.opt = opt  # Public init_network reads the module-global opt.
    agent = official.InferenceAgent(opt)
    assert agent.G.encode_user_motion.__func__ is type(agent.G).encode_user_motion
    audit_weights(agent)
    cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
    import hashlib
    rng_hash = lambda state: hashlib.sha256(state.cpu().numpy().tobytes()).hexdigest()
    runtime = dict(torch=torch.__version__, torchvision=torchvision.__version__, cuda=torch.version.cuda,
        device=torch.cuda.get_device_name(), benchmark=torch.backends.cudnn.benchmark,
        deterministic=torch.backends.cudnn.deterministic,
        matmul_tf32=torch.backends.cuda.matmul.allow_tf32, cudnn_tf32=torch.backends.cudnn.allow_tf32,
        cpu_rng_after_initialization=rng_hash(cpu_rng), cuda_rng_after_initialization=rng_hash(cuda_rng),
        gpu_guard=dict(free_mib=int(gpu[1]), required_mib=required, allocator_fraction=.15), pid=os.getpid())
    save(WORK / 'NATIVE_GENERATION_RUNTIME.json', runtime)
    samples = p['samples']
    if args.action == 'probe':
        samples = [s for s in samples if s['id'] in ['ClNUxPsUByg_000210_000221.0', 'Ew2z_sYABE0_000063_000083.0']]
    for sample in samples:
        uid = sample['id']
        dest = WORK / 'native_generated' / uid
        dest.mkdir(parents=True, exist_ok=True)
        if (dest / 'complete.json').exists():
            receipt = read(dest / 'complete.json')
            assert receipt['protocol_sha256'] == sha(WORK / 'NATIVE_GENERATION_PROTOCOL.json')
            assert sha(dest / 'generated.mp4') == receipt['sha256']
            continue
        started = time.time()
        inputs = WORK / 'inputs' / uid
        for role in ['speaker', 'listener']:
            receipt = read(inputs / role / 'complete.json')
            for path, digest in receipt['files'].items():
                assert sha(path) == digest, path
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state(cuda_rng)
        data = agent.data_processor.preprocess(str(inputs / 'listener/first_original.png'),
            sample['silent'], sample['speaker'], str(inputs / 'speaker/cropped25'))
        if sample['duration'] > 30:
            for key, path in [('user_a', sample['speaker']), ('avatar_a', sample['silent'])]:
                data[key] = agent.data_processor.default_aud_loader(path).unsqueeze(0)
        condition_frames = round(sample['duration']*25)
        frames = math.ceil(data['avatar_a'].shape[-1] / 640)
        assert len(data['user_frame']) == condition_frames
        assert frames - condition_frames in [0, 1]
        assert data['user_a'].shape == data['avatar_a'].shape == (1, round(sample['duration']*16000))
        assert torch.count_nonzero(data['avatar_a']) == 0
        actual = dict(benchmark=torch.backends.cudnn.benchmark, deterministic=torch.backends.cudnn.deterministic,
                      matmul_tf32=torch.backends.cuda.matmul.allow_tf32, cudnn_tf32=torch.backends.cudnn.allow_tf32)
        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            prediction = agent.G.inference(data, a_cfg_scale=2., u_cfg_scale=1., nfe=10,
                                           seed=25, use_kv_cache=True)['d_hat']
        assert prediction.shape == (frames, 3, 512, 512), (prediction.shape, frames)
        assert torch.isfinite(prediction).all(), 'Non-finite prediction; do not hide with padding or replacement'
        clamped = prediction.detach().clamp(-1, 1)
        lo, hi = float(clamped.min()), float(clamped.max())
        assert hi > lo
        agent.save_video(prediction, str(dest / 'generated.mp4'), sample['silent'])
        cap = cv2.VideoCapture(str(dest / 'generated.mp4'))
        decoded = 0
        while cap.read()[0]:
            decoded += 1
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        assert decoded == frames and abs(fps-25) < 1e-9, (decoded, frames, fps)
        receipt = dict(id=uid, sha256=sha(dest / 'generated.mp4'), frames=frames, fps=fps,
            protocol_sha256=sha(WORK / 'NATIVE_GENERATION_PROTOCOL.json'),
            runtime_sha256=sha(WORK / 'NATIVE_GENERATION_RUNTIME.json'), actual_backend=actual,
            export_range=dict(min=lo, max=hi), seconds=time.time()-started,
            peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
            real_user_visual=True, own_audio_zero=True, target_future_conditioned=False,
            initial_cpu_rng=rng_hash(cpu_rng), initial_cuda_rng=rng_hash(cuda_rng))
        save(dest / 'complete.json', receipt)
        save(WORK / 'NATIVE_GENERATION_STATUS.json', dict(stage='generation', last_id=uid,
            completed=len(list((WORK / 'native_generated').glob('*/complete.json'))), total=43,
            action=args.action, pid=os.getpid()))
        print('native generated', uid, frames, 'frames', round(time.time()-started, 2), 'seconds', flush=True)
        del prediction, clamped, data
        torch.cuda.empty_cache()
    completed = len(list((WORK / 'native_generated').glob('*/complete.json')))
    if args.action == 'probe':
        save(WORK / 'NATIVE_GENERATION_PROBE.json', dict(passed=True, ids=[s['id'] for s in samples]))
    if completed == 43:
        save(WORK / 'NATIVE_GENERATION_COMPLETE.json', dict(passed=True, clips=43,
            protocol_sha256=sha(WORK / 'NATIVE_GENERATION_PROTOCOL.json')))


if __name__ == '__main__':
    main()
