"""Pretrained stream model training adapters; retain each model's latent basis."""
import ast
import copy
import importlib
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

from .data import ROOT, WORK, read, save, sha
from .features import ids_for, receipt, backend
from .engine import check_features, setup, exercise


def dystream_paths():
    repo = ROOT / 'compare/DyStream'
    sys.path.insert(0, str(repo))
    import model
    vis = repo / 'tools/visualization_0416/utils/model_0506/model'
    if str(vis) not in model.__path__:
        model.__path__.append(str(vis))
    return repo


def export(method, work, ids, device):
    backend(device)
    if method == 'avatarforcing':
        repo = ROOT / 'compare/AvatarForcing'
        sys.path.insert(0, str(repo))
        from models.avatarforcing.generator import Generator
        encoder = Generator(size=512, style_dim=512, motion_dim=20)
        asset = repo / 'pretrained_dir/flow_transformer.pth'
        state = torch.load(asset, map_location='cpu', weights_only=True)
        state = {k.removeprefix('motion_autoencoder.'): v for k, v in state.items() if k.startswith('motion_autoencoder.')}
        encoder.load_state_dict(state, strict=True)
        code = repo / 'models/avatarforcing/encoder.py'
    else:
        repo = dystream_paths()
        from model.head_animation.LIA_3d.motion_encoder import MotionEncoder
        encoder = MotionEncoder(latent_dim=512, size=512)
        asset = repo / 'tools/pretrained_model/epoch=0-step=312000.ckpt'
        state = torch.load(asset, map_location='cpu', weights_only=False)['state_dict']
        state = {k.removeprefix('motion_encoder.'): v for k, v in state.items() if k.startswith('motion_encoder.')}
        encoder.load_state_dict(state, strict=True)
        code = repo / 'tools/visualization_0416/utils/model_0506/model/head_animation/LIA_3d/motion_encoder.py'
    assert state and all(torch.equal(encoder.state_dict()[k], v) for k, v in state.items())
    del state
    encoder.eval().to(device).requires_grad_(False)
    provenance = dict(checkpoint=str(asset), checkpoint_sha256=sha(asset), encoder_sha256=sha(code),
                      strict_all_tensors_loaded=True, native_latent_basis=True, native_fps=25)
    import soundfile as sf
    for sid in ids_for(work, ids):
        dest = Path(work) / 'features' / method / (sid + '.pt')
        if dest.with_suffix('.json').exists():
            check_features(work, method, sid)
            continue
        values = {}
        roles = ['listener', 'speaker'] if method == 'avatarforcing' else ['listener']
        for role in roles:
            frames = np.load(Path(work) / 'media' / sid / (role + '.npy'), mmap_mode='r')
            outputs = []
            with torch.inference_mode():
                for i in range(0, len(frames), 4):
                    x = torch.from_numpy(np.array(frames[i:i+4])).permute(0, 3, 1, 2).to(device).float() / 127.5 - 1
                    y = encoder.dec.direction(encoder.enc.enc_motion(x)) if method == 'avatarforcing' else encoder(x)[0]
                    outputs.append(y.cpu())
            values[role] = torch.cat(outputs)
            assert values[role].shape == (len(frames), 512) and torch.isfinite(values[role]).all()
        wave, sr = sf.read(Path(work) / 'media' / sid / 'speaker.wav', dtype='float32')
        assert sr == 16000
        values['audio'] = torch.from_numpy(wave)
        dest.parent.mkdir(parents=True, exist_ok=True)
        torch.save(values, dest)
        receipt(dest, sid, work, method, dict(extractor=provenance, synthetic_features=False))
        print(method, sid, len(frames), flush=True)


def native_dystream_training(repo):
    """Execute the published _step and loss verbatim without logging/network IO."""
    from diffusers import FlowMatchEulerDiscreteScheduler
    source = repo / 'main.py'
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'MotionGenLightningModule')
    names = ['_step', 'denoising_loss_fn']
    body = [copy.deepcopy(n) for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(body) == 2
    node = ast.ClassDef(name='NativeStep', bases=[], keywords=[], body=body, decorator_list=[])
    namespace = dict(torch=torch, F=F)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(source), 'exec'), namespace)
    runner = namespace['NativeStep']()
    return runner, FlowMatchEulerDiscreteScheduler


class Stream:
    def __init__(self, method, work, device):
        self.method, self.work, self.device = method, Path(work), device
        protocol = read(self.work / 'DATA_PROTOCOL.json')
        self.ids = dict(train=protocol['diagnostic_train'], validation=protocol['diagnostic_validation'])
        self.data = {sid: check_features(work, method, sid) for ids in self.ids.values() for sid in ids}
        self.data_identity = dict(feature_sha256={sid: sha(self.work / 'features' / method / (sid + '.pt')) for sid in self.data},
                                  train_ids=self.ids['train'], validation_ids=self.ids['validation'])
        if method == 'dystream':
            self.build_dystream()
            self.required_gradient_groups = ['audio_encoder_face', 'audio_encoder_face_other', 'blocks', 'diffusion_head']
        else:
            self.build_avatarforcing()
            self.required_gradient_groups = ['audio_encoder', 'flow_transformer']
        self.optimizer = torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad], lr=1e-5,
                                           betas=(.9, .999), weight_decay=.01)

    def build_dystream(self):
        repo = dystream_paths()
        from omegaconf import OmegaConf
        from transformers import Wav2Vec2Model, Wav2Vec2Processor
        module = importlib.import_module('model.motion_generation.motion_gen_gpt_flowmatching_addaudio_linear_twowavencoder')
        original_functional = module.F
        class DeterministicInterpolation:
            def __getattr__(self, name):
                return getattr(original_functional, name)

            def interpolate(self, value, *args, **kwargs):
                # CUDA linear1d backward is nondeterministic. The CPU kernel
                # implements the same linear align_corners interpolation and
                # preserves autograd across copies for both trainable encoders.
                from .ops import linear_audio_interpolate
                return linear_audio_interpolate(value, *args, **kwargs)
        module.F = DeterministicInterpolation()
        cfg = OmegaConf.load(repo / 'configs/motion_gen/sample.yaml')
        # The released app constructs this exact default architecture (768x12),
        # not cfg.hidden_size512/layers4. Strict checkpoint loading verifies it.
        wav = str(ROOT / 'compare/AvatarForcing/pretrained_dir/wav2vec2-base-960h')
        model_loader, processor_loader = Wav2Vec2Model.from_pretrained, Wav2Vec2Processor.from_pretrained
        try:
            Wav2Vec2Model.from_pretrained = classmethod(lambda cls, *a, **kw: model_loader(wav, local_files_only=True))
            Wav2Vec2Processor.from_pretrained = classmethod(lambda cls, *a, **kw: processor_loader(wav, local_files_only=True))
            self.model = module.Audio2FaceGPT(cfg.model)
        finally:
            Wav2Vec2Model.from_pretrained, Wav2Vec2Processor.from_pretrained = model_loader, processor_loader
        asset = repo / 'checkpoints/last.ckpt'
        loaded = torch.load(asset, map_location='cpu', weights_only=False)
        state = {k.removeprefix('model.'): v for k, v in loaded['state_dict'].items()}
        self.model.load_state_dict(state, strict=True)
        assert all(torch.equal(self.model.state_dict()[k], v) for k, v in state.items())
        del state, loaded
        # Native WrapedWav2Vec detaches feature-extractor output itself.
        for enc in [self.model.audio_encoder_face, self.model.audio_encoder_face_other]:
            enc.feature_extractor.requires_grad_(False)
        self.model.to(self.device)
        self.runner, scheduler = native_dystream_training(repo)
        self.runner.model, self.runner.device, self.runner.cfg = self.model, self.device, cfg
        self.runner.threshold = cfg.threshold
        self.runner.train_noise_scheduler = scheduler(**OmegaConf.to_container(cfg.noise_scheduler_kwargs))
        self.runner.val_noise_scheduler = scheduler(**OmegaConf.to_container(cfg.noise_scheduler_kwargs))
        self.clip_grad = .99
        from torch_ema import ExponentialMovingAverage
        self.ema = ExponentialMovingAverage(self.model.parameters(), decay=cfg.model.ema_decay)
        self.info = dict(checkpoint=str(asset), checkpoint_sha256=sha(asset), strict_all_tensors_loaded=True,
                         native_step_sha256=sha(repo / 'main.py'), native_model_sha256=sha(Path(module.__file__)),
                         architecture='released app defaults768x12, strict state equality',
                         objective='verbatim published _step and denoising_loss_fn; predicts clean motion',
                         adaptation='variable real clips <=96frames, no video padding; listener audio explicit silence',
                         optimizer='AdamW lr1e-5, betas.9/.999, weight_decay.01; native sample config lr0 overridden for actual updates; no warmup in diagnostic',
                         deterministic_audio_interpolation='CPU linear align_corners kernel with autograd copies; both acoustic branches remain trainable',
                         anchor_policy='native training anchor sampled from final10 GT frames; training/diagnostic loss only, not inference quality')

    def build_avatarforcing(self):
        import yaml
        repo = ROOT / 'compare/AvatarForcing'
        sys.path.insert(0, str(repo))
        from models.avatarforcing.AvatarForcing import AvatarForcing
        opt = SimpleNamespace(**yaml.safe_load((repo / 'configs/inference.yaml').read_text()))
        opt.rank = self.device
        opt.wav2vec_model_path = str(repo / 'pretrained_dir/wav2vec2-base-960h')
        self.model = AvatarForcing(opt)
        # Both published files have parameter-prefixed keys; identify by keys,
        # not by misleading filenames. Buffers come from the native constructor.
        state = {}
        hashes = {}
        for name in ['flow_transformer.pth', 'motion_autoencoder.pth']:
            path = repo / 'pretrained_dir' / name
            values = torch.load(path, map_location='cpu', weights_only=True)
            assert not set(state) & set(values), 'Conflicting checkpoint tensors'
            state.update(values)
            hashes[name] = sha(path)
        from .training import strict_wav2vec
        acoustic = strict_wav2vec(self.model.audio_encoder.wav2vec2, opt.wav2vec_model_path)
        expected = {k: p for k, p in self.model.state_dict().items() if not k.startswith('audio_encoder.wav2vec2.')}
        assert set(state) == set(expected), (set(expected) - set(state), set(state) - set(expected))
        with torch.no_grad():
            for k, value in state.items():
                expected[k].copy_(value)
        assert all(torch.equal(expected[k].cpu(), v) for k, v in state.items())
        del state
        self.model.set_requires_grad(True)
        self.model.to(self.device)
        self.info = dict(checkpoint_sha256=hashes, strict_all_parameters_loaded=True,
                         acoustic_load=acoustic, acoustic_sha256=sha(Path(opt.wav2vec_model_path) / 'pytorch_model.bin'),
                         source_sha256={str(p.relative_to(repo)): sha(p) for p in (repo / 'models').rglob('*.py')},
                         architecture='native dim_h1024 depth8', objective='local adapter: native scheduler add_noise/target/weight, blockwise independent noise levels, weighted MSE',
                         official_training_loop=False, preference_optimization_tested=False,
                         optimizer='AdamW lr1e-5, diagnostic adaptation',
                         inputs='real speaker audio+motion; listener silence; first-frame listener reference; no future listener condition')

    def train(self):
        self.model.train()
        if self.method == 'avatarforcing':
            self.model.motion_autoencoder.eval()
            self.model.audio_encoder.wav2vec2.eval()

    def after_step(self):
        if self.method == 'dystream':
            self.ema.update()

    def aux_state(self):
        return dict(ema=self.ema.state_dict()) if self.method == 'dystream' else {}

    def load_aux_state(self, value):
        if self.method == 'dystream':
            self.ema.load_state_dict(value['ema'])

    def loss(self, step, validation):
        ids = self.ids['validation' if validation else 'train']
        sid = ids[step % len(ids)]
        data = self.data[sid]
        n = min(96 if self.method == 'dystream' else 50, len(data['listener']))
        if self.method == 'avatarforcing':
            n = n // 10 * 10
        clean = data['listener'][None, :n].to(self.device)
        audio = data['audio'][None, :n * 640].to(self.device)
        if self.method == 'dystream':
            result = self.runner._step(dict(audio=torch.zeros_like(audio), audio_other=audio,
                 motion_latent=clean, style_latent=clean[:, :1], style_latent_other=clean[:, :1]), is_training=not validation)
            loss, pred = result['loss'], result['pred_motion']
        else:
            model = self.model
            with torch.no_grad():
                # Native frozen acoustic feature extractor; projection remains
                # differentiable (inference() would otherwise detach it).
                speaker = model.audio_encoder.get_wav2vec2_feature(audio, n).clone()
                silence = model.audio_encoder.get_wav2vec2_feature(torch.zeros_like(audio), n).clone()
            wa_user = model.audio_encoder.audio_projection(speaker)
            wa = model.audio_encoder.audio_projection(silence)
            motion_user = data['speaker'][None, :n].to(self.device)
            times = model._get_timestep(0, 1000, 1, n, 10)
            noise = torch.randn_like(clean)
            noisy = model.scheduler.add_noise(clean.reshape(-1, 512), noise.reshape(-1, 512), times.reshape(-1)).reshape_as(clean)
            target = model.scheduler.training_target(clean, noise, times)
            pred = model.flow_transformer(times.float(), noisy, wa, wa_user, motion_user, clean[:, 0], train=not validation)
            weight = model.scheduler.training_weight(times)
            loss = ((pred - target).square().mean(-1) * weight).mean()
        assert torch.isfinite(pred).all()
        return loss, dict(sample_id=sid, frames=n, listener_audio='explicit zeros, listening-only target')


def run(method, work=WORK, steps=4, resume=None, device='cuda:0', validate=True, **kwargs):
    assert not kwargs.get('full_pool'), 'Materialize and admit full-pool native features before formal training'
    setup(device)
    adapter = Stream(method, work, device)
    report = exercise(adapter, work, method, 'finetune', steps, validate, resume)
    save(Path(work) / 'runs' / method / 'ACCEPTED.json', report)
