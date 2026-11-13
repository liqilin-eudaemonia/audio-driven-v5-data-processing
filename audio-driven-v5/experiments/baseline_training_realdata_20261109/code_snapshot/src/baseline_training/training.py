"""Real-data native training, gradients, complete checkpoints and replay test."""
import copy
import hashlib
import importlib
import json
import random
from pathlib import Path
import sys
from types import MethodType

import numpy as np
import torch

from .data import ROOT, WORK, SelectedPool, read, save, sha
from .features import native_glare, backend


def seed(value):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def rng():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(value):
    random.setstate(value['python'])
    np.random.set_state(value['numpy'])
    torch.set_rng_state(value['torch'])
    if value['cuda']:
        torch.cuda.set_rng_state_all(value['cuda'])


def update_digest(value, h):
    if torch.is_tensor(value):
        a = value.detach().cpu().contiguous()
        h.update(str((a.dtype, tuple(a.shape))).encode())
        h.update(a.reshape(-1).view(torch.uint8).numpy().tobytes())
    elif isinstance(value, dict):
        for k in sorted(value, key=str):
            h.update(str(k).encode())
            update_digest(value[k], h)
    elif isinstance(value, (list, tuple)):
        for x in value:
            update_digest(x, h)
    else:
        h.update(repr(value).encode())


def digest(value):
    h = hashlib.sha256()
    update_digest(value, h)
    return h.hexdigest()


def trainable_state(model):
    return {k: p.detach().cpu().clone() for k, p in model.named_parameters() if p.requires_grad}


def strict_wav2vec(model, path):
    state = torch.load(Path(path) / 'pytorch_model.bin', map_location='cpu', weights_only=True)
    values = {}
    expected = model.state_dict()
    for key, value in state.items():
        if not key.startswith('wav2vec2.'):
            continue
        key = key.removeprefix('wav2vec2.')
        if key.endswith('pos_conv_embed.conv.weight_g'):
            key = key.replace('.weight_g', '.parametrizations.weight.original0')
        elif key.endswith('pos_conv_embed.conv.weight_v'):
            key = key.replace('.weight_v', '.parametrizations.weight.original1')
        if key not in expected:
            raise ValueError('Unexpected pretrained acoustic tensor: ' + key)
        values[key] = value
    missing, extra = model.load_state_dict(values, strict=False)
    assert set(missing) <= {'masked_spec_embed'} and not extra, (missing, extra)
    actual = model.state_dict()
    assert all(torch.equal(actual[k].cpu(), v) for k, v in values.items())
    return dict(tensors=len(values), missing_unused=missing, strict_tensor_equality=True)


def build(method, device):
    if method == 'glare':
        opt = native_glare()
        opt.rank = device
        opt.device = device
        opt.wav2vec_sec = 12.0
        from float.models.FLOAT import FLOAT
        model = FLOAT(opt)
        weights = torch.load(opt.lia_path, map_location='cpu', weights_only=True)
        weights = {k.removeprefix('motion_autoencoder.'): v for k, v in weights.items() if k.startswith('motion_autoencoder.')}
        model.motion_autoencoder.load_state_dict(weights, strict=True)
        del weights
        acoustic = strict_wav2vec(model.audio_encoder.wav2vec2, opt.wav2vec_model_path)
        model.emotion_encoder.requires_grad_(False)
        # Upstream decorates this entire method no_grad, although its optimizer
        # explicitly trains audio_projection. Keep frozen W2V, restore projection
        # autograd without changing its numerical forward computation.
        def project(self, audio, seq_len):
            frozen = self.get_wav2vec2_feature(audio, seq_len=seq_len).clone()
            return self.audio_projection(frozen)
        model.audio_encoder.inference = MethodType(project, model.audio_encoder)
        class MaskedReactionLoss(torch.nn.Module):
            def forward(self, prediction, target):
                losses = torch.nn.functional.smooth_l1_loss(prediction, target, reduction='none')
                if self.mask.any():
                    return losses.masked_select(self.mask).mean()
                return losses.sum() * 0
        model.reaction_loss = MaskedReactionLoss()
        model.to(device)
        extra = dict(architecture='native FLOAT; dim_h1024, depth8; original frozen LIA/W2V/SER',
                     assets_sha256={str(path): sha(path) for path in [Path(opt.lia_path),
                         Path(opt.wav2vec_model_path) / 'pytorch_model.bin', Path(opt.audio2emotion_path) / 'pytorch_model.bin']},
                     native_source_sha256={str(p): sha(p) for p in (ROOT / 'related/glare/float').rglob('*.py')},
                     acoustic_load=acoustic,
                     adapter='one first-frame reference; no history at clip start; variable clip length; no temporal padding of video',
                     correction='audio_projection gradient restored; upstream inference() no_grad disabled for this projection only')
        return model, extra
    import yaml
    from box import Box
    repo = ROOT / ('compare/ListenFormer' if method == 'listenformer' else 'compare/vico_challenge_baseline/vico')
    sys.path.insert(0, str(repo))
    module = importlib.import_module('networks.listener_generator')
    config = Box(yaml.safe_load((repo / 'configs/configs.yaml').read_text()))
    cls = module.ListenerGenerator_trans_ca if method == 'listenformer' else module.ListenerGenerator
    model = cls(config).to(device)
    return model, dict(architecture=str(repo), config=dict(config), parameters='full native architecture, no dimension reduction',
                       native_source_sha256={str(p): sha(p) for p in repo.rglob('*.py')})


def feature(method, work, sid):
    kind = 'glare' if method == 'glare' else 'bfm'
    path = Path(work) / 'features' / kind / (sid + '.pt')
    rec = read(path.with_suffix('.json'))
    assert rec['sample_id'] == sid and rec['feature_sha256'] == sha(path), 'Feature changed'
    source = Path(work) / 'media' / sid / 'receipt.json'
    assert rec['media_receipt_sha256'] == sha(source), 'Source contract changed'
    contract = read(source)['contract']
    assert all(rec[k] == contract[k] for k in ['caption', 'caption_sha256', 'group_id']), 'Feature metadata changed'
    return torch.load(path, map_location='cpu', weights_only=True)


def stats_for(method, work, ids):
    if method == 'glare':
        return None
    parts = [feature(method, work, sid) for sid in ids]
    result = {}
    for key in ['audio', 'motion']:
        data = torch.cat([x['audio'] for x in parts] if key == 'audio' else
                         [x[role] for x in parts for role in ['speaker', 'listener']]).float()
        std = data.std(0, correction=1)
        result[key + '_mean'] = data.mean(0)
        result[key + '_std'] = torch.where(std == 0, 1., std)
    return result


def glare_batch(model, value, device):
    from float.models.FMT import get_sinusoid_encoding_table, enc_dec_mask
    from float.utils import compute_alpha_stats
    n = len(value['motion'])
    assert n >= 4
    # The t0 frame is conditioning only. Keep supervision chronological from
    # frame1 onward; append t0 to the native forward's reference slot.
    motion = torch.cat([value['motion'][1:], value['motion'][:1]])
    audio = torch.cat([value['audio'][640:], torch.zeros(640)])
    intensity = torch.cat([value['intensity'][1:], value['intensity'][:1]])
    reaction = torch.cat([value['reaction'][1:], value['reaction'][:1]])
    model.reaction_loss.mask = value['reaction_valid'][1:][None].to(device)
    length = n - 1
    model.num_frames_for_clip = model.fmt.num_frames_for_clip = length
    model.num_total_frames = model.fmt.num_total_frames = n
    model.opt.wav2vec_sec = length / 25
    model.fmt.pos_embed.data = get_sinusoid_encoding_table(n, model.opt.dim_h).unsqueeze(0).to(device)
    mask = enc_dec_mask(n, n, expansion=model.opt.attention_window, num_ref_frames=1).to(device)
    count = model.fmt.num_stat_cond + model.fmt.num_null_cond
    expanded = torch.zeros(n + count, n + count, dtype=torch.bool, device=device)
    expanded[:n, :n] = mask
    model.fmt.alignment_mask = expanded
    alpha = value['motion'][1:] @ value['direction']
    statistics = {k: x[None].to(device) for k, x in compute_alpha_stats(alpha).items()}
    return motion[None].to(device), audio[None].to(device), statistics, intensity[None].to(device), reaction[None].to(device)


def call(model, method, value, stats, device):
    if method == 'glare':
        outputs = model(*glare_batch(model, value, device))
        return outputs['loss'], outputs
    n = min(90, len(value['listener']))
    audio = ((value['audio'][:n] - stats['audio_mean']) / stats['audio_std']).float()[None].to(device)
    speaker = ((value['speaker'][:n] - stats['motion_mean']) / stats['motion_std'])[None].to(device)
    listener = ((value['listener'][:n] - stats['motion_mean']) / stats['motion_std'])[None].to(device)
    lengths = torch.tensor([n], device=device if method == 'listenformer' else 'cpu')
    if method == 'listenformer':
        loss, logs, prediction = model(audio, speaker, listener[:, 0], lengths, 1, 100, listener[:, 1:])
    else:
        # cuDNN maintains an opaque recurrent-dropout RNG cache outside the
        # public torch RNG state. Native ATen LSTM preserves the architecture
        # and dropout distribution and makes checkpoint replay reproducible.
        with torch.backends.cudnn.flags(enabled=False):
            loss, logs, prediction = model(audio, speaker, listener[:, 0], lengths, listener[:, 1:])
    assert torch.isfinite(prediction).all()
    return loss, logs


def set_train(model, method):
    model.train()
    if method == 'glare':
        model.motion_autoencoder.eval()
        model.audio_encoder.wav2vec2.eval()
        model.emotion_encoder.eval()


def run(method, work=WORK, steps=4, resume=None, device='cuda:0', validate=False, full_pool=False, statistics=None):
    assert method in ['glare', 'listenformer', 'vico'], 'Native two-stage method requires its own stage adapter'
    work = Path(work)
    backend(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    seed(137)
    protocol = read(work / 'DATA_PROTOCOL.json')
    assert not (full_pool and validate), 'Use diagnostic group-disjoint IDs for acceptance testing'
    ids = SelectedPool().ids if full_pool else protocol['diagnostic_train']
    validation_ids = [] if full_pool else protocol['diagnostic_validation']
    if not full_pool:
        assert not set(protocol['diagnostic_groups'][sid] for sid in ids) & set(
            protocol['diagnostic_groups'][sid] for sid in validation_ids)
    from functools import lru_cache
    class LazyFeatures:
        @lru_cache(maxsize=4)
        def __getitem__(self, sid):
            return feature(method, work, sid)
    data = LazyFeatures()
    model, implementation = build(method, device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    train_names = [k for k, p in model.named_parameters() if p.requires_grad]
    if full_pool and method != 'glare':
        assert statistics is not None, 'Full native BFM training requires statistics fitted on its training IDs'
        stats = torch.load(statistics, map_location='cpu', weights_only=True)
    else:
        stats = stats_for(method, work, ids)
    out = work / 'runs' / method
    out.mkdir(parents=True, exist_ok=True)
    identity = dict(method=method, protocol_sha256=sha(work / 'DATA_PROTOCOL.json'), scope='full' if full_pool else 'diagnostic',
                    runtime=dict(torch=torch.__version__, numpy=np.__version__),
                    selection_sha256=protocol['selection_sha256'],
                    feature_sha256={sid:sha(work / 'features' / ('glare' if method == 'glare' else 'bfm') / (sid + '.pt')) for sid in ids + validation_ids} if not full_pool else 'checked per sample lazily',
                    source_sha256={str(p):sha(p) for p in Path(__file__).parent.glob('*.py')},
                    trainable_names=train_names, normalization='diagnostic training groups only',
                    full_model=True, implementation=implementation)
    save(out / 'protocol.json', identity)
    cursor = 0
    if resume:
        loaded = torch.load(resume, map_location='cpu', weights_only=False)
        assert loaded['identity'] == identity, 'Resume data, architecture or source changed'
        state = dict(model.named_parameters())
        assert set(loaded['model']) == set(train_names)
        with torch.no_grad():
            for name, value in loaded['model'].items():
                state[name].copy_(value)
        optimizer.load_state_dict(loaded['optimizer'])
        cursor = loaded['step']
        restore_rng(loaded['rng'])
    gradient_groups = {}
    history = []

    def update(step):
        set_train(model, method)
        sid = ids[step % len(ids)]
        optimizer.zero_grad(set_to_none=True)
        loss, logs = call(model, method, data[sid], stats, device)
        assert torch.isfinite(loss), 'Non-finite native loss'
        loss.backward()
        groups = {}
        for name, param in model.named_parameters():
            if not param.requires_grad:
                assert param.grad is None, 'Frozen branch obtained gradient'
            if param.grad is not None:
                assert torch.isfinite(param.grad).all(), name
                group = name.split('.')[0]
                if name.startswith('audio_encoder.audio_projection'):
                    group = 'audio_projection'
                groups[group] = groups.get(group, 0.) + float(param.grad.detach().float().square().sum())
        assert sum(groups.values()) > 0
        optimizer.step()
        for param in model.parameters():
            assert torch.isfinite(param).all()
        for k, v in groups.items():
            gradient_groups[k] = max(gradient_groups.get(k, 0), v)
        value = dict(step=step + 1, sample_id=sid, frames=len(data[sid]['motion' if method == 'glare' else 'listener']),
                     loss=float(loss.detach()), gradient_squared_norm=groups)
        history.append(value)
        print(json.dumps(value), flush=True)
        return value

    checkpoint = out / 'replay_checkpoint.pt'
    before = digest(trainable_state(model))
    while cursor < steps:
        if cursor == steps - 1:
            torch.save(dict(identity=identity, model=trainable_state(model), optimizer=optimizer.state_dict(),
                            rng=rng(), step=cursor, statistics=stats), checkpoint)
        update(cursor)
        cursor += 1
    after = digest(trainable_state(model))
    assert before != after, 'Native trainable weights did not update'
    if method == 'glare' and steps >= 3:
        for name in ['fmt', 'audio_projection', 'intensity_embedder', 'channel_projection', 'reaction_classifier']:
            assert gradient_groups.get(name, 0) > 0, 'Detached training branch: ' + name
    replay = None
    if validate:
        expected = dict(parameters=after, optimizer=digest(optimizer.state_dict()), loss=history[-1]['loss'])
        loaded = torch.load(checkpoint, map_location='cpu', weights_only=False)
        assert loaded['identity'] == identity and set(loaded['model']) == set(train_names)
        state = dict(model.named_parameters())
        with torch.no_grad():
            for name, value in loaded['model'].items():
                state[name].copy_(value)
        optimizer.load_state_dict(loaded['optimizer'])
        restore_rng(loaded['rng'])
        actual = update(loaded['step'])
        replay = dict(loss_exact=actual['loss'] == expected['loss'],
                      all_trainable_parameters_exact=digest(trainable_state(model)) == expected['parameters'],
                      all_adam_states_exact=digest(optimizer.state_dict()) == expected['optimizer'],
                      saved_step=loaded['step'], replayed_step=loaded['step'] + 1)
        assert replay['loss_exact'] and replay['all_trainable_parameters_exact'] and replay['all_adam_states_exact'], replay
    val_loss = None
    if validation_ids:
        state = rng()
        model.eval()
        with torch.no_grad():
            val_loss, _ = call(model, method, data[validation_ids[0]], stats, device)
        restore_rng(state)
        assert torch.isfinite(val_loss)
    final_checkpoint = out / 'checkpoint.pt'
    torch.save(dict(identity=identity, model=trainable_state(model), optimizer=optimizer.state_dict(),
                    rng=rng(), step=cursor, statistics=stats), final_checkpoint)
    result = dict(passed=True, method=method, updates=steps, real_data=True, full_native_model=True,
                  batch_size=1, diagnostic_only=not full_pool, real_samples=ids if not full_pool else len(ids),
                  validation_sample=validation_ids[0] if validation_ids else None,
                  validation_loss=float(val_loss) if val_loss is not None else None,
                  checkpoint=str(final_checkpoint), checkpoint_sha256=sha(final_checkpoint), resume_replay=replay,
                  replay_checkpoint=str(checkpoint), final_parameters_sha256=digest(trainable_state(model)),
                  final_optimizer_sha256=digest(optimizer.state_dict()),
                  gradient_groups=gradient_groups, parameters_changed=True,
                  cuda_peak_gib=torch.cuda.max_memory_allocated() / 1024**3 if device.startswith('cuda') else 0,
                  synthetic_input=False, full500h_convergence_claim=False)
    save(out / 'ACCEPTED.json', result)
    save(out / 'history.json', history)
    print('ACCEPTED', json.dumps(result), flush=True)
