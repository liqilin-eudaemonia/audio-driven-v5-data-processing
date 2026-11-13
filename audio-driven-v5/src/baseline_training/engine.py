"""Acceptance runner shared by native two-stage and pretrained stream models."""
from pathlib import Path

import numpy as np
import torch

from .data import ROOT, read, save, sha
from .features import backend
from .training import seed, rng, restore_rng, digest, trainable_state


def local_rng(model):
    return {(path, key): value.get_state() for path, module in model.named_modules()
            for key, value in vars(module).items() if isinstance(value, np.random.RandomState)}


def set_local_rng(model, states):
    modules = dict(model.named_modules())
    assert set(local_rng(model)) == set(states)
    for (path, key), value in states.items():
        getattr(modules[path], key).set_state(value)


def check_features(work, method, sid):
    path = Path(work) / 'features' / method / (sid + '.pt')
    rec = read(path.with_suffix('.json'))
    assert rec['sample_id'] == sid and rec['feature_sha256'] == sha(path), 'Feature changed'
    assert rec['media_receipt_sha256'] == sha(Path(work) / 'media' / sid / 'receipt.json'), 'Media changed'
    source = read(Path(work) / 'media' / sid / 'receipt.json')['contract']
    assert all(rec[k] == source[k] for k in ['caption', 'caption_sha256', 'group_id']), 'Feature metadata changed'
    return torch.load(path, map_location='cpu', weights_only=True)


def setup(device):
    backend(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    seed(137)


def exercise(adapter, work, method, phase, steps=4, validate=True, resume=None):
    """Check real updates, finite gradients, full mutable-state replay, heldout loss."""
    assert steps >= 2
    model, opt = adapter.model, adapter.optimizer
    raw_opt = getattr(opt, '_optimizer', opt)
    work = Path(work)
    out = work / 'runs' / method / phase
    out.mkdir(parents=True, exist_ok=True)
    identity = dict(method=method, phase=phase, protocol_sha256=sha(work / 'DATA_PROTOCOL.json'),
                    runtime=dict(torch=torch.__version__, numpy=np.__version__),
                    sources={p.name: sha(p) for p in Path(__file__).parent.glob('*.py')},
                    implementation=adapter.info, data=adapter.data_identity,
                    trainable_names=[k for k, p in model.named_parameters() if p.requires_grad])
    save(out / 'protocol.json', identity)

    def snapshot(step):
        return dict(identity=identity, model=trainable_state(model),
                    buffers={k: x.detach().cpu().clone() for k, x in model.named_buffers()},
                    optimizer=raw_opt.state_dict(), scheduler={k: getattr(opt, k) for k in
                        ['init_lr', 'd_model', 'n_warmup_steps', 'n_steps'] if hasattr(opt, k)},
                    auxiliary=adapter.aux_state() if hasattr(adapter, 'aux_state') else {},
                    rng=rng(), local_rng=local_rng(model), step=step)

    def restore(value):
        assert value['identity'] == identity, 'Source, model or data identity changed'
        params = dict(model.named_parameters())
        buffers = dict(model.named_buffers())
        assert set(value['model']) == set(identity['trainable_names'])
        with torch.no_grad():
            for k, x in value['model'].items():
                params[k].copy_(x)
            for k, x in value['buffers'].items():
                buffers[k].copy_(x)
        raw_opt.load_state_dict(value['optimizer'])
        for k, x in value['scheduler'].items():
            setattr(opt, k, x)
        if hasattr(adapter, 'load_aux_state'):
            adapter.load_aux_state(value['auxiliary'])
        restore_rng(value['rng'])
        set_local_rng(model, value['local_rng'])

    cursor = 0
    if resume:
        value = torch.load(resume, map_location='cpu', weights_only=False)
        restore(value)
        cursor = value['step']
    history, groups = [], {}

    def update(step):
        adapter.train()
        raw_opt.zero_grad(set_to_none=True)
        loss, metadata = adapter.loss(step, validation=False)
        assert torch.isfinite(loss), 'Nonfinite loss'
        loss.backward()
        norms = {}
        for name, p in model.named_parameters():
            if not p.requires_grad:
                assert p.grad is None, 'Frozen parameter received gradient: ' + name
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), name
                group = name.removeprefix('module.').split('.')[0]
                norms[group] = norms.get(group, 0.) + float(p.grad.detach().float().square().sum())
        assert sum(norms.values()) > 0
        if hasattr(adapter, 'clip_grad'):
            torch.nn.utils.clip_grad_norm_(model.parameters(), adapter.clip_grad)
        if hasattr(opt, 'step_and_update_lr'):
            opt.step_and_update_lr()
        else:
            opt.step()
        if hasattr(adapter, 'after_step'):
            adapter.after_step()
        assert all(torch.isfinite(p).all() for p in model.parameters())
        for k, x in norms.items():
            groups[k] = max(groups.get(k, 0.), x)
        record = dict(step=step + 1, loss=float(loss.detach()), gradient_squared_norm=norms,
                      learning_rate=raw_opt.param_groups[0]['lr'], **metadata)
        history.append(record)
        print(record, flush=True)
        return record

    before = digest(trainable_state(model))
    checkpoint = out / 'replay_checkpoint.pt'
    while cursor < steps:
        if cursor == steps - 1:
            temporary = checkpoint.with_suffix('.tmp')
            torch.save(snapshot(cursor), temporary)
            temporary.replace(checkpoint)
        update(cursor)
        cursor += 1
    assert before != digest(trainable_state(model)), 'No parameter changed'
    for name in getattr(adapter, 'required_gradient_groups', []):
        assert groups.get(name, 0) > 0, 'Detached expected branch: ' + name
    replay = None
    if validate:
        expected = snapshot(cursor)
        expected_digest = digest({k: expected[k] for k in ['model', 'buffers', 'optimizer', 'scheduler', 'auxiliary']})
        expected_loss = history[-1]['loss']
        del expected
        value = torch.load(checkpoint, map_location='cpu', weights_only=False)
        restore(value)
        del value
        actual_loss = update(cursor - 1)['loss']
        actual = snapshot(cursor)
        actual_digest = digest({k: actual[k] for k in ['model', 'buffers', 'optimizer', 'scheduler', 'auxiliary']})
        del actual
        replay = dict(loss_exact=actual_loss == expected_loss,
                      parameters_buffers_optimizer_scheduler_auxiliary_exact=actual_digest == expected_digest)
        assert all(replay.values()), replay
    state, states = rng(), local_rng(model)
    model.eval()
    with torch.no_grad():
        validation_loss, validation_metadata = adapter.loss(0, validation=True)
    restore_rng(state)
    set_local_rng(model, states)
    assert torch.isfinite(validation_loss)
    final = snapshot(cursor)
    final_digest = digest({k: final[k] for k in ['model', 'buffers', 'optimizer', 'scheduler', 'auxiliary']})
    torch.save(final, out / 'checkpoint.pt')
    del final
    result = dict(passed=True, method=method, phase=phase, updates=steps, real_data=True,
                  full_native_model=True, diagnostic_only=True, parameters_changed=True,
                  validation_loss=float(validation_loss), validation=validation_metadata,
                  resume_replay=replay, gradient_groups=groups,
                  checkpoint=str(out / 'checkpoint.pt'), checkpoint_sha256=sha(out / 'checkpoint.pt'),
                  replay_checkpoint=str(checkpoint), final_mutable_state_sha256=final_digest,
                  peak_cuda_gib=torch.cuda.max_memory_allocated() / 1024**3,
                  synthetic_input=False, full500h_convergence_claim=False)
    save(out / 'history.json', history)
    save(out / 'ACCEPTED.json', result)
    print('ACCEPTED', result, flush=True)
    return result
