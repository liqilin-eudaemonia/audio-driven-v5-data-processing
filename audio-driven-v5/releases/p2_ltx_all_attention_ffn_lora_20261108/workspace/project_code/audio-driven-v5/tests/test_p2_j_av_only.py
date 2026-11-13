import copy
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from p2_j_av_only import AVOnlyModel, av_objective, commands, DEFAULT, CLUSTER
from p2_matrix16.config import load
from p2_matrix16.model import MatrixModel
from p2_listen.config import load as data_config
from p2_listen.audit import fixture, TinyTextRenderer
from p2_listen.contracts import read
from p2_listen.performance import BucketGradientAccumulator


def build():
    c = load(DEFAULT, formal=True)
    c.pop('connector')  # CPU fixture dimensions; production remains 1024 x 4.
    cfg, sample = fixture(data_config())
    torch.manual_seed(c['training']['seed'])
    model = AVOnlyModel(c, cfg, TinyTextRenderer())
    bundle = dict(condition=sample[0], av=None, text={'positive':{}}, teacher=None)
    def av(u, pack, text):
        v, a = model.renderer(u)
        return (v-.3).square().mean()+(a+.2).square().mean(), {}
    return c, cfg, model, bundle, av


def test_only_av_updates_and_event_branch_is_never_called():
    c, cfg, model, bundle, av = build()
    before = {k:v.clone() for k,v in model.state_dict().items()}
    opt = torch.optim.AdamW(model.groups(), weight_decay=0.)
    with patch.object(model.event_heads, 'forward', side_effect=AssertionError('Event called')):
        values = model(bundle, {}, None, av)
    total, base, event = av_objective(values, c, {}, 1, 1)
    assert torch.equal(total, values['av']) and event.item() == 0
    total.backward()
    acc = BucketGradientAccumulator(model.parameters()); acc.add(); acc.finish()
    assert all(p.grad is None and not p.requires_grad for p in model.event_heads.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.connector.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.text_lora.parameters())
    opt.step()
    after = model.state_dict()
    assert all(torch.equal(v, after[k]) for k,v in before.items() if k.startswith('event_heads.'))
    assert any(not torch.equal(v, after[k]) for k,v in before.items() if k.startswith('connector.'))
    assert any(not torch.equal(v, after[k]) for k,v in before.items() if k.startswith('text_lora.'))
    assert not any(p in opt.state for p in model.event_heads.parameters())


def test_shared_initialization_and_rng_are_identical():
    c, cfg, control, bundle, av = build()
    control_rng = torch.random.get_rng_state()
    reference = copy.deepcopy(c)
    reference.update(id='guided_av_event_s1', loss='av_event')
    reference.pop('event_head_training')
    torch.manual_seed(c['training']['seed'])
    event = MatrixModel(reference, cfg, TinyTextRenderer())
    assert torch.equal(control_rng, torch.random.get_rng_state())
    assert all(torch.equal(v, event.state_dict()[k]) for k,v in control.state_dict().items())


def test_distributed_av_normalization_ignores_event_values():
    c = load(DEFAULT)
    x = torch.tensor(2., requires_grad=True)
    loss, base, event = av_objective({'av':x.square(), 'event_sums':{'unused':float('nan')}}, c, {}, 2, 32)
    loss.backward()
    assert x.grad.item() == 4/64 and event.item() == 0


def test_control_config_and_four_node_commands():
    c = load(DEFAULT, formal=True)
    ref = read(ROOT/'experiments/p2_j_guided_av_event_s1_32gpu_resume200_20261101/run.json')['experiment']
    expected = copy.deepcopy(ref)
    expected.update(id='guided_av_s1',loss='av',event_head_training=False,output=c['output'])
    expected['training']['gradient_checkpointing'] = c['training']['gradient_checkpointing']
    assert c == expected
    rows = commands(read(CLUSTER), DEFAULT)
    assert len(rows) == 4
    for i,row in enumerate(rows):
        assert '--nnodes=4' in row and '--nproc_per_node=8' in row and f'--node_rank={i}' in row
        assert '--resume' not in row
        assert str(ROOT/'scripts/p2_j_av_only.py') in row


def test_rejects_accidental_event_arm():
    c, cfg, model, bundle, av = build()
    c['loss'] = 'av_event'
    with pytest.raises(ValueError, match='event_head_training'):
        av_objective({'av':torch.tensor(1.)}, c, {}, 1, 1)


def test_resume_preserves_frozen_head_and_next_av_update(tmp_path):
    from p2_matrix16 import checkpoint
    from p2_listen.training import rng_state
    c, cfg, model, bundle, av = build()
    opt = torch.optim.AdamW(model.groups(), weight_decay=0.)
    av_objective(model(bundle, {}, None, av), c, {}, 1, 1)[0].backward()
    opt.step(); opt.zero_grad(set_to_none=True)
    identity = {'experiment':c, 'test':'pure_av'}
    path = tmp_path/'step.pt'
    checkpoint.save(path, model, opt, 1, identity, [rng_state()])
    rc, _, restored, rb, rav = build()
    ropt = torch.optim.AdamW(restored.groups(), weight_decay=0.)
    assert checkpoint.resume(path, restored, ropt, identity) == 1
    for m, o, b, callback in [(model,opt,bundle,av), (restored,ropt,rb,rav)]:
        av_objective(m(b, {}, None, callback), c, {}, 1, 1)[0].backward(); o.step()
        assert all(p.grad is None for p in m.event_heads.parameters())
    assert all(torch.equal(v, restored.state_dict()[k]) for k,v in model.state_dict().items())
