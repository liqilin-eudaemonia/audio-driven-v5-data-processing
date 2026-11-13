"""Real tiny LTX: coverage, unchanged init, gradients, frozen base and exact resume."""
import copy
from pathlib import Path
from unittest.mock import patch
import pytest
import torch
from p2_listen.contracts import ROOT, read
from p2_listen.training import rng_state
from p2_ltx_all_lora.lora import extend, ATTENTION, FFN, WHISPER
from p2_ltx_all_lora import checkpoint
from p2_ltx_all_lora.config import expected, validate
from p2_ltx_events.heads import objective
from test_p2_ltx_events import build as full_build, forward
from test_p2_whisper_event_feature import build as base_build
from test_p2_whisper_av import tiny, opened

torch.set_num_threads(2)


def fixture():
    c, old, bundle, prepared, strategy, sampler = full_build(clip_only=True)
    _, cfg, *_ = base_build()
    from p2_ltx_all_lora.model import Model
    renderer, *_ = tiny()
    torch.manual_seed(c['training']['seed'])
    model = Model(c, cfg, renderer)
    # All pre-existing modules retain initialization and original key order.
    for name, value in old.state_dict().items():
        assert torch.equal(value, model.state_dict()[name]), name
    return c, old, model, bundle, prepared, strategy, sampler


def test_complete_native_attention_ffn_coverage_and_full_whisper_ownership():
    _, _, model, *_ = fixture()
    from omni_av_p21.lora import AdaptedLinear
    assert len(model.text_targets) == 28
    assert len(model.text_lora) == 28
    assert {g['name'] for g in model.groups()} == {'connector','events','features','ltx_lora','whisper'}
    for block in model.renderer.transformer_blocks:
        for name in ATTENTION:
            assert all(isinstance(getattr(block,name).get_submodule(p),AdaptedLinear)
                       for p in ('to_q','to_k','to_v','to_out.0'))
        for name in FFN:
            ff = getattr(block,name)
            assert isinstance(ff.net[0].proj,AdaptedLinear) and isinstance(ff.net[2],AdaptedLinear)
        for name in WHISPER:
            branch = getattr(block,name)
            assert all(p.requires_grad for p in branch.parameters())
            assert not any(isinstance(m,AdaptedLinear)for m in branch.modules())
    params = [p for g in model.groups() for p in g['params']]
    assert len(params) == len({id(p) for p in params})
    assert {id(p)for p in params} == {id(p)for p in model.parameters()if p.requires_grad}
    model.assert_ownership()
    with pytest.raises(ValueError):extend(model)


@pytest.mark.parametrize('recompute',[False,True])
def test_zero_delta_keeps_original_full_recipe_forward_exact(recompute):
    c, old, model, bundle, _, s, sa = fixture()
    old.renderer.set_gradient_checkpointing(recompute); model.renderer.set_gradient_checkpointing(recompute)
    old.train();model.train()
    # The native timestep/noise sampler also uses the global Torch RNG.
    torch.manual_seed(333);a = forward(old,bundle,s,sa)
    torch.manual_seed(333);b = forward(model,bundle,s,sa)
    for k in ('av',):torch.testing.assert_close(a[k],b[k],rtol=0,atol=0)
    for k in ('video','audio'):torch.testing.assert_close(a['features'][k],b['features'][k],rtol=0,atol=0)
    for k in a['event_sums']:torch.testing.assert_close(a['event_sums'][k],b['event_sums'][k],rtol=0,atol=0)


@pytest.mark.parametrize('task',['av','event','full'])
def test_real_gradients_reach_every_native_module_family_without_updating_base(task):
    c, _, model, bundle, _, s, sa = fixture();opened(model)
    model.renderer.set_gradient_checkpointing(True);model.train()
    frozen = [(p,p.detach().clone())for p in model.renderer.parameters()if not p.requires_grad]
    v = forward(model,bundle,s,sa)
    if task == 'av':loss = v['av']
    elif task == 'event':loss = sum(v['event_sums'].values())
    else:loss = objective(v,c,{k:r['known'].sum()for k,r in bundle['events'].items()})[0]
    loss.backward()
    for name in (*ATTENTION,*FFN):
        modules = [a for target,a in zip(model.text_targets,model.text_lora,strict=True)
                   if target.split('.')[2] == name]
        assert modules and any(p.grad is not None and p.grad.abs().sum()>0
                               for m in modules for p in m.parameters()),name
    for name in WHISPER:
        assert any(p.grad is not None and p.grad.abs().sum()>0
                   for b in model.renderer.transformer_blocks for p in getattr(b,name).parameters()),name
    torch.optim.AdamW(model.groups(),weight_decay=0.).step()
    assert all(p.grad is None and torch.equal(p,value)for p,value in frozen)
    model.assert_ownership()


def test_optional_attention_gate_is_adapted_and_norms_remain_frozen():
    _, model, *_ = full_build(clip_only=True)
    block = model.renderer.transformer_blocks[0]
    block.attn1.to_gate_logits = torch.nn.Linear(32,2).requires_grad_(False)
    report = extend(model)
    from omni_av_p21.lora import AdaptedLinear
    assert report['projections'] == 29
    assert isinstance(block.attn1.to_gate_logits,AdaptedLinear)
    assert not any(p.requires_grad for p in block.attn1.q_norm.parameters())


def test_full_checkpoint_restores_next_adam_update_exactly(tmp_path):
    c, _, model, bundle, _, s, sa = fixture();opened(model)
    optimizer = torch.optim.AdamW(model.groups(),weight_decay=0.)
    identity = dict(world=1,scope='all',text_targets=model.text_targets)
    def step():
        optimizer.zero_grad(set_to_none=True)
        value = forward(model,bundle,s,sa)
        objective(value,c,{k:r['known'].sum()for k,r in bundle['events'].items()})[0].backward()
        optimizer.step()
    step();path=tmp_path/'step.pt'
    checkpoint.save(path,model,optimizer,1,identity,[rng_state()]);step()
    wanted=copy.deepcopy(model.state_dict());adam=copy.deepcopy(optimizer.state_dict())
    checkpoint.resume(path,model,optimizer,identity);step()
    assert all(torch.equal(v,model.state_dict()[k])for k,v in wanted.items())
    for key,state in adam['state'].items():
        for name,v in state.items():assert torch.equal(v,optimizer.state_dict()['state'][key][name])
    with patch('p2_ltx_all_lora.checkpoint.VERSION','old-text-only-version'):
        with pytest.raises(ValueError):checkpoint.checked(path)


@pytest.mark.parametrize('change',[
    lambda c:c['loss_weights'].update(event=0.),
    lambda c:c['ltx_lora_scope']['attention'].remove('attn1'),
    lambda c:c.update(current_cache_index='/old/data'),
    lambda c:c['training'].update(global_batch=32),
])
def test_prepared_contract_rejects_scope_loss_and_data_drift(change):
    c=expected('/tmp/expanded-lora-test');validate(c);change(c)
    with pytest.raises(ValueError):validate(c)

