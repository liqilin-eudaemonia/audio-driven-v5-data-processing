import copy
from unittest.mock import patch
import pytest
import torch
from test_p2_whisper_event_feature import build as joint_build
from test_p2_whisper_av import tiny,opened
from p2_matrix16.training import objective
from p2_whisper_ablation.config import load,validate_baseline,Schedule,REFERENCE,source_identity
from p2_whisper_ablation.model import WhisperAblation
from p2_whisper_ablation import checkpoint
from p2_listen.contracts import ROOT,read


def config(arm):return load(ROOT/f'configs/p2_whisper_{arm}_20261106.json')


def build(arm):
    _,cfg,joint,bundle,_,_,_=joint_build()
    c=config(arm);c.pop('connector');c['whisper'].update(input_dim=12,context_dim=16)
    renderer,strategy,sampler,*_=tiny();torch.manual_seed(c['training']['seed'])
    model=WhisperAblation(c,cfg,renderer)
    if arm=='event':bundle['teacher']=None
    else:
        for v in bundle['events'].values():v['known'].zero_();v['weight'].zero_()
    return c,model,joint,bundle,strategy,sampler


@pytest.mark.parametrize('arm',['event','feature'])
def test_same_common_initialization_real_branch_pruning_and_gradient_paths(arm):
    c,model,joint,bundle,strategy,sampler=build(arm)
    assert all(torch.equal(v,joint.state_dict()[k]) for k,v in model.state_dict().items())
    assert set(g['name'] for g in model.groups())=={'connector','text_lora','whisper',arm+'s'}
    assert bool(model.decoders)==(arm=='feature')
    assert all(p.requires_grad==(arm=='event') for p in model.event_heads.parameters())
    opened(model)
    counts={k:1. if arm=='event' else 0. for k in bundle['events']}
    with patch.object(model.event_heads,'forward',wraps=model.event_heads.forward) as head:
        v=model(bundle,{k:1. for k in bundle['events']},torch.Generator().manual_seed(97),strategy,sampler)
        assert head.call_count==int(arm=='event')
        assert bool(v['features'])==(arm=='feature')
        loss,_,ev=objective(v,c,counts,1,1)
        assert torch.isfinite(loss)
        if arm=='feature':assert ev==0
        loss.backward()
        for g in model.groups():assert any(p.grad is not None and p.grad.abs().sum()>0 for p in g['params']),g['name']
    assert all(p.grad is None for p in model.renderer.parameters() if not p.requires_grad)


@pytest.mark.parametrize('arm',['event','feature'])
def test_gradient_recomputation_and_auxiliary_isolation(arm):
    from p2_listen.performance import configure_renderer
    c,model,_,bundle,strategy,sampler=build(arm);opened(model)
    configure_renderer(model.renderer,'all');model.train()
    v=model(bundle,{k:1. for k in bundle['events']},torch.Generator().manual_seed(97),strategy,sampler)
    auxiliary=sum(v['event_sums'].values()) if arm=='event' else sum(v['features'].values())
    downstream=list(model.whisper_adapter.parameters())+list(model.whisper_branches.parameters())+list(model.text_lora.parameters())
    assert all(x is None for x in torch.autograd.grad(auxiliary,downstream,retain_graph=True,allow_unused=True))
    objective(v,c,{k:1. for k in bundle['events']},1,1)[0].backward()
    assert all(any(p.grad is not None and p.grad.abs().sum()>0 for p in g['params']) for g in model.groups())


def test_frozen_reference_exact_schedule_prefix_and_31_ranks():
    reference=read(REFERENCE);event=config('event');feature=config('feature')
    validate_baseline(event,reference);validate_baseline(feature,reference)
    plan=Schedule(event);other=Schedule(feature)
    assert plan.pairs==other.pairs==plan.native.pairs[:8000*31]
    assert len(plan.pairs)==248000
    for step in [1,2,6000,8000]:
        picked=[plan.indices(plan.native.size,step,31,event['training']['seed'],rank,31)[0] for rank in range(31)]
        assert picked==plan.pairs[(step-1)*31:step*31]
    with pytest.raises(ValueError):plan.indices(plan.native.size,8001,31,event['training']['seed'],0,31)


def test_feature_data_does_not_access_event_rows():
    from p2_whisper_ablation.data import Dataset
    dataset=Dataset.__new__(Dataset);dataset.spec={'loss':'av_feature'};dataset.names=['a','b']
    # No LabelReader, owner or SQLite connection exists here.
    values=dataset.event_targets('test','cpu')
    assert all(not v['known'].any() and not v['weight'].any() for v in values.values())


@pytest.mark.parametrize('arm',['event','feature'])
def test_unapproved_changes_and_checkpoint_cross_arm_rejection(arm,tmp_path):
    c=config(arm);reference=read(REFERENCE);c['training']['connector_lr']*=2
    with pytest.raises(ValueError):validate_baseline(c,reference)
    c,model,_,bundle,strategy,sampler=build(arm);optimizer=torch.optim.AdamW(model.groups())
    def update():
        optimizer.zero_grad(set_to_none=True)
        v=model(bundle,{k:1. for k in bundle['events']},torch.Generator().manual_seed(97),strategy,sampler)
        objective(v,c,{k:1. for k in bundle['events']},1,1)[0].backward();optimizer.step()
    update()
    from p2_listen.training import rng_state
    path=tmp_path/'step_000001.pt';identity={'arm':arm}
    checkpoint.save(path,model,optimizer,1,identity,[rng_state()])
    update();expected=copy.deepcopy(model.state_dict())
    assert checkpoint.resume(path,model,optimizer,identity)==1
    update();assert all(torch.equal(v,model.state_dict()[k]) for k,v in expected.items())
    with pytest.raises(ValueError,match='identity'):checkpoint.resume(path,model,optimizer,{'arm':'other'})


def test_historical_sources_stay_identical():
    saved=read(REFERENCE.parent/'run.json')['source'];current=source_identity()
    assert all(current[k]==h for k,h in saved.items())


@pytest.mark.parametrize('arm',['event','feature'])
def test_ablation_inference_has_no_auxiliary_calls_and_preserves_cfg_conditions(arm):
    from contextlib import ExitStack
    from p2_whisper_ablation.inference import sample
    _,model,_,bundle,_,_=build(arm)
    _,_,_,_,prepared,_,_=joint_build()
    opened(model);model.eval();u=model.connector(bundle['condition']);seen=[]
    def hook(module,args,kwargs):seen.append((kwargs['video'],kwargs['audio']))
    handle=model.renderer.register_forward_pre_hook(hook,with_kwargs=True)
    try:
        with ExitStack() as stack:
            stack.enter_context(patch.object(model.event_heads,'forward',side_effect=AssertionError('Event in inference')))
            for decoder in model.decoders.values():
                stack.enter_context(patch.object(decoder,'loss',side_effect=AssertionError('Teacher in inference')))
            out=sample(model,prepared,u,bundle['whisper'],
                dict(video=torch.zeros_like(prepared.video.context),audio=torch.zeros_like(prepared.audio.context),mask=prepared.video.context_mask),
                steps=2,sigmas=torch.tensor([1.,.5,0.]))
    finally:handle.remove()
    assert len(seen)==4
    for video,audio in seen:
        assert video.qwen_context is not None and audio.qwen_context is not None
        assert video.qwen_time_positions is bundle['whisper'].bounds
        assert audio.qwen_time_positions is bundle['whisper'].bounds
        assert torch.equal(video.context[:,-u.video.shape[1]:],u.video)
        assert torch.equal(audio.context[:,-u.audio.shape[1]:],u.audio)
    keep=~prepared.video_loss_mask
    assert torch.equal(out['video'][keep],prepared.video.latent.float()[keep])


def test_launcher_preserves_reserved_gpu_and_old_entrypoint():
    import p2_whisper_event_feature_train31 as old
    import importlib.util
    script=ROOT/'scripts/p2_whisper_ablation.py'
    spec=importlib.util.spec_from_file_location('single_aux_cli',script);cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)
    previous=old.SELF
    for arm in ['event','feature']:
        commands=cli.commands(config(arm),config=ROOT/'custom_config.json')
        assert len(commands)==4 and old.SELF==previous
        assert all(str(script) in cmd and str(ROOT/'custom_config.json') in cmd for cmd in commands)
        assert 'CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6' in commands[-1]
        assert 'CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7' not in commands[-1]
