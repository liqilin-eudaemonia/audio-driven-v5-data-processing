"""Combined objective on native tiny LTX, with no GPU/pretrained downloads."""
import copy
from dataclasses import replace
from unittest.mock import patch

import pytest
import torch

from test_p2_whisper_av import tiny,features,opened
from p2_listen.audit import fixture
from p2_listen.config import load as data_config
from p2_listen.contracts import read
from p2_listen.training import rng_state
from p2_matrix16.model import MatrixModel
from p2_matrix16.training import objective
from p2_whisper_av.model import QwenWhisperAV
from p2_whisper_event_feature.config import load,validate,validate_baseline,pure_av_view
from p2_whisper_event_feature.model import QwenWhisperEventFeature
from p2_whisper_event_feature import checkpoint


def build():
    from omni_av.ltx_runtime import with_text
    c=load();c.pop('connector');c['whisper'].update(input_dim=12,context_dim=16)
    cfg,f=fixture(data_config());cfg=replace(cfg,video_dim=32,audio_dim=32)
    renderer,strategy,sampler,av,text=tiny()
    torch.manual_seed(c['training']['seed']);model=QwenWhisperEventFeature(c,cfg,renderer)
    prepared=strategy.prepare_training_inputs(with_text(av,text),sampler)
    names=tuple(c['event_supervision']['classes']['video']+c['event_supervision']['classes']['audio'])
    labels={k:dict(y=torch.tensor([float(i%2)]),known=torch.tensor([True]),weight=torch.tensor([1.]))
            for i,k in enumerate(names)}
    bundle=dict(condition=f[0],layout=f[1],teacher=f[2],events=labels,text={'positive':text},
        whisper=features(),av=dict(av=av,support=dict(video=torch.ones_like(prepared.video_loss_mask),
        audio=torch.ones_like(prepared.audio_loss_mask))))
    return c,cfg,model,bundle,prepared,strategy,sampler


def forward(model,bundle,strategy,sampler):
    return model(bundle,{k:1. for k in bundle['events']},torch.Generator().manual_seed(97),strategy,sampler)


@pytest.mark.parametrize('change',[
    lambda c:c.update(teacher_cache=None),lambda c:c.update(event_head_training=False),
    lambda c:c['loss_weights'].update(event=.1),lambda c:c['whisper'].update(enable_audio=False),
    lambda c:c.update(qwen_hidden_source='sft'),lambda c:c.update(whisper_execution_migration={}),
    lambda c:c['training'].update(save_every=100),
])
def test_rejects_incompatible_combination(change):
    c=load();change(c)
    with pytest.raises(ValueError):validate(c)


def test_baseline_conditions_and_duration_order_are_locked():
    c=load();b=read(c['baseline_config']);validate_baseline(c,b)
    c['duration_sampling']['start_step']+=1
    with pytest.raises(ValueError,match='baseline'):validate_baseline(c,b)


def test_shared_initialization_and_zero_residual():
    from omni_av_p2.ltx import append_conditions
    c,cfg,model,bundle,prepared,_,_=build()
    renderer,*_=tiny();torch.manual_seed(c['training']['seed'])
    base=QwenWhisperAV(pure_av_view(c),cfg,renderer)
    for name,value in base.state_dict().items():
        assert torch.equal(value,model.state_dict()[name]),name
    renderer,*_=tiny();torch.manual_seed(c['training']['seed'])
    combined=MatrixModel(c,cfg,renderer)
    for name,value in combined.state_dict().items():
        assert torch.equal(value,model.state_dict()[name]),name
    u=model.connector(bundle['condition']);a=append_conditions(prepared,u)
    b=model.prepare(prepared,u,bundle['whisper'])
    x=model.renderer(video=a.video,audio=a.audio,perturbations=None)
    y=model.renderer(video=b.video,audio=b.audio,perturbations=None)
    assert all(torch.equal(v,w) for v,w in zip(x,y))


def test_all_five_optimizer_groups_receive_gradients_and_backbones_stay_frozen():
    c,_,model,bundle,_,strategy,sampler=build()
    frozen=[(p,p.detach().clone()) for p in model.renderer.parameters() if not p.requires_grad]
    optimizer=torch.optim.AdamW(model.groups(),weight_decay=0.)
    seen=set()
    counts={k:1 for k in bundle['events']}
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        values=forward(model,bundle,strategy,sampler)
        assert set(values['features'])=={'video','audio'}
        loss,_,_=objective(values,c,counts,1,1);assert torch.isfinite(loss)
        loss.backward()
        for group in optimizer.param_groups:
            if any(p.grad is not None and p.grad.abs().sum()>0 for p in group['params']):seen.add(group['name'])
        optimizer.step();model.assert_ownership()
    assert seen=={'connector','events','features','text_lora','whisper'}
    assert all(p.grad is None and torch.equal(p,v) for p,v in frozen)
    assert not bundle['condition'].hidden.requires_grad
    assert all(not bundle['teacher'][m].requires_grad for m in ['video','audio'])
    assert not any(k.startswith('renderer.') for k in model.state_dict())


def test_auxiliary_gradients_stop_at_u_and_av_trains_whisper():
    _,_,model,bundle,_,strategy,sampler=build();opened(model)
    values=forward(model,bundle,strategy,sampler)
    auxiliary=sum(values['features'].values())+sum(values['event_sums'].values())
    downstream=list(model.text_lora.parameters())+list(model.whisper_adapter.parameters())+list(model.whisper_branches.parameters())
    grads=torch.autograd.grad(auxiliary,downstream,retain_graph=True,allow_unused=True)
    assert all(g is None for g in grads)
    values['av'].backward()
    for module in [model.connector,model.text_lora,model.whisper_adapter,model.whisper_branches]:
        assert any(p.grad is not None and p.grad.abs().sum()>0 for p in module.parameters())


def test_transfer_moves_both_dataclasses_and_nested_targets():
    from p2_whisper_event_feature.data import to_device
    _,_,_,bundle,_,_,_=build();moved=to_device(bundle,'meta')
    for value in [moved['condition'].hidden,moved['condition'].video_coordinates,
                  moved['whisper'].hidden,moved['whisper'].bounds,moved['teacher']['video'],
                  moved['events'][next(iter(moved['events']))]['known'],moved['text']['positive']['video'],
                  moved['av']['av']['latents']['latents']]:
        assert value.device.type=='meta'
    assert bundle['condition'].hidden.device.type=='cpu' and bundle['whisper'].hidden.device.type=='cpu'


@pytest.mark.parametrize('depth',[0,1,2,4])
def test_prefetch_keeps_consumed_order_cpu_device_and_rng(depth):
    from types import SimpleNamespace
    from p2_whisper_event_feature.execution import CPUInputStream
    before=torch.get_rng_state().clone();seen=[]
    def sample(index,device):
        assert device=='cpu';seen.append(index);return {'index':index,'tensor':torch.tensor(index)}
    requests=[(s,[(s*2,31+s),(s*2+1,49+s)]) for s in range(5)]
    with CPUInputStream(SimpleNamespace(sample=sample),requests,depth) as stream:
        rows=list(stream)
    assert [r['step'] for r in rows]==list(range(5))
    assert [x['index'] for r in rows for x in r['samples']]==list(range(10))
    assert [r['chosen'] for r in rows]==[r[1] for r in requests]
    assert torch.equal(before,torch.get_rng_state()) and seen==list(range(10))


def test_prefetch_propagates_cache_errors_without_skipping_samples():
    from types import SimpleNamespace
    from p2_whisper_event_feature.execution import CPUInputStream
    def sample(index,device):raise ValueError('bad cache')
    with CPUInputStream(SimpleNamespace(sample=sample),[(1,[(0,0)])],2) as stream:
        with pytest.raises(ValueError,match='bad cache'):next(stream)
    assert stream.closed


@pytest.mark.parametrize('execution',[{'prefetch_batches':5},{'prefetch_batches':True},{'gradient_bucket_mb':0}])
def test_invalid_combined_execution_rejected(execution):
    c=load();c['combined_execution']=execution
    with pytest.raises(ValueError):validate(c)


def test_full_recomputation_keeps_all_loss_paths():
    from p2_listen.performance import configure_renderer
    c,_,model,bundle,_,strategy,sampler=build();opened(model)
    configure_renderer(model.renderer,'all');model.train()
    values=forward(model,bundle,strategy,sampler)
    objective(values,c,{k:1 for k in bundle['events']},1,1)[0].backward()
    for group in model.groups():
        assert any(p.grad is not None and p.grad.abs().sum()>0 for p in group['params']),group['name']


def test_global_objective_uses_known_counts_once():
    c=load();counts={'a':3.,'b':1.}
    # Two ranks, two microbatches each; auxiliary counts intentionally unequal.
    values=[dict(av=torch.tensor(float(i+1)),features=dict(video=torch.tensor(float(i+2)),audio=torch.tensor(float(i+3))),
                 event_sums={'a':torch.tensor(float(i<3)*(i+1)),'b':torch.tensor(float(i==3)*4)}) for i in range(4)]
    actual=sum(objective(v,c,counts,2,2)[0] for v in values)
    expected=sum(v['av']+.025*(v['features']['video']+v['features']['audio']) for v in values)/4+.05*((1+2+3)/3+4)/2
    torch.testing.assert_close(actual,expected)


def test_longest_sample_canary_restores_parameters_optimizer_and_rng(tmp_path,monkeypatch):
    from contextlib import nullcontext
    from types import SimpleNamespace
    from p2_whisper_event_feature.training import longest_check
    from p2_whisper_av.execution import settings
    c,_,model,bundle,_,strategy,sampler=build();model.train()
    c['j_cache']=str(tmp_path)
    (tmp_path/'sample_list.csv').write_text('sample_id,duration_s\nshort,2\nlong,12\n')
    model_state=copy.deepcopy(model.state_dict());optimizer=torch.optim.AdamW(model.groups(),weight_decay=0.)
    before_rng=torch.get_rng_state().clone();calls=[]
    def sample(index,device):
        calls.append(index);return copy.deepcopy(bundle)
    fork=torch.random.fork_rng
    monkeypatch.setattr(torch.random,'fork_rng',lambda devices=None,**kw:fork(devices=[],**kw))
    monkeypatch.setattr(torch.cuda,'reset_peak_memory_stats',lambda device:None)
    monkeypatch.setattr(torch.cuda,'max_memory_reserved',lambda device:2**30)
    monkeypatch.setattr(torch.cuda,'empty_cache',lambda:None)
    monkeypatch.setattr(torch,'autocast',lambda *a,**kw:nullcontext())
    names=list(bundle['events'])
    result=longest_check(model,SimpleNamespace(sample=sample),c,optimizer,settings(c),strategy,sampler,
                         {k:1. for k in names},names,names,'cpu',0,0,1)
    assert result['updates_restored'] and result['rng_restored'] and calls==[1,1]
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in model_state.items())
    assert not optimizer.state and torch.equal(before_rng,torch.get_rng_state())


def test_complete_resume_reproduces_next_update_and_rejects_pure_av(tmp_path):
    c,_,model,bundle,_,strategy,sampler=build();optimizer=torch.optim.AdamW(model.groups(),weight_decay=0.)
    def update():
        optimizer.zero_grad(set_to_none=True)
        objective(forward(model,bundle,strategy,sampler),c,{k:1 for k in bundle['events']},1,1)[0].backward()
        optimizer.step()
    update();identity={'test':'combined','world':1};path=tmp_path/'step_000001.pt'
    checkpoint.save(path,model,optimizer,1,identity,[rng_state()])
    update();expected=copy.deepcopy(model.state_dict())
    assert checkpoint.resume(path,model,optimizer,identity)==1
    update();assert all(torch.equal(v,model.state_dict()[k]) for k,v in expected.items())
    with pytest.raises(ValueError,match='identity'):checkpoint.resume(path,model,optimizer,dict(identity,world=2))
    from p2_whisper_av.checkpoint import save as old_save
    old=tmp_path/'pure.pt';old_save(old,model,optimizer,1,identity,[rng_state()])
    with pytest.raises(ValueError,match='Not a complete'):checkpoint.checked(old)


def test_inference_skips_teacher_and_events_and_keeps_whisper_in_cfg():
    from p2_whisper_event_feature.inference import sample
    _,_,model,bundle,prepared,_,_=build();opened(model);u=model.connector(bundle['condition']);seen=[]
    def hook(module,args,kwargs):seen.append((kwargs['video'],kwargs['audio']))
    handle=model.renderer.register_forward_pre_hook(hook,with_kwargs=True)
    try:
        with patch.object(model.event_heads,'forward',side_effect=AssertionError('Event called')),patch.object(model.decoders['video'],'loss',side_effect=AssertionError('Teacher called')):
            out=sample(model,prepared,u,bundle['whisper'],dict(video=torch.zeros_like(prepared.video.context),
                audio=torch.zeros_like(prepared.audio.context),mask=prepared.video.context_mask),
                steps=2,sigmas=torch.tensor([1.,.5,0.]))
    finally:handle.remove()
    assert len(seen)==4
    for video,audio in seen:
        assert video.qwen_context is not None and audio.qwen_context is not None
        assert video.qwen_time_positions is bundle['whisper'].bounds
        assert torch.equal(video.context[:,-u.video.shape[1]:],u.video)
    keep=~prepared.video_loss_mask
    assert torch.equal(out['video'][keep],prepared.video.latent.float()[keep])
