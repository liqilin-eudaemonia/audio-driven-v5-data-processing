import copy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from p2_matrix16.config import emit,load,validate,feature_enabled,event_enabled,source_identity
from p2_matrix16.model import MatrixModel
from p2_matrix16.training import objective,gradient_audit
from p2_matrix16 import checkpoint


@pytest.mark.parametrize('cadence', [100, 200])
def test_checkpoint_interval_accepts_configured_positive_integer(cadence):
    from p2_listen.contracts import ROOT, read
    from p2_matrix16.config import validate
    c = read(ROOT/'configs/p2_j_clip_events_20261101/guided_av_event_s1.json')
    c['training']['save_every'] = cadence
    assert validate(c,formal=True)['training']['save_every'] == cadence


@pytest.mark.parametrize('cadence', [0, -1, True, 2.5])
def test_checkpoint_interval_rejects_invalid_values(cadence):
    from p2_listen.contracts import ROOT, read
    from p2_matrix16.config import validate
    c = read(ROOT/'configs/p2_j_clip_events_20261101/guided_av_event_s1.json')
    c['training']['save_every'] = cadence
    with pytest.raises(ValueError,match='common training'):
        validate(c,formal=True)
from p2_listen.config import load as old_config
from p2_listen.audit import fixture,TinyTextRenderer
from p2_listen.contracts import EVENTS,ROOT
from p2_listen.training import rng_state

torch.set_num_threads(2)

@pytest.fixture(scope='module')
def configs(tmp_path_factory):
    root=tmp_path_factory.mktemp('matrix');rows=emit(root)
    return {r['id']:load(root/r['config']) for r in rows}

def native_qwen(cfg):
    from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import Qwen2_5OmniThinkerTextModel
    from transformers.models.qwen2_5_omni.configuration_qwen2_5_omni import Qwen2_5OmniTextConfig
    tc=Qwen2_5OmniTextConfig(vocab_size=32,hidden_size=8,intermediate_size=16,num_hidden_layers=3,
        num_attention_heads=2,num_key_value_heads=2,rope_scaling={'rope_type':'default','mrope_section':[1,1,0]})
    tc._attn_implementation='sdpa'
    model=torch.nn.Module();model.model=Qwen2_5OmniThinkerTextModel(tc)
    model.config=SimpleNamespace(text_config=tc)
    model.get_input_embeddings=lambda:model.model.embed_tokens
    tok=SimpleNamespace(encode=lambda *a,**kw:[4,5,6],all_special_ids=[2],unk_token_id=0,
                        convert_tokens_to_ids=lambda s:2)
    return SimpleNamespace(thinker=model,config=cfg),SimpleNamespace(tokenizer=tok)

def setup(c):
    torch.manual_seed(37)
    cfg,s=fixture(old_config());runtime,processor=native_qwen(cfg)
    model=MatrixModel(c,cfg,TinyTextRenderer(),runtime if c['stage']==2 else None,processor)
    # Represent a nondegenerate trained Stage1 parent, then test upstream paths.
    with torch.no_grad():
        for name,p in model.connector.named_parameters():
            if p.numel()>1 and not p.any():p.normal_(std=.02)
        for p in model.decoders.parameters():
            if p.numel()>1 and not p.any():p.normal_(std=.02)
    bundle=dict(condition=s[0] if c['stage']==1 else None,layout=s[1],
        teacher=s[2] if feature_enabled(c) else None,av=None,text={'positive':{'prompt':'same plan'}},events=s[4],
        observation={'requested_duration_s':1.},plan='same plan' if c['text_condition']=='guided' else None)
    prefix=dict(prefix=torch.randn(1,3,8),prefix_valid=torch.ones(1,3,dtype=torch.bool),
                prefix_positions=torch.arange(3)[None,None].expand(3,1,-1))
    def av(u,pack,text):
        a,b=model.renderer(u)
        return (a-.3).square().mean()+(b+.2).square().mean(),{}
    return model,bundle,prefix,av

def forward(model,bundle,prefix,av):
    with patch('p3_listen.features.prefix',return_value=prefix),patch('p2_listen.features.make_observation',side_effect=lambda x:x),patch('p2_listen.features.image_grid',return_value=(None,None,(1,1))):
        return model(bundle,{k:1. for k in EVENTS},torch.Generator().manual_seed(19),av)

def test_matrix_is_exact_and_budget_explicit(configs):
    assert len(configs)==16
    assert len({(c['text_condition'],c['loss'],c['stage']) for c in configs.values()})==16
    for c in configs.values():
        with pytest.raises(ValueError,match='budget'):validate(c,formal=True)
    files=source_identity()
    assert any(p.endswith('omni_av_p21/lora.py') for p in files)

@pytest.mark.parametrize('text',['fixed','guided'])
@pytest.mark.parametrize('loss',['av','av_event','av_feature','av_event_feature'])
@pytest.mark.parametrize('stage',[1,2])
def test_all_gradient_paths_and_frozen_weights(configs,text,loss,stage):
    c=configs[f'{text}_{loss}_s{stage}'];model,b,pref,av=setup(c);model.train()
    frozen=[(p,p.detach().clone()) for owner in [model.renderer]+([model.qwen_runtime.thinker] if stage==2 else []) for p in owner.parameters() if not p.requires_grad]
    r=forward(model,b,pref,av);audit=gradient_audit(model,r,c,r['event_counts'])
    assert audit['connector']['av']>0 and audit['text_lora']['av']>0
    assert (audit['connector']['event']>0)==event_enabled(c)
    assert bool(model.decoders)==feature_enabled(c)
    if stage==2:
        assert audit['qwen_lora']['av']>0
        assert (audit['qwen_lora']['event']>0)==event_enabled(c)
        if feature_enabled(c):assert audit['qwen_lora']['feature']>0
        assert len(model.qwen_targets)==8  # 2 selected upstream blocks, 4 targets each.
        assert model.qwen_runtime.thinker.model.gradient_checkpointing
    if feature_enabled(c):assert audit['connector']['feature']>0
    optimizer=torch.optim.AdamW(model.groups(),weight_decay=0)
    total,_,_=objective(r,c,r['event_counts'],1,1);total.backward();optimizer.step()
    assert all(p.grad is None and torch.equal(p,old) for p,old in frozen)
    assert not any('.base.' in k or 'thinker' in k or 'renderer' in k for k in model.state_dict())
    model.assert_ownership()

def test_stage2_rejects_cached_hidden(configs):
    m,b,p,a=setup(configs['fixed_av_s2']);b['condition']=object()
    with pytest.raises(ValueError,match='forbids cached'):forward(m,b,p,a)

def test_online_condition_preserves_native_outputs_at_zero_lora(configs):
    from p2_matrix16.qwen import inject_qwen_lora
    cfg,_=fixture(old_config());runtime,_=native_qwen(cfg)
    native=runtime.thinker.model.eval();x=torch.randn(1,3,8);pos=torch.arange(3)[None,None].expand(3,1,-1)
    args=dict(inputs_embeds=x,position_ids=pos,output_hidden_states=True,use_cache=False)
    expected=native(**args).last_hidden_state.detach()
    inject_qwen_lora(runtime,configs['fixed_av_s2']['qwen_lora'])
    assert torch.equal(expected,native(**args).last_hidden_state)
    with pytest.raises(ValueError,match='already'):inject_qwen_lora(runtime,configs['fixed_av_s2']['qwen_lora'])

def test_parent_transfer_and_resume_exact(configs,tmp_path):
    c=configs['guided_av_event_feature_s1'];m,b,p,a=setup(c)
    opt=torch.optim.AdamW(m.groups(),weight_decay=0)
    r=forward(m,b,p,a);objective(r,c,r['event_counts'],1,1)[0].backward();opt.step();opt.zero_grad()
    ident=dict(experiment=c,data_identity={'data':'same'},source=source_identity(),parent=None)
    path=tmp_path/'step.pt';checkpoint.save(path,m,opt,1,ident,[rng_state()])
    c2=configs['guided_av_event_feature_s2'];m2,b2,p2,a2=setup(c2)
    result=checkpoint.initialize_stage2(path,m2,c2,ident['data_identity'])
    assert result['step']==1
    assert all(torch.equal(v,m2.state_dict()[k]) for k,v in m.state_dict().items())
    for adapter in m2.qwen_lora:assert torch.count_nonzero(adapter.b.weight)==0
    with pytest.raises(ValueError,match='parent must match'):
        checkpoint.initialize_stage2(path,m2,configs['guided_av_s2'],ident['data_identity'])
    restored,rb,rp,ra=setup(c);op2=torch.optim.AdamW(restored.groups(),weight_decay=0)
    assert checkpoint.resume(path,restored,op2,ident)==1
    # Same input and RNG produce exactly the same second update.
    for model,optimizer,bundle,callback in [(m,opt,b,a),(restored,op2,rb,ra)]:
        r=forward(model,bundle,p,callback);objective(r,c,r['event_counts'],1,1)[0].backward();optimizer.step()
    assert all(torch.equal(v,restored.state_dict()[k]) for k,v in m.state_dict().items())
    with pytest.raises(ValueError,match='identity'):checkpoint.resume(path,restored,op2,dict(ident,parent='other'))

def test_teacher_not_loaded_if_disabled(configs):
    m,b,p,a=setup(configs['fixed_av_s1']);b['teacher']={}
    with pytest.raises(ValueError,match='must not load teacher'):forward(m,b,p,a)

def test_stage2_online_reader_uses_same_plan_and_never_cached_hidden(configs,tmp_path):
    from p2_matrix16.data import Dataset
    cfg,s=fixture(old_config());dataset=Dataset.__new__(Dataset)
    dataset.spec=configs['guided_av_s2'];dataset.data={'output':str(tmp_path)}
    obs={'sample_id':'sample','requested_duration_s':1.};dataset.rows=[obs];dataset.targets={'sample':{}}
    dataset.cfg=cfg;dataset.processor=object()
    calls=[]
    def part(sid,kind,device):
        calls.append(kind)
        assert kind=='text'
        return {'positive':{'prompt':'the exact retained plan'}}
    dataset.plans=SimpleNamespace(work=tmp_path,protocol={},part=part)
    item={'guidance':{'prompt':'the exact retained plan'}}
    with (patch('p2_plan_cache.protocol.sample',return_value=item),patch('p2_listen.features.load_input',side_effect=AssertionError('cached hidden read')),
         patch('listen_audio_postprocess.overlay.sample_contract',return_value=(tmp_path,tmp_path,{},None,{},{})),
         patch('listen_audio_postprocess.overlay.compose',return_value={'av':'composed'}),
         patch('p2_matrix16.data.load_part',return_value={}),patch('p2_listen.features.identity',return_value={}),
         patch('p2_listen.features.folder',return_value=tmp_path),patch('p2_listen.features.image_grid',return_value=(None,None,(1,1)))):
        bundle=dataset.sample(0)
    assert calls==['text'] and bundle['condition'] is None
    assert bundle['plan']==bundle['text']['positive']['prompt']
    assert bundle['teacher'] is None

def test_teacher_uses_processed_wave_and_keeps_silence_valid():
    import numpy as np
    from p2_matrix16.teacher import extract
    cfg,s=fixture(old_config());seen=[]
    model=torch.nn.Linear(1,1).requires_grad_(False)
    model.get_video_features=lambda *a:torch.zeros(4,8)
    model.get_audio_features=lambda *a:torch.zeros(50,8)
    def features(wave,**kwargs):
        seen.append(wave.copy())
        assert kwargs['sampling_rate']==16000
        return {'input_features':torch.zeros(1,8,100),'attention_mask':torch.ones(1,100)}
    runtime=SimpleNamespace(thinker=model,config=cfg);processor=SimpleNamespace(feature_extractor=features)
    with patch('p2_listen.features.image_grid',return_value=(None,(28,28),(1,1))),patch('p2_listen.features.sample_frames',return_value=np.zeros((8,28,28,3))),patch('p2_listen.features.video_inputs',return_value={'pixel_values_videos':torch.zeros(1),'video_grid_thw':torch.zeros(1)}),patch('listen_audio_postprocess.recipes.read_listener_audio',return_value=(np.zeros((48000,2),dtype=np.float32),48000,{'processed_waveform_sha256':'processed'})):
        value=extract({}, {'requested_duration_s':1.},{'target_video':'target'},s[1],runtime,processor,{})
    assert len(seen[0])==16000 and not seen[0].any()
    assert value['audio_valid'].all() and value['video_valid'].all()
    assert value['waveform']['processed_waveform_sha256']=='processed'

def test_stage2_lora_optimizer_resume_exact(configs,tmp_path):
    c=configs['guided_av_event_feature_s2'];m,b,p,a=setup(c);m.train()
    opt=torch.optim.AdamW(m.groups(),weight_decay=0)
    r=forward(m,b,p,a);objective(r,c,r['event_counts'],1,1)[0].backward();opt.step();opt.zero_grad()
    ident={'parent':'stage1','config':c};path=tmp_path/'s2.pt'
    checkpoint.save(path,m,opt,1,ident,[rng_state()])
    restored,rb,rp,ra=setup(c);restored.train();op2=torch.optim.AdamW(restored.groups(),weight_decay=0)
    checkpoint.resume(path,restored,op2,ident)
    for model,optimizer,bundle,callback in [(m,opt,b,a),(restored,op2,rb,ra)]:
        r=forward(model,bundle,p,callback);objective(r,c,r['event_counts'],1,1)[0].backward();optimizer.step()
    assert all(torch.equal(v,restored.state_dict()[k]) for k,v in m.state_dict().items())

def test_inference_restores_both_adapters_and_marks_oracle(configs,tmp_path):
    from p2_matrix16 import inference
    c=configs['guided_av_event_feature_s2'];m,b,p,a=setup(c)
    cfg,_=fixture(old_config());runtime,processor=native_qwen(cfg);renderer=TinyTextRenderer()
    for source,dest in [(m.renderer,renderer),(m.qwen_runtime.thinker,runtime.thinker)]:
        state={k.replace('.base.','.'):v for k,v in source.state_dict().items() if '.update.' not in k}
        dest.load_state_dict(state,strict=True)
    ident=dict(experiment=c,source=source_identity(),assets={k:{} for k in ('omni','ltx','gemma')},qwen_targets=m.qwen_targets,text_targets=m.text_targets)
    path=tmp_path/'model.pt';checkpoint.save(path,m,torch.optim.AdamW(m.groups()),0,ident,[rng_state()])
    with patch('p2_listen.config.extractor_config',return_value=cfg),patch('p2_listen.features.weights_identity',return_value={}):
        restored=inference.restore(path,renderer,runtime,processor,'cpu')
    assert not any(p.requires_grad for p in restored.parameters())
    assert all(torch.equal(v,restored.state_dict()[k]) for k,v in m.state_dict().items())
    with patch('p2_listen.contracts.observation',side_effect=lambda x:x),patch('p3_listen.features.prefix',return_value=p),patch('p2_listen.features.make_observation',side_effect=lambda x:x),patch('p2_listen.features.image_grid',return_value=(None,None,(1,1))):
        for src in ('predicted','annotation_oracle'):
            u,meta=inference.conditions(restored,runtime,processor,b['observation'],b['text']['positive'],plan='same plan',plan_source=src)
            assert meta['target_annotation_access']==(src=='annotation_oracle')
            assert not u.video.requires_grad
        with pytest.raises(ValueError,match='exact same'):
            inference.conditions(restored,runtime,processor,b['observation'],b['text']['positive'],plan='different',plan_source='predicted')
