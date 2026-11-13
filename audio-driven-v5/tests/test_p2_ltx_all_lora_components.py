"""Real native tiny LTX: both all-LoRA leave-one-out arms."""
import copy
from types import SimpleNamespace
from unittest.mock import patch
import pytest,torch
from p2_ltx_all_lora_components import ARMS,checkpoint
from p2_ltx_all_lora_components.config import expected,validate
from p2_ltx_all_lora_components.model import Model
from p2_ltx_all_lora_components.lora import ATTENTION,FFN,WHISPER
from p2_ltx_events.heads import objective
from p2_listen.training import rng_state
from test_p2_current_component_ablation import build as old_build,forward
from test_p2_whisper_av import tiny,opened

torch.set_num_threads(2)

def build(arm):
 c,cfg,old,b,s,sa=old_build(arm)
 renderer,*_=tiny();torch.manual_seed(c['training']['seed']);m=Model(c,cfg,renderer)
 return c,old,m,b,s,sa

@pytest.mark.parametrize('arm',ARMS)
def test_all_lora_and_exactly_one_branch_absent(arm):
 from omni_av_p21.lora import AdaptedLinear
 c,old,m,b,s,sa=build(arm)
 assert len(m.text_targets)==28
 assert {g['name']for g in m.groups()}=={'connector','ltx_lora','events', 'whisper' if arm=='no_feature'else'features'}
 for block in m.renderer.transformer_blocks:
  for name in ATTENTION:
   assert all(isinstance(getattr(block,name).get_submodule(p),AdaptedLinear)for p in ['to_q','to_k','to_v','to_out.0'])
  for name in FFN:
   assert isinstance(getattr(block,name).net[0].proj,AdaptedLinear)and isinstance(getattr(block,name).net[2],AdaptedLinear)
  for name in WHISPER:
   if arm=='no_whisper':assert not hasattr(block,name)
   else:
    assert all(p.requires_grad for p in getattr(block,name).parameters())
    assert not any(isinstance(x,AdaptedLinear)for x in getattr(block,name).modules())
 assert bool(m.decoders)==(arm=='no_whisper')
 assert (m.ltx_lora_report['whisper_full_parameters']>0)==(arm=='no_feature')
 m.assert_ownership()

@pytest.mark.parametrize('arm',ARMS)
@pytest.mark.parametrize('recompute',[False,True])
def test_unchanged_initialization_and_forward(arm,recompute):
 c,old,m,b,s,sa=build(arm)
 for k,v in old.state_dict().items():assert torch.equal(v,m.state_dict()[k]),k
 for model in [old,m]:model.train();model.renderer.set_gradient_checkpointing(recompute)
 torch.manual_seed(991);a=forward(old,b,s,sa)
 torch.manual_seed(991);v=forward(m,b,s,sa)
 for k in ['av']:torch.testing.assert_close(a[k],v[k],rtol=0,atol=0)
 for part in ['features','event_sums']:
  for k in a[part]:torch.testing.assert_close(a[part][k],v[part][k],rtol=0,atol=0)

@pytest.mark.parametrize('arm',ARMS)
@pytest.mark.parametrize('task',['av','event','full'])
def test_gradients_and_frozen_base(arm,task):
 c,_,m,b,s,sa=build(arm)
 if arm=='no_feature':opened(m)
 m.train();m.renderer.set_gradient_checkpointing(True)
 frozen=[(p,p.detach().clone())for p in m.renderer.parameters()if not p.requires_grad]
 v=forward(m,b,s,sa);counts={k:x['known'].sum()for k,x in b['events'].items()}
 loss=v['av']if task=='av'else sum(v['event_sums'].values())if task=='event'else objective(v,c,counts)[0]
 loss.backward()
 for family in (*ATTENTION,*FFN):
  targets=[x for n,x in zip(m.text_targets,m.text_lora)if n.split('.')[2]==family]
  assert targets and any(p.grad is not None and p.grad.abs().sum()>0 for x in targets for p in x.parameters()),family
 assert any(p.grad is not None and p.grad.abs().sum()>0 for p in m.connector.parameters())
 if arm=='no_feature':
  for mod in [m.whisper_branches,m.whisper_adapter]:assert any(p.grad is not None and p.grad.abs().sum()>0 for p in mod.parameters())
 if task=='event':assert all(p.grad is None for p in m.decoders.parameters())
 torch.optim.AdamW(m.groups(),weight_decay=0.).step()
 assert all(p.grad is None and torch.equal(p,v)for p,v in frozen)
 m.assert_ownership()

def test_feature_only_updates_connector_and_decoders():
 c,_,m,b,s,sa=build('no_whisper');o=torch.optim.AdamW(m.groups(),weight_decay=0.)
 for _ in range(3):
  o.zero_grad(set_to_none=True);v=forward(m,b,s,sa);sum(v['features'].values()).backward();o.step()
 for mod in [m.connector,m.decoders]:assert any(p.grad is not None and p.grad.abs().sum()>0 for p in mod.parameters())
 for mod in [m.text_lora,m.event_heads]:assert all(p.grad is None for p in mod.parameters())
 assert not b['condition'].hidden.requires_grad

@pytest.mark.parametrize('arm',ARMS)
def test_full_model_shared_initialization(arm):
 from test_p2_ltx_all_lora import fixture
 _,_,full,fb,_,fs,fsa=fixture()
 c,_,m,b,s,sa=build(arm)
 for k,v in m.state_dict().items():assert torch.equal(v,full.state_dict()[k]),k
 shared=dict(b,teacher=b['teacher'] if b.get('teacher') is not None else fb['teacher'],whisper=fb['whisper'],events=fb['events'],hidden_event_layout=fb['hidden_event_layout'])
 torch.manual_seed(818);a=forward(m,b,s,sa)
 torch.manual_seed(818);v=forward(full,shared,fs,fsa)
 torch.testing.assert_close(a['av'],v['av'],rtol=0,atol=0)
 for k in a['event_sums']:torch.testing.assert_close(a['event_sums'][k],v['event_sums'][k],rtol=0,atol=0)
 for k in a['features']:torch.testing.assert_close(a['features'][k],v['features'][k],rtol=0,atol=0)

@pytest.mark.parametrize('arm',ARMS)
def test_loss_and_exact_resume(arm,tmp_path):
 c,_,m,b,s,sa=build(arm);o=torch.optim.AdamW(m.groups(),weight_decay=0.);identity=dict(world=1,arm=arm,scope=m.text_targets)
 def step():
  o.zero_grad(set_to_none=True);v=forward(m,b,s,sa);loss,parts,event=objective(v,c,{k:x['known'].sum()for k,x in b['events'].items()})
  torch.testing.assert_close(loss,v['av']+.05*event+sum(.025*x for x in v['features'].values()))
  assert parts['temporal']==0
  loss.backward();o.step()
 step();p=tmp_path/'step.pt';checkpoint.save(p,m,o,1,identity,[rng_state()]);step();wanted=copy.deepcopy(m.state_dict());adam=copy.deepcopy(o.state_dict())
 checkpoint.resume(p,m,o,identity);step()
 assert all(torch.equal(v,m.state_dict()[k])for k,v in wanted.items())
 for i,state in adam['state'].items():
  for k,v in state.items():assert torch.equal(v,o.state_dict()['state'][i][k])
 with pytest.raises(ValueError):checkpoint.resume(p,m,o,dict(identity,arm='other'))
 with patch('p2_ltx_all_lora_components.checkpoint.VERSION','old'):
  with pytest.raises(ValueError):checkpoint.checked(p)

@pytest.mark.parametrize('arm',ARMS)
def test_readers_and_strict_recipe(arm):
 from p2_ltx_all_lora_components.data import Dataset
 from p2_current_component_ablation.config import flags
 c=expected('/tmp/expanded-'+arm,arm);validate(c)
 d=Dataset.__new__(Dataset);d.spec=c;d.index={};d.ids=['sample'];d.components=flags(c)
 teacher=SimpleNamespace(h=SimpleNamespace(protocol={'sample_ids':['sample']}))
 with patch('p2_current_component_ablation.data.open_teacher',return_value=teacher)as t,patch('p2_current_component_ablation.data.open_speaker')as w,patch('p2_current_component_ablation.data.event_readers')as e:
  d.open_aux();assert t.called==(arm=='no_whisper')and w.called==(arm=='no_feature')and e.called
 for change in [lambda x:x['loss_weights'].update(event=0),lambda x:x.update(j_cache='/old'),lambda x:x['text_lora'].update(rank=32),lambda x:x['ltx_lora_scope']['attention'].remove('attn1'),lambda x:x.update(component_arm='no_event')]:
  bad=copy.deepcopy(c);change(bad)
  with pytest.raises(ValueError):validate(bad)

@pytest.mark.parametrize('arm', ARMS)
def test_native_cfg_retains_enabled_conditions_and_reference(arm):
    from omni_av.ltx_runtime import with_text
    from p2_ltx_all_lora_components.inference import sample
    _, _, model, bundle, strategy, sampler = build(arm)
    if (arm=='no_feature'):
        opened(model)
    model.eval()
    prepared = strategy.prepare_training_inputs(
        with_text(bundle['av']['av'], bundle['text']['positive']), sampler)
    with torch.no_grad():
        u = model.connector(bundle['condition'])
    speaker = bundle.get('whisper')
    seen = []

    def capture(module, args, kwargs):
        seen.append((kwargs['video'], kwargs['audio']))

    handle = model.renderer.register_forward_pre_hook(capture, with_kwargs=True)
    try:
        out = sample(model, prepared, u, speaker, {
            'video': torch.zeros_like(prepared.video.context),
            'audio': torch.zeros_like(prepared.audio.context),
            'mask': prepared.video.context_mask,
        }, steps=2, sigmas=torch.tensor([1., .5, 0.]))
    finally:
        handle.remove()
    assert len(seen) == 4
    for video, audio in seen:
        for mod, value in [('video', video), ('audio', audio)]:
            assert (value.qwen_context is not None) == (arm=='no_feature')
            if speaker is not None:
                assert value.qwen_time_positions is speaker.bounds
                assert torch.equal(value.qwen_context_mask, speaker.valid)
            condition = getattr(u, mod)
            assert torch.equal(value.context[:, -condition.shape[1]:], condition)
    keep = ~prepared.video_loss_mask
    assert torch.equal(out['video'][keep], prepared.video.latent.float()[keep])
    assert all(torch.isfinite(v).all() for v in out.values())


@pytest.mark.parametrize('arm',ARMS)
def test_cli_stop_and_reserved_gpu_and_source_contract(arm,monkeypatch):
 import sys
 import p2_ltx_all_lora_components_train as cli
 import p2_ltx_all_lora_components.training as training
 from p2_listen.contracts import read,ROOT
 from p2_ltx_all_lora_components.config import load,source_identity
 out=ROOT/'experiments/p2_ltx_all_lora_components_20261109'/arm
 c=load(out/'config.json');commands=cli.commands(c,stop_after=6000)
 assert len(commands)==4 and all('P2_ALL_LORA_COMPONENT_STOP_AFTER=6000'in cmd for cmd in commands)
 assert 'CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7'in commands[-1]
 assert all(str(cli.SELF)in cmd for cmd in commands)
 with pytest.raises(ValueError):cli.commands(c,resume=out.parent/'wrong/step.pt')
 calls=[];monkeypatch.setenv('P2_ALL_LORA_COMPONENT_STOP_AFTER','6000')
 monkeypatch.setattr(training,'train',lambda c,resume,stop:calls.append((c['component_arm'],stop)))
 monkeypatch.setattr(sys,'argv',['cli','train','--config',str(out/'config.json')]);cli.main()
 assert calls==[(arm,6000)]
 assert str(ROOT/'src/p2_ltx_all_lora_components/lora.py')in source_identity()
