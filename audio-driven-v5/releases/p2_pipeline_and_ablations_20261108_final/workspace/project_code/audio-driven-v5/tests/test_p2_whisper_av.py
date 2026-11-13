"""Native tiny LTX/Whisper regressions; no pretrained renderer or GPUs needed."""
import copy
from dataclasses import asdict, replace
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
sys.path.insert(0,str(ROOT/'src'))
from p2_whisper_av.bootstrap import activate_ltx, CORE
activate_ltx()
from p2_whisper_av.config import load, validate, validate_baseline, DEFAULT
from p2_whisper_av.features import FrozenWhisper, SpeakerFeatures, windows
from p2_whisper_av.model import QwenWhisperAV
from p2_whisper_av.cache import validate_features, observation_contract, speaker_wave
from p2_whisper_av import checkpoint
from p2_whisper_av.training import scaled_av
from p2_listen.audit import fixture
from p2_listen.config import load as data_config, activate
from p2_listen.contracts import read
from p2_listen.training import rng_state

torch.set_num_threads(2)
activate(data_config())


def tiny():
    from ltx_core.model.transformer.model import LTXModel
    from ltx_core.model.transformer.attention import AttentionFunction
    from ltx_trainer.training_strategies.text_to_video import TextToVideoConfig, TextToVideoStrategy
    from ltx_trainer.timestep_samplers import UniformTimestepSampler
    torch.manual_seed(51)
    renderer = LTXModel(num_attention_heads=2,attention_head_dim=16,num_layers=1,
        cross_attention_dim=32,audio_num_attention_heads=2,audio_attention_head_dim=16,
        audio_cross_attention_dim=32,attention_type=AttentionFunction.PYTORCH)
    with torch.no_grad():
        for name,p in renderer.named_parameters():
            if 'scale_shift_table' in name:
                p.normal_(0,.02)
    renderer.requires_grad_(False)
    strategy = TextToVideoStrategy(TextToVideoConfig(with_audio=True,i2av_only=True,first_frame_conditioning_p=1))
    sampler = UniformTimestepSampler(.2,.8)
    av = dict(latents=dict(latents=torch.randn(1,128,2,2,2),num_frames=torch.tensor([2]),
              height=torch.tensor([2]),width=torch.tensor([2]),fps=torch.tensor([25.])),
              audio_latents=dict(latents=torch.randn(1,8,8,16),num_time_steps=torch.tensor([8])))
    text = dict(video=torch.randn(1,4,32),audio=torch.randn(1,4,32),mask=torch.ones(1,4,dtype=torch.bool))
    return renderer,strategy,sampler,av,text


def features():
    starts = torch.arange(9).float()*.02
    return SpeakerFeatures(torch.randn(1,9,12),torch.tensor([[True]*7+[False]*2]),
                           torch.stack((starts,starts+.02),-1)[None,None])


def build():
    from omni_av.ltx_runtime import with_text
    torch.manual_seed(11)
    c = load();c.pop('connector');c['whisper'].update(input_dim=12,context_dim=16)
    cfg,f = fixture(data_config())
    cfg=replace(cfg,video_dim=32,audio_dim=32)
    renderer,strategy,sampler,av,text = tiny()
    torch.manual_seed(c['training']['seed'])
    model = QwenWhisperAV(c,cfg,renderer)
    prepared = strategy.prepare_training_inputs(with_text(av,text),sampler)
    bundle = dict(condition=f[0],layout=f[1],teacher=None,text={'positive':text},whisper=features(),
        av=dict(av=av,support=dict(video=torch.ones_like(prepared.video_loss_mask),
                                  audio=torch.ones_like(prepared.audio_loss_mask))))
    return c,cfg,model,bundle,prepared,strategy,sampler


def opened(model):
    with torch.no_grad():
        for branch in model.whisper_branches:
            branch.to_out[0].weight.normal_(0,.02)


def outputs(model,prepared,speaker):
    p = model.condition(prepared,speaker)
    return model.renderer(video=p.video,audio=p.audio,perturbations=None)


def test_matches_original_av_initialization_and_zero_residual():
    from p2_j_av_only import AVOnlyModel
    from omni_av_p2.ltx import append_conditions
    c,cfg,model,bundle,prepared,_,_ = build()
    renderer,*_ = tiny()
    torch.manual_seed(c['training']['seed'])
    baseline = AVOnlyModel(c,cfg,renderer)
    for name,tensor in baseline.state_dict().items():
        assert torch.equal(tensor,model.state_dict()[name]),name
    u = model.connector(bundle['condition'])
    original = append_conditions(prepared,u)
    found = model.prepare(prepared,u,bundle['whisper'])
    assert found.video.context is original.video.context or torch.equal(found.video.context,original.video.context)
    assert torch.equal(found.audio.context,original.audio.context)
    a = model.renderer(video=original.video,audio=original.audio,perturbations=None)
    b = model.renderer(video=found.video,audio=found.audio,perturbations=None)
    for x,y in zip(a,b):
        torch.testing.assert_close(x,y,rtol=0,atol=0)
    assert all(not b.to_out[0].weight.any() for b in model.whisper_branches)


def test_only_av_gradients_and_frozen_backbones_for_two_updates():
    c,cfg,model,bundle,prepared,strategy,sampler = build()
    frozen = [(p,p.detach().clone()) for p in model.renderer.parameters() if not p.requires_grad]
    opt = torch.optim.AdamW(model.groups(),weight_decay=0.)
    with patch.object(model.event_heads,'forward',side_effect=AssertionError('Event forward called')):
        for step in (1,2):
            opt.zero_grad(set_to_none=True)
            value = model(bundle,strategy,sampler)
            assert set(value) == {'av','av_components'}
            value['av'].backward()
            assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.whisper_branches.parameters())
            assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.text_lora.parameters())
            assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.connector.parameters())
            if step == 2:
                assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.whisper_adapter.parameters())
                assert all(b.to_q.weight.grad.abs().sum()>0 for b in model.whisper_branches)
            opt.step();model.assert_ownership()
    assert all(p.grad is None and torch.equal(p,before) for p,before in frozen)
    assert not any(p.requires_grad for p in model.event_heads.parameters())
    assert len(model.decoders) == len(model.qwen_lora) == 0
    assert not any(k.startswith('renderer.') for k in model.state_dict())
    assert {g['name'] for g in model.groups()} == {'connector','text_lora','whisper'}


@pytest.mark.parametrize('temporal',[True,False])
def test_time_offsets_and_masked_tokens(temporal):
    _,_,model,bundle,prepared,_,_ = build();opened(model)
    model.spec['whisper']['temporal_rope'] = temporal
    speaker = bundle['whisper']
    a = outputs(model,prepared,speaker)
    b = outputs(model,prepared,replace(speaker,bounds=speaker.bounds+.5))
    if temporal:
        assert all(not torch.allclose(x,y) for x,y in zip(a,b))
    else:
        assert all(torch.equal(x,y) for x,y in zip(a,b))
    changed = speaker.hidden.clone();changed[:,7:] = 1000
    z = outputs(model,prepared,replace(speaker,hidden=changed))
    assert all(torch.equal(x,y) for x,y in zip(a,z))
    with pytest.raises(ValueError,match='valid observation'):
        model.condition(prepared,replace(speaker,valid=torch.zeros_like(speaker.valid)))
    with pytest.raises(ValueError,match='required'):
        model.condition(prepared,None)


def test_checkpointing_keeps_whisper_and_qwen_gradient_paths():
    from p2_listen.performance import configure_renderer
    _,_,model,bundle,_,strategy,sampler = build();opened(model)
    configure_renderer(model.renderer,'all');model.train()
    model(bundle,strategy,sampler)['av'].backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.whisper_adapter.parameters())
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.connector.parameters())


def test_cfg_retains_qwen_and_whisper_both_paths_and_reference():
    from p2_whisper_av.inference import sample
    _,_,model,bundle,prepared,_,_ = build();opened(model);model.eval()
    u = model.connector(bundle['condition']);speaker = bundle['whisper'];seen=[]
    def hook(module,args,kwargs):
        seen.append((kwargs['video'],kwargs['audio']))
    handle = model.renderer.register_forward_pre_hook(hook,with_kwargs=True)
    try:
        out = sample(model,prepared,u,speaker,
            dict(video=torch.zeros_like(prepared.video.context),audio=torch.zeros_like(prepared.audio.context),
                 mask=prepared.video.context_mask),steps=2,sigmas=torch.tensor([1.,.5,0.]))
    finally:
        handle.remove()
    assert len(seen) == 4
    for video,audio in seen:
        assert video.qwen_context is not None and audio.qwen_context is not None
        assert video.qwen_time_positions is speaker.bounds
        assert torch.equal(video.context[:,-u.video.shape[1]:],u.video)
        assert torch.equal(audio.context[:,-u.audio.shape[1]:],u.audio)
    keep = ~prepared.video_loss_mask
    assert torch.equal(out['video'][keep],prepared.video.latent.float()[keep])


def test_strict_checkpoint_resume_restores_next_update_and_rejects_old(tmp_path):
    _,_,model,bundle,_,strategy,sampler = build()
    opt = torch.optim.AdamW(model.groups(),weight_decay=0.)
    def update():
        opt.zero_grad(set_to_none=True);model(bundle,strategy,sampler)['av'].backward();opt.step()
    update();identity={'test':'whisper_av','world':1}
    path=tmp_path/'step_000001.pt';checkpoint.save(path,model,opt,1,identity,[rng_state()])
    update();expected=copy.deepcopy(model.state_dict())
    assert checkpoint.resume(path,model,opt,identity) == 1
    update()
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in expected.items())
    with pytest.raises(ValueError,match='identity'):
        checkpoint.resume(path,model,opt,dict(identity,world=2))
    from p2_matrix16.checkpoint import save as old_save
    old=tmp_path/'old.pt';old_save(old,model,opt,2,identity,[rng_state()])
    with pytest.raises(ValueError,match='Not a complete'):
        checkpoint.checked(old)


@pytest.mark.parametrize('n',[1,319,320,321,26*16000,30*16000,61*16000+19])
def test_window_grid_no_duplicates_or_truncation(n):
    indices=[]
    for begin,end,cb,ce in windows(n):
        assert end-begin <= 30*16000 and begin%320 == 0
        indices.extend(i for i in range(begin,end,320) if cb <= i < ce)
    assert indices == list(range(0,n,320))


def test_real_tiny_whisper_batch_preserves_silence_and_matches_serial():
    from transformers import WhisperConfig,WhisperModel,WhisperFeatureExtractor
    encoder=FrozenWhisper.__new__(FrozenWhisper)
    encoder.encoder=WhisperModel(WhisperConfig(d_model=16,encoder_layers=1,decoder_layers=1,
        encoder_attention_heads=2,decoder_attention_heads=2,encoder_ffn_dim=32,decoder_ffn_dim=32)).encoder.eval().requires_grad_(False)
    encoder.processor=WhisperFeatureExtractor(feature_size=80);encoder.device='cpu';encoder.dtype=torch.float32
    waves=[np.zeros(321,dtype=np.float32),np.sin(np.arange(16000,dtype=np.float32)*.1)*.1]
    batch=encoder.encode_batch(waves,batch_size=2)
    for wave,value in zip(waves,batch):
        expected=encoder.encode(wave)
        torch.testing.assert_close(value.hidden,expected.hidden,atol=2e-6,rtol=2e-5)
        assert torch.equal(value.bounds,expected.bounds)
        assert not value.hidden.requires_grad and not torch.is_inference(value.hidden)
    assert batch[0].valid.all()  # All-zero audio is observed silence, not missing data.
    assert batch[0].hidden.shape == (1,2,16)
    assert batch[0].bounds[0,0,0,0] == 0


def test_cache_time_and_hash_validation(tmp_path):
    from p2_listen.cache import save_part,load_part
    n=641;starts=torch.arange(3)*320
    x=SpeakerFeatures(torch.randn(1,3,12),torch.ones(1,3,dtype=torch.bool),
        torch.stack((starts,(starts+320).clamp(max=n)),-1).double().div(16000).float()[None,None])
    value=dict(features=asdict(x),num_samples_16k=n)
    validate_features(value,12);save_part(tmp_path,{'speaker':'source'},value)
    validate_features(load_part(tmp_path,{'speaker':'source'}),12)
    with pytest.raises(ValueError,match='identity'):
        load_part(tmp_path,{'speaker':'listener'})
    bad=copy.deepcopy(value);bad['features']['bounds'] += .1
    with pytest.raises(ValueError,match='time grid'):
        validate_features(bad,12)
    payload=tmp_path/'payload.pt';payload.write_bytes(payload.read_bytes()+b'tamper')
    with pytest.raises(ValueError,match='Corrupt'):
        load_part(tmp_path,{'speaker':'source'})


def test_whisper_finalize_rejects_missing_sample_and_publishes_only_full_audit(tmp_path):
    from p2_whisper_av.cache import finalize
    from p2_listen.contracts import atomic_json
    class FakeReader:
        work=tmp_path
        protocol={'sample_ids':['one','two']}
        ids={'one','two'}
        missing=True
        def __init__(self,*args,**kwargs):
            pass
        def load(self,sid):
            if self.missing and sid == 'two':
                raise FileNotFoundError('Missing sample')
            return features()
        def folder(self,sid):
            return self.work/'cache'/sid
    atomic_json(tmp_path/'protocol.json',{'frozen':'samples'})
    for sid in ('one','two'):
        atomic_json(tmp_path/'cache'/sid/'complete.json',{'identity':sid})
    with patch('p2_whisper_av.cache.Reader',FakeReader):
        with pytest.raises(FileNotFoundError):
            finalize({'whisper_cache':str(tmp_path),'j_cache':'source'})
        assert not (tmp_path/'COMPLETE.json').exists()
        FakeReader.missing=False
        result=finalize({'whisper_cache':str(tmp_path),'j_cache':'source'})
        assert result['success'] and result['samples'] == 2


def test_resampling_uses_only_speaker_preserving_leading_silence():
    item={'source':{'speaker_id':'S','listener_id':'L','source_start_s':1.,'source_end_s':2.},
          'speaker':{'path':'speaker.wav','contract':{'source':{'member':'S.mp4'}}},'sample_id':'sample','split':'train'}
    assert observation_contract(item)['origin_s'] == 0
    wave=np.zeros((48000,1),dtype=np.float32);wave[24000:,0]=.1
    with patch('p2_j_cache.data.read_speaker',return_value=(wave,48000)) as reader:
        result=speaker_wave(item)
    reader.assert_called_once_with(item)
    assert len(result) == 16000 and not result[:7900].any() and result[-1]>0
    item['source']['speaker_id']='L'
    with pytest.raises(ValueError,match='Speaker/listener'):
        observation_contract(item)


def test_config_baseline_and_four_node_commands():
    from p2_j_whisper_av import commands,CLUSTER
    c=load();validate_baseline(c,read(c['baseline_config']))
    assert c['training']['save_every'] == 500 and c['training']['steps'] == 8927
    rows=commands(read(CLUSTER),DEFAULT,stop_after=2)
    assert len(rows) == 4
    for i,row in enumerate(rows):
        assert '--nnodes=4' in row and '--nproc_per_node=8' in row and f'--node_rank={i}' in row
        assert str(ROOT/'scripts/p2_j_whisper_av.py') in row
        assert row[-2:] == ['--stop-after','2']
    for key,value in [('loss','av_event'),('qwen_hidden_source','sft'),('teacher_cache','teacher')]:
        bad=copy.deepcopy(c);bad[key]=value
        with pytest.raises(ValueError):
            validate(bad)


def test_distributed_av_scaling_and_private_core():
    loss=torch.tensor(2.,requires_grad=True)
    scaled_av(loss.square(),2,32).backward()
    assert loss.grad.item() == 4/64
    import ltx_core
    assert Path(ltx_core.__file__).is_relative_to(CORE)


def test_gemma_relocation_requires_identical_content_and_retains_assets():
    from p2_whisper_av.compat import canonical_dependencies
    rel='text_encoders/gemma/tokenizer.py'
    metadata={'origin':'/baseline/ltx_core','files':{rel:'same'}}
    original={'code':{str(CORE/'ltx_core'/rel):'same','/other.py':'other'},'assets':{'weights':'pinned'}}
    found=canonical_dependencies(original,metadata)
    assert found == {'code':{'/baseline/ltx_core/'+rel:'same','/other.py':'other'},'assets':{'weights':'pinned'}}
    original['code'][str(CORE/'ltx_core'/rel)]='changed'
    with pytest.raises(ValueError,match='Changed private Gemma'):
        canonical_dependencies(original,metadata)


def _distributed_worker(rank, rendezvous, destination):
    import torch.distributed as dist
    from p2_whisper_av.execution import StreamingFP64Reduction
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method=rendezvous,rank=rank,world_size=2)
    try:
        _,_,model,bundle,_,strategy,sampler=build()
        opt=torch.optim.AdamW(model.groups(),weight_decay=0.)
        for step in (1,2):
            opt.zero_grad(set_to_none=True)
            acc=StreamingFP64Reduction(model.parameters(),1)
            torch.manual_seed(step*1000+rank)
            (model(bundle,strategy,sampler)['av']/2).backward()
            acc.add();acc.finish(2);opt.step()
        if rank == 0:
            torch.save(model.state_dict(),destination)
    finally:
        dist.destroy_process_group()


def test_two_rank_cpu_matches_serial_av_updates(tmp_path):
    from p2_listen.performance import BucketGradientAccumulator
    import torch.multiprocessing as mp
    torch.set_num_threads(1)
    destination=tmp_path/'two_rank.pt'
    mp.spawn(_distributed_worker,args=('file://'+str(tmp_path/'rendezvous'),str(destination)),nprocs=2,join=True)
    _,_,model,bundle,_,strategy,sampler=build()
    opt=torch.optim.AdamW(model.groups(),weight_decay=0.)
    for step in (1,2):
        opt.zero_grad(set_to_none=True);acc=BucketGradientAccumulator(model.parameters(),1)
        for rank in range(2):
            torch.manual_seed(step*1000+rank)
            (model(bundle,strategy,sampler)['av']/2).backward();acc.add()
        acc.finish();opt.step()
    actual=torch.load(destination,weights_only=True)
    for k,v in model.state_dict().items():
        torch.testing.assert_close(actual[k],v,atol=0,rtol=0,msg=k)
