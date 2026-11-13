"""Real native tiny LTX, time alignment, weak labels, gradients and strict resume."""
import copy,json,hashlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import pytest
import torch

from test_p2_whisper_event_feature import build as old_build
from test_p2_whisper_av import tiny,opened
from p2_listen.contracts import ROOT,read
from p2_ltx_events import VERSION
from p2_ltx_events.config import TEMPORAL_CLASSES,validate,validate_baseline
from p2_ltx_events.layout import make_layout,rasterize,check_prepared
from p2_ltx_events.heads import pool_time,objective
from p2_ltx_events.renderer import forward_with_hidden
from p2_ltx_events.labels import media_key,normalize_old,join_target,current_audio_support


def new_config(c):
    c=copy.deepcopy(c)
    c.update(output='/tmp/p2_ltx_hidden_events_test',hidden_event_experiment=VERSION,
        hidden_event_parent_config=str(ROOT/'configs/p2_j_qwen_whisper_event_feature_20261105.json'),
        temporal_labels='/tmp/p2_ltx_hidden_events_test_labels',hidden_event_supervision=dict(
        readout='last_block_before_output',classes=TEMPORAL_CLASSES,clip_weight=1.,temporal_weight=1.,
        background_weight=.1,label_policy='historical-weak-intervals-v1'))
    return c


def build(time_classes=None,clip_only=False):
    from p2_ltx_events.model import LTXHiddenEvents
    c,cfg,_,bundle,prepared,strategy,sampler=old_build();c=new_config(c)
    if time_classes is not None:c['hidden_event_supervision']['classes']=time_classes
    if clip_only:
        c.pop('temporal_labels')
        c['hidden_event_supervision']=dict(readout='last_block_before_output',clip_weight=1.,temporal_weight=0.)
    renderer,*_=tiny();torch.manual_seed(c['training']['seed']);model=LTXHiddenEvents(c,cfg,renderer)
    layout=make_layout(bundle['av']['av'],bundle['av']['support'],.36)
    selected=c['hidden_event_supervision'].get('classes',{'video':[],'audio':[]})
    row={'classes':{n:dict(coverage=[[0.,.36]],spans=[[.10,.25]] if n in ('head_nod','brow_lowering','brows_raised') else (
        [[.08,.16]] if n=='listener_vocalization' else []),unknown=[]) for ns in selected.values() for n in ns}}
    labels={'clip/'+k:v for k,v in bundle['events'].items()};labels.update(rasterize(row,layout,selected))
    bundle.update(events=labels,hidden_event_layout=layout)
    return c,model,bundle,prepared,strategy,sampler


def test_clip_only_needs_no_temporal_targets_and_backpropagates_through_ltx():
    from p2_ltx_events.heads import names
    c,model,bundle,_,strategy,sampler=build(clip_only=True);opened(model)
    assert set(model.event_heads.heads)=={'clip_video','clip_audio'}
    assert len(names(c))==19 and set(names(c))==set(bundle['events'])
    values=forward(model,bundle,strategy,sampler)
    counts={k:v['known'].sum() for k,v in bundle['events'].items()}
    total,losses,event=objective(values,c,counts)
    torch.testing.assert_close(event,losses['clip'])
    torch.testing.assert_close(total,values['av']+.05*losses['clip']+
        .025*values['features']['video']+.025*values['features']['audio'])
    assert losses['temporal']==0
    event.backward()
    for module in (model.event_heads,model.connector,model.text_lora,model.whisper_branches,model.whisper_adapter):
        assert any(p.grad is not None and p.grad.abs().sum()>0 for p in module.parameters())
    assert all(p.grad is None for p in model.decoders.parameters())


def test_disabled_temporal_objective_ignores_invalid_temporal_loss():
    p=torch.tensor(.3,requires_grad=True)
    c=dict(loss_weights=dict(av=1.,event=.05,video_feature=.025,audio_feature=.025),
           hidden_event_supervision=dict(clip_weight=1.,temporal_weight=0.))
    values=dict(av=p.square(),features={'video':p.square(),'audio':p.square()},
                event_sums={'clip/a':p.square()*3,'temporal/b':p*float('nan')})
    total,losses,event=objective(values,c,{'clip/a':3.})
    assert torch.isfinite(total) and losses['temporal']==0
    torch.testing.assert_close(event,p.square())
    torch.testing.assert_close(torch.autograd.grad(total,p)[0],2*p*1.10)


def test_legacy_clip_only_duration_comes_from_frozen_media_not_id_only_rows():
    from p2_ltx_events.data import LegacyDataset,OriginalDataset
    c,_,bundle,_,_,_=build(clip_only=True)
    cached=dict(bundle,sample_id='example',events={k.removeprefix('clip/'):v for k,v in bundle['events'].items()})
    dataset=LegacyDataset.__new__(LegacyDataset)
    dataset.spec=c;dataset.temporal=None;dataset.rows=[dict(sample_id='example')]
    dataset.reader=SimpleNamespace(work=Path('/unused'),manifest={})
    with patch.object(OriginalDataset,'sample',return_value=cached),patch('p2_j_cache.data.sample',return_value={'source':{'duration_s':.36}}):
        value=dataset.sample(0)
    assert set(value['events'])==set(bundle['events'])
    assert value['hidden_event_layout']['video_token_valid'].any()


def forward(model,bundle,strategy,sampler):
    return model(bundle,{k:1. for k in bundle['events']},torch.Generator().manual_seed(97),strategy,sampler)


@pytest.mark.parametrize('checkpointing',[False,True])
def test_native_forward_and_gradient_parity(checkpointing):
    from omni_av.ltx_runtime import with_text
    c,model,bundle,_,strategy,sampler=build();opened(model)
    model.renderer.set_gradient_checkpointing(checkpointing);model.train()
    u=model.connector(bundle['condition'])
    prepared=strategy.prepare_training_inputs(with_text(bundle['av']['av'],bundle['text']['positive']),sampler)
    prepared=model.prepare(prepared,u,bundle['whisper']);check_prepared(bundle['hidden_event_layout'],prepared)
    native=model.renderer(video=prepared.video,audio=prepared.audio,perturbations=None)
    found,hidden=forward_with_hidden(model.renderer,prepared.video,prepared.audio)
    assert all(torch.equal(a,b) for a,b in zip(native,found))
    assert all(v.requires_grad for v in hidden.values())
    params=list(model.text_lora.parameters())+list(model.whisper_branches.parameters())
    a=torch.autograd.grad(sum(x.square().mean() for x in native),params,retain_graph=True,allow_unused=True)
    b=torch.autograd.grad(sum(x.square().mean() for x in found),params,retain_graph=True,allow_unused=True)
    for x,y in zip(a,b):
        if x is None:assert y is None
        else:torch.testing.assert_close(x,y,rtol=0,atol=0)


@pytest.mark.parametrize('task',['clip','temporal'])
def test_event_alone_reaches_generator_connector_and_whisper(task):
    _,model,bundle,_,strategy,sampler=build();opened(model)
    model.renderer.set_gradient_checkpointing(True);model.train()
    frozen=[(p,p.detach().clone()) for p in model.renderer.parameters() if not p.requires_grad]
    values=forward(model,bundle,strategy,sampler)
    event=sum(v for k,v in values['event_sums'].items() if k.startswith(task+'/'))
    event.backward()
    for module in [model.event_heads,model.connector,model.text_lora,model.whisper_branches,model.whisper_adapter]:
        assert any(p.grad is not None and p.grad.abs().sum()>0 for p in module.parameters())
    assert all(p.grad is None and torch.equal(p,old) for p,old in frozen)
    assert all(p.grad is None for p in model.decoders.parameters())
    assert not bundle['condition'].hidden.requires_grad


def test_event_alone_opens_fresh_zero_initialized_whisper_residual():
    _,model,bundle,_,strategy,sampler=build()
    optimizer=torch.optim.AdamW(model.groups())
    for step in range(2):
        optimizer.zero_grad(set_to_none=True);values=forward(model,bundle,strategy,sampler)
        sum(values['event_sums'].values()).backward()
        assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.whisper_branches.parameters())
        if step:assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.whisper_adapter.parameters())
        optimizer.step();model.assert_ownership()


def test_current_nine_time_classes_run_through_native_joint_backbone():
    from p2_ltx_events.config import CURRENT_TEMPORAL_CLASSES
    _,model,bundle,_,strategy,sampler=build(CURRENT_TEMPORAL_CLASSES)
    values=forward(model,bundle,strategy,sampler)
    assert len(values['event_sums'])==28
    assert set(values['event_sums'])==set(bundle['events'])
    assert bundle['events']['temporal/brow_lowering']['y'][0,1]>0
    sum(values['event_sums'].values()).backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.text_lora.parameters())


def test_real_causal_time_grid_reference_and_fractional_occupancy():
    _,_,bundle,_,_,_=build();layout=bundle['hidden_event_layout']
    torch.testing.assert_close(layout['video_intervals'],torch.tensor([[[0.,.04],[.04,.36]]]))
    assert not layout['video_token_valid'][0,:4].any()
    assert layout['video_token_valid'][0,4:].all()
    z=bundle['events']['temporal/head_nod']
    assert not z['known'][0,0] and z['known'][0,1]
    assert float(z['y'][0,1])==pytest.approx(.15/.32)
    assert not bundle['events']['temporal/smile']['y'].any()
    assert float(bundle['events']['temporal/smile']['weight'][0,1])==pytest.approx(.1)


def test_timestamp_mismatch_is_rejected():
    _,_,bundle,prepared,_,_=build()
    changed=copy.deepcopy(bundle['hidden_event_layout']);changed['audio_positions']+=.01
    with pytest.raises(ValueError,match='timestamp'):check_prepared(changed,prepared)


@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
def test_long_clip_native_coordinate_dtype_preserves_precise_label_clock(dtype):
    from omni_av.ltx_runtime import with_text
    from p2_ltx_events.layout import _position_strategy
    from ltx_trainer.timestep_samplers import UniformTimestepSampler
    _,_,_,av,text=tiny()
    # Covers frame indices >256, where BF16 rounds integers before /fps,
    # and audio positions whose BF16 end/start coordinates can coincide.
    av['latents'].update(latents=torch.zeros(1,128,40,2,2,dtype=dtype),num_frames=torch.tensor([40]))
    av['audio_latents'].update(latents=torch.zeros(1,8,313,16,dtype=dtype),num_time_steps=torch.tensor([313]))
    strategy=_position_strategy();prepared=strategy.prepare_training_inputs(with_text(av,text),UniformTimestepSampler(.2,.8))
    support={'video':torch.ones(1,160,dtype=torch.bool),'audio':torch.ones(1,313,dtype=torch.bool)}
    layout=make_layout(av,support,12.52);check_prepared(layout,prepared)
    assert layout['video_intervals'].dtype==layout['audio_intervals'].dtype==torch.float32
    assert (layout['audio_intervals'][...,1]>layout['audio_intervals'][...,0]).all()
    torch.testing.assert_close(layout['video_intervals'][0,-1],torch.tensor([12.2,12.52]))
    assert layout['video_positions'].dtype==dtype


def test_spatial_pooling_and_masked_tokens_cannot_change_predictions():
    c,model,bundle,_,_,_=build();layout=bundle['hidden_event_layout']
    hidden={'video':torch.randn(1,8,32),'audio':torch.randn(1,8,32)}
    masks={m:layout[m+'_token_valid'].clone() for m in hidden}
    a=model.event_heads(hidden,layout,masks)
    changed={k:v.clone() for k,v in hidden.items()};changed['video'][:,:4]=1e8
    b=model.event_heads(changed,layout,masks)
    assert all(torch.equal(a[k],b[k]) for k in a)
    x=torch.tensor([[[1.],[3.],[10.],[14.]]],requires_grad=True)
    y,valid=pool_time(x,torch.tensor([[0,0,1,1]]),torch.tensor([[1,1,1,0]],dtype=torch.bool),2)
    torch.testing.assert_close(y,torch.tensor([[[2.],[10.]]]))
    y.sum().backward();torch.testing.assert_close(x.grad,torch.tensor([[[.5],[.5],[1.],[0.]]]))


def test_unknown_partial_coverage_and_padding_are_masked():
    _,_,bundle,_,_,_=build();layout=bundle['hidden_event_layout']
    row={'classes':{'head_nod':dict(spans=[[.1,.2]],coverage=[[0.,.35]],unknown=[])}}
    z=rasterize(row,layout,TEMPORAL_CLASSES)
    assert not z['temporal/head_nod']['known'].any()
    assert not z['temporal/smile']['known'].any()
    row['classes']['head_nod']['coverage']=[[0.,.36]];row['classes']['head_nod']['unknown']=[[.2,.21]]
    assert not rasterize(row,layout,TEMPORAL_CLASSES)['temporal/head_nod']['known'].any()


def target_item():
    from listen_benchmark.official_audio_gate import POLICY,make_gate
    official=dict(words=[dict(start=.2,end=.4)],sentences=[],vad=[])
    _,info=make_gate(official,1000,1000)
    source=dict(source=dict(archive='/a/train.tar',member='listener.mp4',offset=512,byte_size=100),
        listener_id='listener',source_start_s=10.,source_end_s=11.,duration_s=1.)
    return dict(sample_id='new',source=source,audio_recipe=dict(schema='j-listener-segment-official-union-v1',
        listener_id='listener',source=dict(sample_rate=1000,first_sample=10000,last_sample=11000),
        official_listener=official,gate_policy=POLICY,extra_mute_intervals=[],source_time_origin_s=10.),
        listener_audio_identity=dict(positive_samples=info['positive_samples']))


def test_join_uses_exact_listener_media_source_time_and_current_audio_gate():
    item=target_item();s=copy.deepcopy(item['source']);s.update(source_start_s=9.5,source_end_s=11.5)
    old=dict(sample_id='old',source=s,duration_s=2.,raw_completion_status='complete_candidates',events=[
        dict(start=.6,end=1.2,modality='visual',categories=['head_nod']),
        dict(start=.5,end=1.4,modality='audio',categories=['listener_vocalization'])])
    history=normalize_old(old);row=join_target(item,[history])
    torch.testing.assert_close(torch.tensor(row['classes']['head_nod']['spans']),torch.tensor([[.1,.7]]))
    torch.testing.assert_close(torch.tensor(row['classes']['listener_vocalization']['spans']),torch.tensor([[.101,.499]]))
    wrong=copy.deepcopy(history);wrong['media_key']='wrong'
    with pytest.raises(ValueError,match='media'):join_target(item,[wrong])
    wrong=copy.deepcopy(item['source']);wrong['listener_id']='speaker'
    with pytest.raises(ValueError,match='Listener'):media_key(wrong)
    assert not any(x['coverage'] for x in join_target(item,[])['classes'].values())


@pytest.mark.parametrize('rate',[1000,16000,48000])
def test_analytical_gate_matches_actual_waveform_mask(rate):
    from listen_benchmark.official_audio_gate import make_gate
    item=target_item();item['audio_recipe']['source'].update(sample_rate=rate,first_sample=10*rate,last_sample=11*rate)
    for spans in [[],[[-1.,.01]],[[.2,.4]],[[.1,.11],[.15,.8]],[[.9,2.]],[[0.,1.]]]:
        item['audio_recipe']['official_listener']['words']=[dict(start=a,end=b) for a,b in spans]
        _,info=make_gate(item['audio_recipe']['official_listener'],rate,rate)
        item['listener_audio_identity']['positive_samples']=info['positive_samples']
        assert current_audio_support(item)==[[a/rate,b/rate] for a,b in info['positive_intervals_samples']]


def test_prompt_millisecond_rounding_is_clipped_but_wrong_clock_is_rejected():
    from p2_ltx_events.labels import clip_cue_interval
    assert clip_cue_interval(dict(start=5.262,end=5.652),5.6515,'real-case')==(5.262,5.6515)
    with pytest.raises(ValueError,match='outside'):clip_cue_interval(dict(start=5.262,end=6.),5.6515,'wrong-clock')


def test_visual_laughter_is_not_audio_and_categories_overlap():
    item=target_item();old=dict(sample_id='old',source=item['source'],duration_s=1.,
        raw_completion_status='complete_candidates',events=[dict(start=.1,end=.4,modality='visual',categories=['laugh','smile'])])
    normalized=normalize_old(old)
    assert normalized['classes']['laugh']['spans'] and normalized['classes']['smile']['spans']
    assert not normalized['classes']['listener_vocalization']['spans']


def test_current_physical_brow_cues_do_not_become_frown_or_surprise():
    item=target_item()
    production=dict(nod_coverage=[[0.,1.]],vocal_spans=[[.2,.4]],classes={
        'brow_lowering':dict(spans=[[.1,.3]],coverage=[[0.,1.]],unknown=[]),
        'brows_raised':dict(spans=[[.5,.8]],coverage=[[0.,1.]],unknown=[])})
    row=join_target(item,[],None,production)
    assert row['classes']['brow_lowering']['spans'] and row['classes']['brows_raised']['spans']
    assert not row['classes']['frown']['spans'] and not row['classes']['surprise']['spans']
    assert row['classes']['listener_vocalization']['spans']==[[.2,.4]]
    assert row['classes']['listener_vocalization']['coverage']


def test_restored_video_joins_original_time_only_with_same_listener_and_clock():
    item=target_item();old=copy.deepcopy(item['source'])
    item['historical_source']=old;item['source']=copy.deepcopy(old)
    item['source']['source']['archive']='/derived/upright.tar'
    label=dict(sample_id='old',source=old,duration_s=1.,raw_completion_status='complete_candidates',
        events=[dict(start=.2,end=.4,modality='visual',categories=['smile'])])
    row=join_target(item,[normalize_old(label)])
    torch.testing.assert_close(torch.tensor(row['classes']['smile']['spans']),torch.tensor([[.2,.4]]))
    item['historical_source']['source_start_s']+=.1
    with pytest.raises(ValueError,match='Restored'):join_target(item,[])


def test_sample_accurate_duration_wins_over_rounded_source_endpoints():
    item=target_item();item['source']['source_end_s']=11.00001
    old=dict(sample_id='old',source=item['source'],duration_s=1.,raw_completion_status='complete_candidates',
        events=[dict(start=.2,end=1.000005,modality='visual',categories=['smile'])])
    row=join_target(item,[normalize_old(old)])
    assert row['duration_s']==1.
    torch.testing.assert_close(torch.tensor(row['classes']['smile']['spans']),torch.tensor([[.2,1.]]))
    assert all(0<=a<b<=1. for fields in row['classes'].values() for spans in fields.values() for a,b in spans)


def test_sum_reduction_matches_global_objective_with_uneven_known_counts():
    # Simulates SUM reduction across ranks and two local microbatches; masks
    # differ by rank, so averaging local means would produce a wrong gradient.
    p=torch.tensor(.3,requires_grad=True);names=['clip/a','temporal/b']
    counts={'clip/a':3.,'temporal/b':7.}
    c=dict(loss_weights=dict(av=1.,event=.05,video_feature=.025,audio_feature=.025),
           hidden_event_supervision=dict(clip_weight=1.,temporal_weight=1.))
    sums=[]
    for rank in range(2):
        for micro in range(2):
            value=dict(av=p.square()*(rank+micro+1),features={'video':p.square(),'audio':p.square()},
                event_sums={'clip/a':p.square()*(rank+1),'temporal/b':p.square()*(micro+1)})
            sums.append(objective(value,c,counts,2,2)[0])
    gradient=torch.autograd.grad(sum(sums),p)[0]
    expected=p.square()*(2.+.05)+.05*p.square()*(6/3+6/7)
    torch.testing.assert_close(gradient,torch.autograd.grad(expected,p)[0])


def test_checkpoint_roundtrip_and_old_schema_rejected(tmp_path):
    from p2_ltx_events import checkpoint
    from p2_listen.training import rng_state
    c,model,bundle,_,strategy,sampler=build();opened(model)
    optimizer=torch.optim.AdamW(model.groups());value=forward(model,bundle,strategy,sampler)
    counts={k:v['known'].sum() for k,v in bundle['events'].items()}
    objective(value,c,counts)[0].backward();optimizer.step();optimizer.zero_grad(set_to_none=True)
    path=tmp_path/'step.pt';identity={'config':c,'world':1}
    checkpoint.save(path,model,optimizer,1,identity,[rng_state()])
    before={k:v.clone() for k,v in model.state_dict().items()}
    with torch.no_grad():next(model.parameters()).add_(1)
    assert checkpoint.resume(path,model,optimizer,identity)==1
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in before.items())
    old=copy.deepcopy(identity);old['world']=2
    with pytest.raises(ValueError,match='identity'):checkpoint.resume(path,model,optimizer,old)
    from p2_listen.contracts import atomic_json,sha
    saved=torch.load(path,weights_only=True);saved['schema']='p2-qwen-whisper-event-feature-v1'
    torch.save(saved,path);atomic_json(path.with_suffix('.complete.json'),dict(sha256=sha(path),step=1))
    with pytest.raises(ValueError,match='complete'):checkpoint.checked(path)


def test_resume_reproduces_the_next_native_joint_update(tmp_path):
    from p2_ltx_events import checkpoint
    from p2_listen.training import rng_state
    c,model,bundle,_,strategy,sampler=build();optimizer=torch.optim.AdamW(model.groups())
    counts={k:v['known'].sum() for k,v in bundle['events'].items()}
    def update():
        optimizer.zero_grad(set_to_none=True);values=forward(model,bundle,strategy,sampler)
        objective(values,c,counts)[0].backward();optimizer.step()
    update();path=tmp_path/'step.pt';identity={'native_joint_resume':True}
    checkpoint.save(path,model,optimizer,1,identity,[rng_state()])
    update();expected={k:v.clone() for k,v in model.state_dict().items()}
    state=copy.deepcopy(optimizer.state_dict())
    checkpoint.resume(path,model,optimizer,identity);update()
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in expected.items())
    for key,value in state['state'].items():
        for name,tensor in value.items():assert torch.equal(tensor,optimizer.state_dict()['state'][key][name])


def test_real_sqlite_export_roundtrip_and_tamper_rejection(tmp_path):
    from p2_ltx_events.labels import index_history,prepare,Reader
    from p2_listen.contracts import digest
    item=target_item();cache=tmp_path/'cache';(cache/'samples').mkdir(parents=True)
    caption=tmp_path/'caption.json'
    caption.write_text(json.dumps(dict(sample_id='new',source=item['source'],input_evidence=dict(nod_supplement=dict(
        sample_id='new',fps=25,profile='recall',events=[dict(start=.6,end=.8,id='nod-1')],
        detector_file='/frozen/detector.json',detector_sha256='frozen_sha')))))
    item['statistics']=dict(caption_file=str(caption),caption_file_sha256=hashlib.sha256(caption.read_bytes()).hexdigest())
    (cache/'samples/new.json').write_text(json.dumps(item))
    (cache/'manifest.json').write_text(json.dumps(dict(splits={'train':['new']},entries={'new':digest(item)})))
    source=tmp_path/'old';(source/'integrated').mkdir(parents=True)
    old=dict(sample_id='old',source=item['source'],duration_s=1.,raw_completion_status='complete_candidates',
             events=[dict(start=.2,end=.4,modality='visual',categories=['head_nod'])])
    (source/'integrated/one.jsonl').write_text(json.dumps(old)+'\n')
    history=tmp_path/'history';index_history(source,history,1)
    output=tmp_path/'labels';report=prepare(cache,history,output,1)
    reader=Reader(output);assert reader.ids()==['new'];row=reader.row('new')
    assert row['nod_source']['detector_sha256']=='frozen_sha'
    assert len(row['classes']['head_nod']['spans'])==2
    assert report['full_train'] and report['coverage']['with_production_nod']==1
    assert not report['human_ground_truth'];reader.conn.close()
    projected=tmp_path/'projected';prepare(cache,history,projected,1,evidence_from=output)
    other=Reader(projected);assert other.row('new')==row;other.conn.close()
    with (output/'labels.sqlite').open('ab') as stream:stream.write(b'tampered')
    with pytest.raises(ValueError,match='changed'):Reader(output)


def test_canary_snapshot_restores_trainables_aliases_and_buffers_without_copying_frozen_base():
    from p2_ltx_events.training import canary_snapshot,restore_canary
    model=torch.nn.Module()
    model.base=torch.nn.Linear(32,32);model.base.requires_grad_(False)
    model.adapter=torch.nn.Linear(32,4);model.alias=model.adapter
    model.register_buffer('counter',torch.tensor(7))
    original={k:v.clone() for k,v in model.state_dict().items()}
    snapshot=canary_snapshot(model)
    assert not any(k.startswith('base.') for k in snapshot)
    assert {'adapter.weight','alias.weight','counter'}<=set(snapshot)
    with torch.no_grad():
        model.adapter.weight.add_(1);model.adapter.bias.add_(2);model.counter.add_(3)
    restore_canary(model,snapshot)
    assert all(torch.equal(v,original[k]) for k,v in model.state_dict().items())

@pytest.mark.parametrize('nodes',[3,4])
def test_cluster_commands_resolve_remote_config_and_resume_paths(nodes):
    import runpy
    cli=runpy.run_path(str(ROOT/'scripts/p2_ltx_events.py'),run_name='cluster_path_test')
    cluster=read(ROOT/'configs/p2_ltx_hidden_clip_32gpu_node12_8_4_11_20261106.json')
    cluster['nodes']=cluster['nodes'][:nodes]
    commands=cli['commands'](cluster,Path('relative_config.json'),Path('relative_resume.pt'),5)
    assert len(commands)==nodes
    for rank,command in enumerate(commands):
        assert Path(command[command.index('--config')+1]).is_absolute()
        assert Path(command[command.index('--resume')+1]).is_absolute()
        assert '--node_rank='+str(rank) in command
        assert '--nnodes='+str(nodes) in command and '--nproc_per_node=8' in command


def test_single_gpu_commands_isolate_physical_gpu6_and_reset_distributed_environment():
    import runpy
    cli=runpy.run_path(str(ROOT/'scripts/p2_ltx_events.py'),run_name='single_gpu_test')
    cluster=read(ROOT/'configs/p2_ltx_hidden_clip_32gpu_node12_8_4_11_20261106.json')
    node=cluster['nodes'][3];node['gpus']=[6];cluster['nodes']=[node]
    result=cli['commands'](cluster,Path('relative.json'))
    assert len(result)==1
    cmd=result[0]
    assert 'CUDA_VISIBLE_DEVICES=6' in cmd
    assert all(v in cmd for v in ('RANK=0','WORLD_SIZE=1','LOCAL_RANK=0'))
    assert 'torch.distributed.run' not in cmd and Path(cmd[cmd.index('--config')+1]).is_absolute()


def test_checkpoint50_changes_only_save_cadence():
    from p2_ltx_events.config import DEFAULT
    c=read(DEFAULT);parent=read(c['hidden_event_parent_config']);c['training']['save_every']=50
    validate_baseline(c,parent)
    c['training']['connector_lr']*=2
    with pytest.raises(ValueError):validate_baseline(c,parent)


def test_single_node_two_gpus_use_two_ranks_and_preserve_batch_partition():
    import runpy
    cli=runpy.run_path(str(ROOT/'scripts/p2_ltx_events.py'),run_name='two_gpu_test')
    cluster=read(ROOT/'configs/p2_ltx_hidden_clip_32gpu_node12_8_4_11_20261106.json')
    node=cluster['nodes'][3];node['gpus']=[6,7];cluster['nodes']=[node]
    command=cli['commands'](cluster,Path('config.json'),Path('resume.pt'))[0]
    assert 'CUDA_VISIBLE_DEVICES=6,7' in command and '--nproc_per_node=2' in command
    assert 'WORLD_SIZE=1' not in command and '--nnodes=1' in command
    assert '--resume' in command
