import copy
from types import SimpleNamespace
from unittest.mock import patch
import pytest
import torch

from p2_clip_events.model import ClipEventHeads, event_sums, normalized_loss, classes
from p2_clip_events.labels import derive, tensors
from p2_matrix16.training import objective, gradient_audit
from p2_matrix16.config import emit, load
from test_p2_matrix16 import setup, forward as old_forward

SPEC = dict(mode='clip_multilabel', classes=dict(video=['visual:nodding','visual:smiling'], audio=['audio:speech']))


def label(positive=('visual:nodding','visual:smiling'), unknown=(), allow_weak=True):
    return tensors(dict(positive=positive, unknown=unknown, allow_weak=allow_weak), classes(SPEC), .1)


def test_masked_pooling_is_invariant_to_padding_permutation_and_repetition():
    torch.manual_seed(9)
    head=ClipEventHeads(4,3,SPEC)
    v,a=torch.randn(2,3,4),torch.randn(2,2,3)
    u=SimpleNamespace(video=v,audio=a,video_valid=torch.ones(2,3,dtype=torch.bool),audio_valid=torch.ones(2,2,dtype=torch.bool))
    expected=head(u)
    padded=SimpleNamespace(video=torch.cat([v[:,[2,0,1]].repeat(1,2,1),torch.full((2,1,4),float('nan'))],1),
        audio=a[:,[1,0]], video_valid=torch.tensor([[1]*6+[0]]*2,dtype=torch.bool),audio_valid=u.audio_valid)
    actual=head(padded)
    for k in expected:
        assert expected[k].shape==(2,)
        torch.testing.assert_close(expected[k],actual[k])


def test_multiple_positive_classes_and_unknowns_have_independent_gradients():
    scores={k:torch.zeros(1,requires_grad=True) for k in classes(SPEC)}
    sums,counts=event_sums(scores,label(unknown=['audio:speech']),{k:1. for k in scores})
    normalized_loss(sums,counts).backward()
    assert scores['visual:nodding'].grad<0 and scores['visual:smiling'].grad<0
    assert scores['audio:speech'].grad==0
    with pytest.raises(ValueError,match='shape'):
        event_sums({k:v[:,None] for k,v in scores.items()},label(),{k:1. for k in scores})


def test_probe_does_not_update_condition_but_joint_does():
    for mode in ['probe_only','joint_event']:
        head=ClipEventHeads(4,3,SPEC)
        v=torch.randn(1,4,4,requires_grad=True);a=torch.randn(1,3,3,requires_grad=True)
        u=SimpleNamespace(video=v,audio=a,video_valid=torch.ones(1,4,dtype=torch.bool),audio_valid=torch.ones(1,3,dtype=torch.bool))
        sums,counts=event_sums(head(u,mode=mode),label(),{k:1. for k in classes(SPEC)})
        normalized_loss(sums,counts).backward()
        assert (v.grad is not None)==(mode=='joint_event')
        assert any(p.grad is not None and p.grad.abs().sum()>0 for p in head.parameters())


def test_weighted_ddp_normalization_matches_full_batch():
    names=classes(SPEC);p={k:2. for k in names}
    x=torch.randn(2,3,requires_grad=True)
    rows=[label(),label(['audio:speech'],['visual:smiling'])]
    batched={k:{f:torch.cat([r[k][f] for r in rows]) for f in ['y','known','weight']} for k in names}
    s,n=event_sums({k:x[:,i] for i,k in enumerate(names)},batched,p)
    expected=torch.autograd.grad(normalized_loss(s,n),x,retain_graph=True)[0]
    loss=0
    for rank in range(2):
        local,_=event_sums({k:x[rank:rank+1,i] for i,k in enumerate(names)},rows[rank],p)
        loss=loss+normalized_loss(local,n,world_size=2)/2
    torch.testing.assert_close(torch.autograd.grad(loss,x)[0],expected)


def test_weak_negative_weight_is_not_cancelled_by_normalization():
    scores={k:torch.zeros(1) for k in classes(SPEC)}
    labels=label(positive=[]);weights={k:1. for k in scores}
    sums,counts=event_sums(scores,labels,weights)
    weak=normalized_loss(sums,counts)
    for value in labels.values():value['weight'].fill_(1)
    sums,counts=event_sums(scores,labels,weights)
    torch.testing.assert_close(weak,normalized_loss(sums,counts)*.1)


def test_clip_checkpoint_resume_and_stage_migration_reject_class_change(tmp_path):
    from p2_matrix16 import checkpoint
    from p2_matrix16.config import source_identity
    from p2_listen.training import rng_state
    emit(tmp_path/'config')
    c=load(tmp_path/'config/guided_av_event_s1.json');c['event_supervision']=copy.deepcopy(SPEC)
    c['connector']=dict(fusion_width=16,fusion_depth=4,fusion_heads=4)
    model,bundle,prefix,av=setup(c);optimizer=torch.optim.AdamW(model.groups())
    identity=dict(experiment=c,data_identity={'labels':'same'},source=source_identity(),parent=None)
    path=tmp_path/'checkpoint.pt';checkpoint.save(path,model,optimizer,0,identity,[rng_state()])
    assert checkpoint.resume(path,model,optimizer,identity)==0
    c2=load(tmp_path/'config/guided_av_event_s2.json');c2['event_supervision']=copy.deepcopy(SPEC)
    c2['connector']=copy.deepcopy(c['connector'])
    model2,_,_,_=setup(c2)
    assert checkpoint.initialize_stage2(path,model2,c2,identity['data_identity'])['step']==0
    # Head count changes attention semantics without changing tensor shapes.
    c2['connector']['fusion_heads']=2
    with pytest.raises(ValueError,match='Connector'):
        checkpoint.initialize_stage2(path,model2,c2,identity['data_identity'])
    c2['connector']['fusion_heads']=4
    c2['event_supervision']['classes']['video'].reverse()
    with pytest.raises(ValueError,match='supervision'):
        checkpoint.initialize_stage2(path,model2,c2,identity['data_identity'])


def test_caption_derivation_uses_accepted_evidence_and_masks_local_negations():
    item=dict(sample_id='s',caption='a caption',caption_sha256='hash',statistics=dict(sample_id='s',caption='a caption',
        counted_types=['visual:nodding','visual:smiling'],evidence_ledger=[dict(raw=dict(modality='visual'),
        canonical='smiling',disposition='not_positive')],record_status='computed',pending_behavior_count=0,
        unusable_behaviors=[],original_validation_error=None,statistics_version='v3'))
    row=derive(item)
    assert row['positive']==['visual:nodding'] and row['unknown']==['visual:smiling']
    assert row['conflicts']==['visual:smiling']
    assert tensors(row,classes(SPEC),.1)['audio:speech']['known'].item()
    item['statistics']['pending_behavior_count']=1
    row=derive(item)
    labels=tensors(row,classes(SPEC),.1)
    assert not labels['audio:speech']['known'].item()
    assert labels['visual:nodding']['known'].item()


@pytest.mark.parametrize('stage',[1,2])
@pytest.mark.parametrize('loss',['av','av_event'])
@pytest.mark.parametrize('text',['fixed','guided'])
def test_clip_supervision_through_native_connector_and_stage2_lora(tmp_path,stage,loss,text):
    emit(tmp_path);c=load(tmp_path/f'{text}_{loss}_s{stage}.json');c['event_supervision']=copy.deepcopy(SPEC)
    c['connector']=dict(fusion_width=16,fusion_depth=4,fusion_heads=4)
    model,bundle,prefix,av=setup(c);bundle['events']=label()
    assert model.connector.memory.projection.out_features==16
    assert len(model.connector.blocks)==4
    assert all(b.cross.num_heads==4 and b.self_attn.num_heads==4 for b in model.connector.blocks)
    with patch('p3_listen.features.prefix',return_value=prefix),patch('p2_listen.features.make_observation',side_effect=lambda x:x),patch('p2_listen.features.image_grid',return_value=(None,None,(1,1))):
        values=model(bundle,{k:1. for k in classes(SPEC)},torch.Generator().manual_seed(19),av)
    audit=gradient_audit(model,values,c,values['event_counts'])
    assert (audit['connector']['event']>0)==(loss=='av_event')
    assert audit['text_lora']['event']==0
    if stage==2:assert (audit['qwen_lora']['event']>0)==(loss=='av_event')
    optimizer=torch.optim.AdamW(model.groups())
    objective(values,c,values['event_counts'],1,1)[0].backward();optimizer.step()
    assert all(torch.isfinite(p).all() for p in model.parameters())
