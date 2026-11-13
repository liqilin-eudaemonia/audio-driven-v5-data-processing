import copy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from p2_matrix16.config import load,validate
from p2_matrix16.training import objective,gradient_audit
from p2_listen.contracts import ROOT,atomic_json,sha
from p2_clip_events.backend import teacher_identity,Dataset
from test_p2_matrix16 import setup,forward


def config():return load(ROOT/'configs/p2_j_guided_av_feature_base025_20261102.json',formal=True)


def test_exact_coefficients_and_frozen_base_source():
    c=config()
    assert c['loss_weights']==dict(av=1.,event=.05,video_feature=.025,audio_feature=.025)
    assert c['qwen_hidden_source']=='base_pre_sft' and c['stage']==1 and not c['qwen_lora']['enabled']
    assert c['j_cache']==c['plan_cache'] and 'sft_hidden' not in c['j_cache']
    assert c['event_head_training'] is False
    # Numeric identity: .025*(4+6)=.05*mean(4,6), not .05 per branch.
    v=dict(av=torch.tensor(2.),features=dict(video=torch.tensor(4.,requires_grad=True),audio=torch.tensor(6.,requires_grad=True)),
           event_sums={'a':torch.tensor(0.)})
    total,base,_=objective(v,c,{'a':torch.tensor(1)},1,1)
    assert total.item()==base.item()==2.25
    total.backward()
    for x in v['features'].values():assert x.grad.item()==pytest.approx(.025)


def test_legacy_coefficients_remain_loadable():
    c=config();c['loss_weights'].update(video_feature=.1,audio_feature=.1);validate(c)


@pytest.mark.parametrize('bad',[float('nan'),float('inf'),True,-.025,.05])
def test_unapproved_or_nonfinite_coefficients_rejected(bad):
    c=config();c['loss_weights']['video_feature']=bad
    with pytest.raises(ValueError):validate(c)


def test_feature_updates_connector_without_event_probe():
    c=config();c.pop('event_supervision')
    c['connector']=dict(fusion_width=16,fusion_depth=2,fusion_heads=4)
    model,bundle,pref,av=setup(c)
    with patch.object(model.event_heads,'forward',side_effect=AssertionError('Event head must not run')):
        values=forward(model,bundle,pref,av)
    norms=gradient_audit(model,values,c,values['event_counts'])
    assert norms['connector']['feature']>0 and norms['connector']['av']>0
    assert norms['connector']['event']==0 and norms['text_lora']['feature']==0
    assert all(not p.requires_grad for p in model.event_heads.parameters())
    total,_,_=objective(values,c,values['event_counts'],1,1);total.backward()
    assert all(p.grad is None for p in model.event_heads.parameters())


def test_teacher_completion_and_source_are_required(tmp_path):
    c=config();c['teacher_cache']=str(tmp_path);source=c['j_cache']
    atomic_json(tmp_path/'protocol.json',dict(source=source,samples=2))
    row=dict(source=source,samples=2,success=True,protocol_sha256=sha(tmp_path/'protocol.json'))
    atomic_json(tmp_path/'COMPLETE.json',row);assert teacher_identity(c)['path']==str(tmp_path)
    for change in [dict(source='/wrong'),dict(samples=1),dict(success=False),dict(protocol_sha256='wrong')]:
        atomic_json(tmp_path/'COMPLETE.json',dict(row,**change))
        with pytest.raises(ValueError):teacher_identity(c)


def test_reader_keeps_original_h_and_adds_teacher():
    ds=Dataset.__new__(Dataset);ds.spec=config();ds.rows=[dict(sample_id='s')];ds.cfg=SimpleNamespace(target_dim=2048)
    ds.names=['visual:nodding'];ds.labels=SimpleNamespace(row=lambda _:dict(source_digest='same'),manifest=dict(absence_weight=.1))
    old_h=object();calls=[]
    def read_original(sid,variant,device):
        calls.append((sid,variant));return dict(condition=old_h,layout={},text={},av={},support={})
    ds.reader=SimpleNamespace(manifest=dict(entries=dict(s='same')),training_sample=read_original)
    teacher={k:torch.tensor([1.]) for k in ['video','audio','video_valid','audio_valid']}
    ds.teacher_reader=SimpleNamespace(load=lambda *args:teacher)
    with patch('p2_clip_events.backend.tensors',return_value={}),patch('p2_j_teacher.core.validate_geometry') as geometry:
        result=ds.sample(0)
    assert result['condition'] is old_h and calls==[('s','guided')]
    assert result['teacher']['video'] is teacher['video'];geometry.assert_called_once()
