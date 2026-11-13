import copy
from unittest.mock import patch
import pytest
import torch
from p2_listen.config import load as old_config
from p2_listen.audit import fixture,TinyTextRenderer
from p2_listen.model import ListeningExtractor
from p2_matrix16.config import load
from p2_matrix16.model import MatrixModel
from p2_legacy_qwen_lora.backend import map_parent_state,data_config,parent_record
from p2_listen.contracts import ROOT
from test_p2_matrix16 import native_qwen

@pytest.mark.parametrize('arm',['E0','E1','E0F','E1F'])
def test_exact_historical_data_and_parameter_mapping(arm):
    c=load(ROOT/'experiments/p2_legacy50h_qwen_lora_20261027/configs'/(arm+'.json'))
    d=data_config(c);p=parent_record(c)
    assert not d['data'].get('audio_postprocess') and c['plan_cache'] is None
    assert d==p['config'] and c['training']['steps']==1268 and c['training']['global_batch']==24
    cfg,_=fixture(old_config());runtime,processor=native_qwen(cfg)
    spec={k:v for k,v in c['text_lora'].items() if k!='enabled'}
    old=ListeningExtractor(cfg,128,feature_loss_enabled='F' in arm,text_lora=spec,renderer=TinyTextRenderer())
    new=MatrixModel(c,cfg,TinyTextRenderer(),runtime,processor)
    initial=copy.deepcopy(new.qwen_lora.state_dict());state=map_parent_state(old.state_dict(),new.state_dict())
    new.load_state_dict(state,strict=True)
    for module,other in [(old.extractor.connector,new.connector),(old.event_heads,new.event_heads),(old.text_lora,new.text_lora)]:
        assert all(torch.equal(v,other.state_dict()[k]) for k,v in module.state_dict().items())
    if 'F' in arm:
        for mod in ('video','audio'):
            assert all(torch.equal(v,new.decoders[mod].state_dict()[k]) for k,v in getattr(old.extractor,mod+'_decoder').state_dict().items())
    assert all(torch.equal(v,new.qwen_lora.state_dict()[k]) for k,v in initial.items())
    assert all(not adapter.b.weight.any() for adapter in new.qwen_lora)
    wrong=dict(old.state_dict());wrong['unexpected']=torch.zeros(1)
    with pytest.raises(ValueError,match='Unmapped'):map_parent_state(wrong,new.state_dict())
    wrong=dict(old.state_dict());wrong.pop(next(iter(wrong)))
    with pytest.raises(ValueError,match='Incomplete'):map_parent_state(wrong,new.state_dict())

def test_cross_arm_and_new_data_rejected():
    c=load(ROOT/'experiments/p2_legacy50h_qwen_lora_20261027/configs/E0.json')
    bad=copy.deepcopy(c);bad['legacy_arm']='E1'
    with pytest.raises(ValueError):parent_record(bad)
    bad=copy.deepcopy(c);bad['plan_cache']='new50h'
    with pytest.raises(ValueError):parent_record(bad)
    bad=copy.deepcopy(c);bad['data_config']=str(ROOT/'experiments/p2_train50h_postprocessed_reselected_20261027/config.json')
    with pytest.raises(ValueError,match='original'):data_config(bad)
