import copy
from types import SimpleNamespace
from unittest.mock import patch
import pytest
import p2_j_sft_event_backend as b
from p2_listen.contracts import ROOT, read


def config():return read(ROOT/'configs/p2_j_sft_h_av_event_20261102.json')


@pytest.mark.parametrize('change',[
    {'stage':2}, {'loss':'av_feature'}, {'text_condition':'fixed'},
    {'qwen_hidden_source':'base_pre_sft'}, {'event_head_training':False},
    {'teacher_cache':'unwanted'}, {'plan_cache':'wrong'},
    {'qwen_lora':{'enabled':True}},
])
def test_invalid_source_or_recipe_rejected(change):
    c=config();c.update(change)
    with pytest.raises(ValueError):b.base_config(c)


def test_base_view_does_not_modify_formal_config():
    c=config();old=copy.deepcopy(c);v=b.base_config(c)
    assert c==old and 'qwen_hidden_source' not in v
    assert v['plan_cache']==c['j_cache']
    assert c['training']['save_every']==500 and c['duration_sampling']['start_step']==0


def test_sample_uses_new_h_without_old_h_or_teacher():
    ds=b.Dataset.__new__(b.Dataset);ds.rows=[dict(sample_id='sid')];ds.names=['audio:speech']
    ds.labels=SimpleNamespace(row=lambda _:dict(source_digest='good'),manifest=dict(absence_weight=.1))
    def forbidden(*args,**kwargs):raise AssertionError('Old H must not be loaded')
    ds.reader=SimpleNamespace(manifest=dict(entries={'sid':'good'}),training_sample=forbidden)
    new_h=object();calls=[]
    def sample(sid,device,include_teacher):
        calls.append((sid,device,include_teacher));return dict(condition=new_h,layout={},text={},av={},support={})
    ds.overlay=SimpleNamespace(training_sample=sample)
    with patch.object(b,'tensors',return_value={'labels':True}):result=ds.sample(0)
    assert result['condition'] is new_h and result['teacher'] is None
    assert calls==[('sid','cpu',False)] and result['events']=={'labels':True}


def test_changed_event_source_rejected():
    ds=b.Dataset.__new__(b.Dataset);ds.rows=[dict(sample_id='sid')]
    ds.labels=SimpleNamespace(row=lambda _:dict(source_digest='wrong'))
    ds.reader=SimpleNamespace(manifest=dict(entries={'sid':'correct'}))
    with pytest.raises(ValueError,match='source mismatch'):ds.sample(0)


@pytest.mark.parametrize('change',[{'success':False},{'samples':1},{'protocol_sha256':'wrong'},{'adapter':{'wrong':True}}])
def test_completion_mismatch_rejected(change):
    p=dict(samples=2,sample_ids=['a','b'],ids_sha256=b.digest(['a','b']),adapter={'a':1})
    fake=SimpleNamespace(work=ROOT,protocol=p,ids={'a','b'},protocol_sha='correct')
    done=dict(success=True,samples=2,protocol_sha256='correct',adapter=p['adapter']);done.update(change)
    b.overlay.cache_clear()
    with patch('p2_j_sft_hidden.Reader',return_value=fake),patch('p2_j_sft_hidden.require_ready'),patch.object(b,'read',return_value=done):
        with pytest.raises(ValueError,match='final audit'):b.overlay('fake')
    b.overlay.cache_clear()
