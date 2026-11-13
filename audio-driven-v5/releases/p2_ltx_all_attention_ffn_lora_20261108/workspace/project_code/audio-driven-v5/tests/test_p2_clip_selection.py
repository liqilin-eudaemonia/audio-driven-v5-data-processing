import copy

import pytest
import torch

from p2_clip_events.backend import data_identity, selected_statistics
from p2_clip_events.labels import select_classes, tensors
from p2_clip_events.model import classes, validate_spec
from p2_listen.contracts import atomic_json, sha, ROOT, read
from p2_matrix16.config import emit, load
from p2_matrix16.training import objective
from test_p2_matrix16 import setup


def manifest():
    return dict(classes=dict(video=['visual:nodding','visual:rare','visual:smiling'],audio=['audio:speech']),
        statistics={k:dict(positive=p,enabled=True,pos_weight=2.) for k,p in [
            ('visual:nodding',5000),('visual:rare',4999),('visual:smiling',6000),('audio:speech',8000)]})


def test_threshold_is_inclusive_and_keeps_order_and_statistics():
    m=manifest();old=copy.deepcopy(m);groups=select_classes(m,5000)
    assert groups==dict(video=['visual:nodding','visual:smiling'],audio=['audio:speech'])
    c=dict(event_supervision=dict(mode='clip_multilabel',classes=groups,min_positive=5000))
    stats=selected_statistics(c,m)
    assert tuple(stats)==classes(c['event_supervision'])
    assert all(stats[k]==m['statistics'][k] for k in stats)
    assert m==old
    with pytest.raises(ValueError,match='No clip classes'):select_classes(m,9000)


@pytest.mark.parametrize('minimum',[0,-1,True,5000.5])
def test_invalid_threshold(minimum):
    with pytest.raises(ValueError,match='min_positive'):select_classes(manifest(),minimum)
    with pytest.raises(ValueError,match='min_positive'):
        validate_spec(dict(mode='clip_multilabel',classes=manifest()['classes'],min_positive=minimum))


def test_subset_identity_rejects_wrong_class_order_membership_and_threshold(tmp_path):
    m=manifest();cache=tmp_path/'cache';labels=tmp_path/'labels'
    atomic_json(cache/'manifest.json',{'source':'unchanged'})
    m.update(cache_manifest_sha256=sha(cache/'manifest.json'),database_sha256='same-database')
    atomic_json(labels/'manifest.json',m)
    c=dict(j_cache=str(cache),clip_labels=str(labels),event_supervision=dict(
        mode='clip_multilabel',classes=select_classes(m,5000),min_positive=5000))
    identity=data_identity(c)
    assert identity['database']=='same-database'
    for mode in ('order','membership','threshold'):
        bad=copy.deepcopy(c);spec=bad['event_supervision']
        if mode=='order':spec['classes']['video'].reverse()
        elif mode=='membership':spec['classes']['video'].append('visual:rare')
        else:spec['min_positive']=6000
        with pytest.raises(ValueError,match='identity'):data_identity(bad)
    # Old full-vocabulary configs remain explicit and readable.
    c['event_supervision']=dict(mode='clip_multilabel',classes=m['classes'])
    assert data_identity(c)['database']==identity['database']


@pytest.mark.parametrize('allow_weak',[True,False])
def test_no_selected_positive_keeps_av_gradient_and_masks_uncertainty(tmp_path,allow_weak):
    emit(tmp_path);c=load(tmp_path/'fixed_av_event_s1.json')
    c['event_supervision']=dict(mode='clip_multilabel',classes=select_classes(manifest(),5000),min_positive=5000)
    c['connector']=dict(fusion_width=16,fusion_depth=4,fusion_heads=4)
    model,bundle,_,av=setup(c);names=classes(c['event_supervision'])
    bundle['events']=tensors(dict(positive=['visual:rare'],unknown=['visual:smiling'],allow_weak=allow_weak),names,.1)
    assert not any(v['y'].item() for v in bundle['events'].values())
    assert not bundle['events']['visual:smiling']['known'].item()
    assert bundle['events']['visual:nodding']['known'].item()==allow_weak
    assert bundle['events']['visual:nodding']['weight'].item()==pytest.approx(.1)
    value=model(bundle,{k:2. for k in names},torch.Generator(),av)
    total,base,event=objective(value,c,value['event_counts'],1,1)
    assert base.item()>0 and torch.isfinite(total)
    if not allow_weak:assert event.item()==0
    total.backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.connector.parameters())


def test_production_configs_share_exact_19_class_train_selection():
    m=read(ROOT/'experiments/p2_j_clip_labels_20261101/manifest.json')
    expected=select_classes(m,5000)
    assert {k:len(v) for k,v in expected.items()}==dict(video=16,audio=3)
    for p in (ROOT/'configs/p2_j_clip_events_20261101').glob('*_s[12].json'):
        c=load(p)
        assert c['event_supervision']==dict(mode='clip_multilabel',classes=expected,min_positive=5000)
        assert len(selected_statistics(c,m))==19
