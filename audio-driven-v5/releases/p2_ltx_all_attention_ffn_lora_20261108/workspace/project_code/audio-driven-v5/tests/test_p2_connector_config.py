import pytest

from p2_matrix16.config import connector_config, validate_connector, load
from p2_listen.config import extractor_config, load as data_config
from p2_listen.contracts import ROOT, atomic_json


def test_training_override_preserves_frozen_extraction_config():
    cfg=extractor_config(data_config());original=cfg.to_dict()
    spec=dict(connector=dict(fusion_width=1024,fusion_depth=4,fusion_heads=16))
    trained=connector_config(spec,cfg)
    assert cfg.to_dict()==original
    assert trained.to_dict()==dict(original,**spec['connector'])
    assert connector_config({},cfg) is cfg


@pytest.mark.parametrize('value',[
    dict(fusion_width=1024,fusion_depth=0,fusion_heads=16),
    dict(fusion_width=1024,fusion_depth=4,fusion_heads=15),
    dict(fusion_width=True,fusion_depth=4,fusion_heads=1),
    dict(fusion_width=1024,fusion_depth=4,fusion_heads=16,hidden_size=1024),
])
def test_invalid_connector_override(value):
    with pytest.raises(ValueError):validate_connector(value)


def test_j_config_emitter_and_all_eight_production_arms(tmp_path):
    from p2_clip_events.backend import emit
    expected=dict(fusion_width=1024,fusion_depth=4,fusion_heads=16)
    folder=ROOT/'configs/p2_j_clip_events_20261101'
    paths=list(folder.glob('*_s[12].json'))
    assert len(paths)==8
    for path in paths:
        c=load(path)
        assert c['connector']==expected
        assert c['event_supervision']['mode']=='clip_multilabel'
    labels=tmp_path/'labels'
    atomic_json(labels/'manifest.json',dict(cache_work='/cache',classes=dict(video=['visual:nodding'],audio=['audio:speech']),
        statistics={k:dict(positive=5000) for k in ['visual:nodding','audio:speech']}))
    result=emit(labels,tmp_path/'configs')
    assert len(result['configs'])==8
    for path in result['configs']:assert load(path)['connector']==expected
