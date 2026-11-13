from unittest.mock import patch
import pytest
from p2_testsets_eval.protocol import ROOT,read,rows
from p2_testsets_eval.runtime import planner_identity,checkpoint_source

WORK=ROOT/'experiments/p2_response200_feature_sft_event_20261103'


def test_fixed200_and_four_checkpoints():
    p=read(WORK/'protocol.json')
    original=rows(ROOT/'experiments/responsenet_initial200_20261102/references.jsonl')
    assert rows(WORK/'references/references.jsonl')==original
    assert p['total_samples']==200 and p['total_outputs']==1000 and len(p['arms'])==5
    assert [a['step'] for a in p['arms']]==[6000,8927,6000,8927,6000]
    flat=[r for i in range(32) for r in read(WORK/'shards'/f'{i:02d}.json')]
    assert len(flat)==200 and {r['id'] for r in flat}=={r['id'] for r in original}
    for r in flat:assert (r['geometry']['width'],r['geometry']['height'])==(512,512)


def test_planner_is_training_matched_and_changed_weights_rejected():
    base=read(WORK/'base_feature/protocol.json');sft=read(WORK/'sft_event/protocol.json')
    assert planner_identity(base) is None
    assert sft['planner_adapter'] and planner_identity(sft)==sft['planner_adapter_identity']
    with patch('p2_j_sft_hidden.adapter_identity',return_value={'wrong':True}):
        with pytest.raises(ValueError):planner_identity(sft)


def test_inference_source_matches_both_training_runs():
    for group in ('base_feature','sft_event'):
        p=read(WORK/group/'protocol.json');arm=p['arms'][0]
        identity=read(__import__('pathlib').Path(arm['checkpoint']).parent/'run.json')
        assert checkpoint_source(identity['experiment'],arm['kind'])==identity['source']


def test_cross_arm_shares_exact_sft_condition_directory():
    from p2_testsets_eval.runtime import verify_checkpoint_source
    from pathlib import Path
    child=WORK/'sft_on_base_event';p=read(child/'protocol.json');sft=read(WORK/'sft_event/protocol.json')
    assert (child/'conditions').resolve()==(WORK/'sft_event/conditions').resolve()
    assert (child/'negative').resolve()==(WORK/'sft_event/negative').resolve()
    assert p['planner_adapter_identity']==sft['planner_adapter_identity']
    arm=p['arms'][0];identity=read(Path(arm['checkpoint']).parent/'run.json')
    verify_checkpoint_source(identity,identity['experiment'],arm)
    from copy import deepcopy
    changed=deepcopy(identity);changed['source']['tampered']='wrong'
    with pytest.raises(ValueError):verify_checkpoint_source(changed,changed['experiment'],arm)


def test_metrics_controller_can_be_adopted(tmp_path):
    import os
    from p2_response200_fourway import JoinedMetrics
    assert JoinedMetrics(os.getpid(),tmp_path).poll() is None
    assert JoinedMetrics(-1,tmp_path).poll()==1
    (tmp_path/'COMPLETE.json').write_text('{}')
    assert JoinedMetrics(-1,tmp_path).poll()==0
