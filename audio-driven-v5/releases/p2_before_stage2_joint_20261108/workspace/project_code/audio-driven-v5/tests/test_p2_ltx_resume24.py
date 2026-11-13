"""Regression checks for a second migration with a nonzero parent schedule base."""
import copy
from collections import Counter
from types import SimpleNamespace
import pytest
from p2_ltx_events.resume24 import continuation_plan,select_indices,validate_resume24
from p2_ltx_resume24 import layout,commands


def test_second_migration_uses_relative_consumed_steps_and_retains_every_pair():
    parent=SimpleNamespace(base_step=3982,pairs=[(i%79,i+127424) for i in range(31*18)],
                           durations=[float(i%11+1) for i in range(79)],seed=7)
    plan=continuation_plan(parent,3987)
    expected=parent.pairs[5*31:]
    original_ordinals={o for _,o in expected}
    retained=[tuple(p) for p in plan['pairs'] if p[1] in original_ordinals]
    assert Counter(retained)==Counter(expected)
    assert len(plan['pairs'])%24==0 and plan['padding']<24
    assert len({o for _,o in plan['pairs']})==len(plan['pairs'])
    assert plan==continuation_plan(parent,3987)
    actual=[]
    for step in range(3988,3988+len(plan['pairs'])//24):
        for rank in range(24):actual+=select_indices(plan['pairs'],step,3987,rank,24,24)
    assert actual==list(map(tuple,plan['pairs']))


@pytest.mark.parametrize('step,rank,world,batch',[(3987,0,24,24),(9999,0,24,24),
    (3988,24,24,24),(3988,0,31,24),(3988,0,24,31)])
def test_wrong_rank_geometry_or_replayed_step_rejected(step,rank,world,batch):
    with pytest.raises(ValueError):select_indices([(i,i) for i in range(240)],step,3987,rank,world,batch)


def test_node11_absent_from_every_rank_and_launch_command():
    c=dict(nodes=[dict(name=n,gpus=list(range(8))) for n in ['node12','node8','node4']],
           python='python',network={})
    assert [r for node in layout(c) for r in node['ranks']]==list(range(24))
    assert len(commands(c,'config','resume','logs'))==3
    c['nodes'].append(dict(name='node11',gpus=list(range(8))))
    with pytest.raises(ValueError):layout(c)


@pytest.mark.parametrize('world,parent,batch',[(31,31,24),(24,32,24),(24,31,31)])
def test_invalid_explicit_contract_rejected(world,parent,batch):
    c=dict(training=dict(global_batch=batch),resume_execution=dict(kind='preserve-parent-suffix-v1',
        parent_config='missing',parent_config_sha256='x',resume_step=6000,manifest='x',
        manifest_sha256='x',world=world,parent_world=parent))
    with pytest.raises(ValueError):validate_resume24(c)
