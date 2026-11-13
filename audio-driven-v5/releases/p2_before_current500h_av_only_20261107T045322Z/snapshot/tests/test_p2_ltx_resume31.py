"""Cadence boundaries and exact sample/noise continuation across world sizes."""
import copy,random
from collections import Counter
import pytest
from p2_ltx_events.resume31 import checkpoint_due,remaining_plan,select_indices,validate_checkpoint_schedule


def config():
    return dict(training=dict(save_every=500),
                checkpoint_schedule=dict(threshold_step=5500,before_every=1000,after_every=500))


def test_all_cadence_boundaries_and_final_partial_save():
    c=config();validate_checkpoint_schedule(c)
    assert [s for s in range(1,9044) if checkpoint_due(c,s)]==[
        1000,2000,3000,4000,5000,5500,6000,6500,7000,7500,8000,8500,9000]
    assert not checkpoint_due(c,5499) and checkpoint_due(c,5500)
    assert not checkpoint_due(c,9043)  # native final-save branch must save it


def test_legacy_cadence_unchanged():
    c=dict(training=dict(save_every=300))
    assert all(checkpoint_due(c,s)==(s%300==0) for s in range(1,9000))


@pytest.mark.parametrize('key,value',[('threshold_step',True),('before_every',0),('after_every',-1),('threshold_step',5501)])
def test_invalid_cadence_is_rejected(key,value):
    c=config();c['checkpoint_schedule'][key]=value
    with pytest.raises(ValueError):validate_checkpoint_schedule(c)


def test_exact_parent_suffix_noise_order_and_all31_rank_shards():
    pairs=[(i%103,i) for i in range(32*23)]
    random.Random(7).shuffle(pairs)
    plan=remaining_plan(pairs,7,32,31);actual=[]
    for step in range(8,8+len(plan['pairs'])//31):
        for rank in range(31):
            actual.extend(select_indices(plan['pairs'],step,7,31,rank,31))
    remaining=pairs[7*32:]
    assert actual[:len(remaining)]==remaining
    assert len(actual)==len(remaining)+plan['padding']
    assert len(actual)%31==0 and plan['padding']<31
    assert [p[1] for p in actual[len(remaining):]]==list(range(736,736+plan['padding']))
    assert len({p[1] for p in actual})==len(actual)


@pytest.mark.parametrize('step,rank,world',[(7,0,31),(100,0,31),(8,31,31),(8,0,32)])
def test_no_replay_or_out_of_range_rank(step,rank,world):
    plan=remaining_plan([(i,i) for i in range(320)],7,32,31)
    with pytest.raises(ValueError):select_indices(plan['pairs'],step,7,31,rank,world)


def test_completed_parent_cannot_restart_an_epoch():
    with pytest.raises(ValueError):remaining_plan([(i,i) for i in range(320)],10,32,31)


def test_reserved_gpu_and_rank_layout():
    from p2_ltx_resume31 import layout
    c=dict(nodes=[dict(name=n,gpus=list(range(7 if n=='node11' else 8)))
                  for n in ['node12','node8','node4','node11']])
    entries=layout(c)
    assert [r for n in entries for r in n['ranks']]==list(range(31))
    assert entries[-1]['gpus']==list(range(7)) and entries[-1]['ranks']==list(range(24,31))
    bad=copy.deepcopy(c);bad['nodes'][-1]['gpus'].append(7)
    with pytest.raises(ValueError):layout(bad)


def test_duration_rebatch_preserves_pairs_and_avoids_crossed_long_batches():
    import statistics
    pairs=[(i,i) for i in range(32*40)]
    durations=[1. if (i//32)%2==0 else 10. for i in range(len(pairs))]
    flat=remaining_plan(pairs,2,32,31)
    plan=remaining_plan(pairs,2,32,31,durations,123)
    assert Counter(map(tuple,plan['pairs']))==Counter(map(tuple,flat['pairs']))
    assert plan==remaining_plan(pairs,2,32,31,durations,123)
    def average_max(p):
        return statistics.mean(max(durations[i] for i,_ in p[k:k+31]) for k in range(0,len(p),31))
    assert average_max(plan['pairs'])<average_max(flat['pairs'])
    assert plan['window']==496
