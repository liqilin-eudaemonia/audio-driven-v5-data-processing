import importlib.util
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[1] / 'scripts/p2_j_event_feature_continue24.py'
spec = importlib.util.spec_from_file_location('event_feature_continue24', PATH)
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


def test_production_budget_preserves_every_unconsumed_real_ordinal():
    n = 285638
    pairs = [(i % n, i) for i in range(10202 * 28)]
    durations = [1 + (i % 111) / 10 for i in range(n)]
    remaining, proof = m.remaining_plan(pairs, n, 5500, 28, 24, durations, 20261023)
    real = [p for p in remaining if p[1] < n]
    assert sorted(real) == pairs[154000:n]
    assert proof['consumed_real'] == 154000
    assert proof['remaining_real'] == 131638
    assert proof['final_padding'] == 2
    assert proof['removed_parent_padding'] == 16
    assert 5500 + proof['remaining_steps'] == 10985
    assert len(remaining) == proof['remaining_steps'] * 24
    again, _ = m.remaining_plan(pairs, n, 5500, 28, 24, durations, 20261023)
    assert again == remaining


def test_all_24_ranks_get_exactly_one_sample_per_update():
    n = 100
    pairs = [(i % n, i) for i in range(112)]
    remaining, proof = m.remaining_plan(pairs, n, 1, 28, 24, [float(i + 1) for i in range(n)], 3)
    schedule = m.RemainingSchedule.__new__(m.RemainingSchedule)
    schedule.pairs = remaining; schedule.size = n; schedule.prefix = 1; schedule.seed = 3
    observed = []
    for step in range(2, 2 + proof['remaining_steps']):
        for rank in range(24):
            one = schedule.indices(n, step, 24, 3, rank, 24)
            assert len(one) == 1
            observed.extend(one)
    assert observed == remaining
    with pytest.raises(ValueError, match='24 ranks'):
        schedule.indices(n, 2, 24, 3, 0, 28)


def test_dropped_real_sample_is_rejected():
    pairs = [(i % 100, i) for i in range(112)]
    del pairs[50]
    with pytest.raises(ValueError, match='lost or duplicated'):
        m.remaining_plan(pairs, 100, 1, 28, 24, [1.] * 100, 3)
