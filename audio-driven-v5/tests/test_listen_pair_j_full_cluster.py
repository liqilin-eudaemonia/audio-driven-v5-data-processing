import importlib.util
from pathlib import Path
import sys

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
from run_listen_pair_j_full_cluster import required_workers, key


def test_deadline_capacity_has_headroom_and_rounds_up():
    # The 32-source measured mean makes four workers infeasible; reserve 15%.
    count = required_workers(64622, 1 / 37.516929221751184, 8 * 3600)
    assert count == 100
    assert required_workers(64622, 1 / 37.516929221751184, 7 * 3600) > count


def test_missing_rate_and_overdue_are_not_reported_as_instant_completion():
    assert required_workers(10, 0, 100) is None
    assert required_workers(10, 1, -1) is None
    assert required_workers(0, 0, 0) == 0


def test_node_slots_cannot_collide():
    assert len({key(node, slot) for node in ['node8', 'node4', 'node17', 'node12']
                for slot in range(32)}) == 128
