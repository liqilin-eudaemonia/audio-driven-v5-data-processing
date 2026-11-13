import importlib.util
from pathlib import Path
import sys
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from listen_pair_j_fast_queue import Queue
from listen_pair_j_queue import read, lock

# Exercise the exact existing crash/commit/concurrency contracts with the new queue.
spec = importlib.util.spec_from_file_location('fast_queue_contract_tests', Path(__file__).with_name('test_listen_pair_j_cluster.py'))
old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
old.Queue = Queue

@pytest.mark.parametrize('name',[
 'test_live_claim_is_skipped_without_deadlock_and_completion_once',
 'test_crashed_owner_recovered_and_own_shard_order',
 'test_commit_before_queue_finish_recovers_without_recompute',
 'test_multi_process_work_is_unique',
 'test_three_failures_are_explicit_not_completed',
 'test_corrupt_completed_result_is_not_skipped'])
def test_existing_queue_contracts(tmp_path,name):
    getattr(old,name)(tmp_path)


def test_bounded_scan_eventually_recovers_abandoned_claim(tmp_path):
    q=old.make_queue(tmp_path,40); claims=[q.claim('a') for _ in range(40)]
    abandoned, h=claims[21]; h.close()
    found=None
    for _ in range(11):
        found=q.claim('b')
        if found: break
    assert found and found[0]==abandoned
    found[1].close()
    for i,(_,h) in enumerate(claims):
        if i!=21:h.close()
