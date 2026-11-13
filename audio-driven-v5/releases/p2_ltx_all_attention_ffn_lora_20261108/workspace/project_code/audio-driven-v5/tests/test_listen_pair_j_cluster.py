import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from listen_pair_j_queue import Queue, Guard, lock, read, write, sha, metrics, shard
from run_listen_pair_j_cluster import TrainingLog, activity


def make_queue(tmp_path, count=9):
    write(tmp_path / 'manifest.json', dict(contract='test', max_attempts=3,
          nodes=[dict(name='a'), dict(name='b'), dict(name='c'), dict(name='d')],
          items=[dict(pair='pair_' + str(i)) for i in range(count)]))
    q = Queue(tmp_path); q.initialize(); return q


def commit(q, pair):
    p = q.work / 'results' / pair / 'result.json'
    write(p, dict(complete=True))
    write(p.parent / 'complete.json', dict(contract='test', result_sha256=sha(p)))


def crash_claim(path, pipe):
    q = Queue(path); pair, handle = q.claim('a'); pipe.send(pair)
    os._exit(0)


def consumer(path, pipe):
    q = Queue(path); result = []
    while claim := q.claim('b'):
        pair, handle = claim
        time.sleep(.01); commit(q, pair); q.finish(pair); handle.close(); result.append(pair)
    pipe.send(result)


def test_live_claim_is_skipped_without_deadlock_and_completion_once(tmp_path):
    q = make_queue(tmp_path, 2); pair, handle = q.claim('a')
    other, other_handle = q.claim('b'); assert pair != other
    commit(q, pair); q.finish(pair); q.finish(pair)
    handle.close(); other_handle.close()
    assert read(q.path)['completed'] == 1


def test_crashed_owner_recovered_and_own_shard_order(tmp_path):
    q = make_queue(tmp_path)
    a, b = mp.Pipe(); p = mp.Process(target=crash_claim, args=(tmp_path, b)); p.start()
    pair = a.recv(); p.join(5); assert p.exitcode == 0
    reclaimed, handle = q.claim('d'); assert reclaimed == pair
    assert shard(pair, 4) == 0 or not q.buckets[0]
    handle.close()


def test_commit_before_queue_finish_recovers_without_recompute(tmp_path):
    q = make_queue(tmp_path, 1); pair, handle = q.claim('a'); commit(q, pair); handle.close()
    assert q.claim('b') is None
    assert read(q.path)['completed'] == 1


def test_multi_process_work_is_unique(tmp_path):
    q = make_queue(tmp_path, 20); processes=[]; pipes=[]
    for _ in range(4):
        a,b=mp.Pipe(); p=mp.Process(target=consumer,args=(tmp_path,b));p.start();processes.append(p);pipes.append(a)
    values=[x for pipe in pipes for x in pipe.recv()]
    for p in processes:p.join(10);assert p.exitcode==0
    assert len(values)==len(set(values))==20
    assert read(q.path)['completed']==20


def test_three_failures_are_explicit_not_completed(tmp_path):
    q=make_queue(tmp_path,1)
    for _ in range(3):
        pair,handle=q.claim('a');q.finish(pair,'bad input');handle.close()
    assert q.claim('b') is None
    assert read(q.path)['completed']==0 and len(read(q.path)['failed'])==1


def test_corrupt_completed_result_is_not_skipped(tmp_path):
    q=make_queue(tmp_path,1);pair,handle=q.claim('a');commit(q,pair);handle.close()
    write(q.work/'results'/pair/'result.json',dict(broken=True))
    with pytest.raises(ValueError,match='checksum'):q.claim('b')


def timing(start,n,value):
    return [dict(step=i,seconds=value,between_batches_seconds=.01) for i in range(start,start+n)]


def test_guard_requires_two_nonoverlapping_bad_windows():
    g=Guard(metrics(timing(0,100,2.)),window=40)
    assert not g.add(timing(100,40,2.2))
    assert not g.add(timing(140,40,2.))
    assert not g.add(timing(180,40,2.2))
    assert g.add(timing(220,40,2.2))
    with pytest.raises(ValueError):g.add(timing(259,1,2.))


def test_idle_workers_cannot_establish_no_slowdown():
    values=[dict(at=i,workers={'a':dict(stage='idle',heartbeat=i,pair=None)}) for i in range(10)]
    assert activity(values,['a','b'],0)=={'a':0.,'b':0.}
    for v in values:v['workers']['a'].update(stage='screening',pair='x')
    assert activity(values,['a','b'],0)=={'a':1.,'b':0.}
    for v in values:v['workers']['a']['heartbeat']=-100
    assert activity(values,['a'],0)=={'a':0.}


def test_training_tail_dedup_phase_and_truncation(tmp_path):
    path=tmp_path/'train.log'
    line=lambda i:'TRAIN_PROGRESS '+json.dumps(dict(step=i,seconds=2.,between_batches_seconds=.01,warmup=False))+'\n'
    path.write_text(line(1)+'Validation: running\n'+line(2)+line(3))
    reader=TrainingLog(path);rows,_=reader.fetch();assert [r['step'] for r in rows]==[1,3]
    assert reader.fetch()[0]==[]
    with path.open('a') as f:f.write(line(4))
    assert reader.fetch()[0][0]['step']==4
    path.write_text(line(5))
    with pytest.raises(ValueError,match='truncated'):reader.fetch()
