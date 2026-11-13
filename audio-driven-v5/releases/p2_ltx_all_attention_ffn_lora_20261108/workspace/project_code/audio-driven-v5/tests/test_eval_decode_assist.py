from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
import threading

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from p2_testsets_eval.runtime import decoded_output_lock


def test_export_helpers_and_normal_workers_commit_once(tmp_path):
    out=tmp_path/'output';barrier=threading.Barrier(3);calls=[]
    def worker(_):
        # All workers observe the same initially missing commit before locking.
        assert not (out/'complete.json').exists()
        barrier.wait()
        with decoded_output_lock(out):
            if (out/'complete.json').exists():return
            calls.append(True)
            (out/'complete.json').write_text('{"complete":true}')
    with ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(worker,range(3)))
    assert len(calls)==1


def test_export_exception_releases_sample_lock(tmp_path):
    out=tmp_path/'output'
    try:
        with decoded_output_lock(out):raise RuntimeError('failed attempt')
    except RuntimeError:pass
    with decoded_output_lock(out):
        assert not (out/'complete.json').exists()
        (out/'retry').write_text('acquired')


def test_small_scoring_queue_uses_all_authorized_gpus():
    from p2_testsets_eval.metrics import balanced_metric_slots
    slots=balanced_metric_slots([('node8',8,[0,1,2,3])],{})[:24]
    assert all(slots.count(('node8',g))==6 for g in range(4))


def test_scoring_allocation_fills_idle_gpu_first_and_respects_caps():
    from p2_testsets_eval.metrics import balanced_metric_slots
    active={i:(None,'node8',i//8,None) for i in range(24)}
    slots=balanced_metric_slots([('node8',8,[0,1,2,3])],active)
    assert slots==[('node8',3)]*8
