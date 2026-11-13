import json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
import monitor_listen_richness_goal as monitor
from listen_benchmark.j_annotation import write


def test_live_chain_is_not_restarted_for_stale_state_file(tmp_path,monkeypatch):
    ticks=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]
    write(tmp_path/'chain-child.json',dict(pid=os.getpid(),start_ticks=ticks))
    write(tmp_path/'chain.json',dict(stage='merging',at=1))
    monkeypatch.setattr(monitor.subprocess,'Popen',lambda *a,**kw:(_ for _ in ()).throw(AssertionError('live process restarted')))
    assert monitor.ensure_chain(tmp_path,tmp_path)=='live'


def test_dead_chain_with_failed_chunks_is_reported_not_blindly_retried(tmp_path):
    write(tmp_path/'config.json',dict(code_hashes={}))
    write(tmp_path/'queues/extract/state.json',dict(failed={'job':{'error':'bad input'}}))
    assert monitor.ensure_chain(tmp_path,tmp_path)=='blocked_failed_chunks'


def test_finish_requires_full_coverage_and_committed_queue(tmp_path):
    write(tmp_path/'config.json',dict(expected_upstream_clips=100))
    s=tmp_path/'summary.json'
    write(s,dict(total_indexed_samples=100,source_errors=[],sample_count=90))
    write(tmp_path/'complete.json',dict(total_indexed_samples=100,summary=str(s)))
    q=tmp_path/'queues/extract/state.json'
    write(q,dict(failed={},inflight={'remaining':{}},completed_clips=90));assert not monitor.finished(tmp_path)
    write(q,dict(failed={},inflight={},completed_clips=89));assert not monitor.finished(tmp_path)
    write(q,dict(failed={},inflight={},completed_clips=90));assert monitor.finished(tmp_path)
    write(s,dict(total_indexed_samples=99,source_errors=[],sample_count=90));assert not monitor.finished(tmp_path)
