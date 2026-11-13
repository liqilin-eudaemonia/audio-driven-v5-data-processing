import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from p2_testsets_eval.metrics import scoring_stopped,acquire_queue_or_observe_completion


def test_only_explicit_owned_handoff_is_ignored(tmp_path):
 p=dict(scoring_handoff_stop_reason='own safe handoff')
 assert not scoring_stopped(tmp_path,p)
 (tmp_path/'STOP').write_text(json.dumps(dict(reason='own safe handoff')))
 assert not scoring_stopped(tmp_path,p)
 assert scoring_stopped(tmp_path,{})
 (tmp_path/'STOP').write_text(json.dumps(dict(reason='user requested stop')))
 assert scoring_stopped(tmp_path,p)
 for value in ['stop','[]','"stop"']:
  (tmp_path/'STOP').write_text(value)
  assert scoring_stopped(tmp_path,p)


def test_scorer_can_lock_during_its_declared_handoff(tmp_path):
 p=dict(scoring_handoff_stop_reason='own safe handoff')
 (tmp_path/'protocol.json').write_text(json.dumps(p));(tmp_path/'STOP').write_text(json.dumps(dict(reason='own safe handoff')))
 with (tmp_path/'queue.lock').open('a') as lock:assert acquire_queue_or_observe_completion(tmp_path,lock)
 (tmp_path/'STOP').write_text('user')
 with (tmp_path/'another.lock').open('a') as lock:assert not acquire_queue_or_observe_completion(tmp_path,lock)
