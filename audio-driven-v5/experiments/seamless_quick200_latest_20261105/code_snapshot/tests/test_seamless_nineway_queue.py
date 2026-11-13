import json
from pathlib import Path
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src'),str(ROOT.parent/'OmniResponse')]
from p2_seamless200_nineway import compatibility,owned_workers_active


def test_only_previously_verified_exact_source_changes_are_allowed(tmp_path):
    path=tmp_path/'step.pt';path.with_suffix('.complete.json').write_text(json.dumps(dict(sha256='f'*64)))
    old={'generation':'same','training':'old'};current={'generation':'same','training':'new'}
    proof=dict(checkpoint_source=old,current_source=current,inference_math_unchanged=True)
    assert compatibility(old,current,path,proof)['checkpoint_sha256']=='f'*64
    with pytest.raises(ValueError,match='Unverified'):
        compatibility(old,dict(current,generation='changed'),path,proof)
    with pytest.raises(ValueError,match='Unverified'):
        compatibility(old,dict(current,training='unreviewed'),path,proof)


def test_guardian_waits_for_surviving_metric_scheduler(tmp_path):
    import fcntl
    (tmp_path/'groups.json').write_text('[]')
    assert not owned_workers_active(tmp_path)
    with (tmp_path/'metrics_queue.lock').open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert owned_workers_active(tmp_path)
    assert not owned_workers_active(tmp_path)
