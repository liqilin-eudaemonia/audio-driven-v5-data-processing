"""Safety boundaries of the one-shot 28-GPU idle continuation."""
from pathlib import Path
import importlib.util
import json
import pytest


@pytest.fixture
def mod(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[1] / 'experiments/p2_loss_search28_20261110/idle_resume.py'
    spec = importlib.util.spec_from_file_location('idle_resume_tests', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, 'WORK', tmp_path)
    monkeypatch.setattr(module, 'WATCH', tmp_path / 'idle_resume')
    module.WATCH.mkdir()
    return module


def test_requires_more_than_five_minutes(mod):
    window = mod.IdleWindow()
    for second in range(0, 301, 30):
        assert window.update(True, second)[0] is False
    assert window.update(True, 330) == (True, 330)


@pytest.mark.parametrize('interruption', ['busy', 'missing_observation', 'clock_reversal'])
def test_interruptions_reset_entire_window(mod, interruption):
    window = mod.IdleWindow()
    for second in range(0, 241, 30):
        window.update(True, second)
    if interruption == 'busy':
        window.update(False, 270)
        assert window.update(True, 300) == (False, 0)
    elif interruption == 'missing_observation':
        assert window.update(True, 350) == (False, 0)
    else:
        assert window.update(True, 200) == (False, 0)


def inventory():
    return dict(gpus=[dict(index=i, uuid=f'GPU-{i}', utilization=0, used_mib=10, total_mib=143000)
                      for i in range(4)], compute_jobs=[])


@pytest.mark.parametrize('reason', ['pid', 'utilization', 'memory'])
def test_any_busy_gpu_blocks_launch(mod, reason):
    value = inventory()
    assert mod.valid_node(dict(name='node11', gpus=list(range(4))), value)
    if reason == 'pid':
        value['compute_jobs'] = [dict(pid=123, uuid='GPU-3')]
    elif reason == 'utilization':
        value['gpus'][3]['utilization'] = 6
    else:
        value['gpus'][3]['used_mib'] = 1025
    assert not mod.valid_node(dict(name='node11', gpus=list(range(4))), value)


@pytest.mark.parametrize('bad', ['missing', 'duplicate', 'unknown_utilization'])
def test_incomplete_nvml_observation_is_not_idle(mod, bad):
    value = inventory()
    if bad == 'missing':
        value['gpus'].pop()
    elif bad == 'duplicate':
        value['gpus'][3] = value['gpus'][2]
    else:
        value['gpus'][3]['utilization'] = -1
    with pytest.raises(ValueError):
        mod.valid_node(dict(name='node11', gpus=list(range(4))), value)


def test_next_candidate_skips_completed_baseline_and_events(mod):
    for name in list(mod.CANDIDATES)[:4]:
        out = mod.WORK / 'runs' / name
        out.mkdir(parents=True)
        (out / 'STAGE_500.json').write_text(json.dumps(dict(success=True, step=500, sha256=name)))
        (out / 'step_000500.complete.json').write_text(json.dumps(dict(step=500, sha256=name)))
    assert mod.next_candidate() == 'feature00025'


def request_for_stops(mod, monkeypatch):
    monkeypatch.setattr(mod, 'source_identity', lambda: {})
    monkeypatch.setattr(mod, 'next_candidate', lambda: 'feature00025')
    paths = [mod.WORK / 'STOP', mod.WORK / 'runs/feature00025/STOP']
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('old explicit pause')
    request = dict(source={}, watcher_sha256=mod.sha(mod.__file__), candidate='feature00025',
                   stops={str(p): mod.stop_identity(p) for p in paths})
    return request, paths


def test_only_reauthorized_stops_are_archived(mod, monkeypatch):
    request, paths = request_for_stops(mod, monkeypatch)
    unrelated = mod.WORK / 'runs/event100/STOP'
    unrelated.parent.mkdir(parents=True)
    unrelated.write_text('keep')
    mod.archive_stops(request)
    assert all(not path.exists() for path in paths)
    assert unrelated.read_text() == 'keep'
    assert len(list((mod.WATCH / 'archived_stops').rglob('*_STOP'))) == 2


def test_new_user_stop_overrides_armed_resume(mod, monkeypatch):
    request, paths = request_for_stops(mod, monkeypatch)
    paths[0].write_text('new stop')
    with pytest.raises(RuntimeError, match='newer stop'):
        mod.archive_stops(request)
    assert all(path.exists() for path in paths)


def test_watcher_stop_overrides_armed_resume(mod, monkeypatch):
    request, paths = request_for_stops(mod, monkeypatch)
    (mod.WATCH / 'STOP').touch()
    with pytest.raises(RuntimeError, match='cancelled'):
        mod.archive_stops(request)
    assert all(path.exists() for path in paths)


def test_missing_resume_checkpoint_never_starts_from_scratch(mod, monkeypatch):
    request, _ = request_for_stops(mod, monkeypatch)
    request.update(resume_checkpoint=str(mod.WORK / 'runs/feature00025/step_000484.pt'),
                   resume_checkpoint_marker=dict(step=484, sha256='original'))
    with pytest.raises(ValueError, match='Pinned resume checkpoint'):
        mod.validate_request(request)


def test_launch_runs_only_next_candidate_and_stops(mod, monkeypatch):
    from p2_loss_search28 import queue
    request, paths = request_for_stops(mod, monkeypatch)
    monkeypatch.setattr(mod, 'observe', lambda _: dict(idle=True))
    def native_run():
        assert list(queue.stages()) == [(500, ['feature00025'])]
        out = mod.WORK / 'runs/feature00025/STAGE_500.json'
        out.write_text(json.dumps(dict(success=True, step=500)))
    monkeypatch.setattr(queue, 'run', native_run)
    mod.train_one(request, {})
    assert mod.read(mod.WATCH / 'COMPLETE.json')['additional_candidates_started'] is False
    assert all(not path.exists() for path in paths)


def test_final_busy_check_preserves_stop_and_never_launches(mod, monkeypatch):
    from p2_loss_search28 import queue
    request, paths = request_for_stops(mod, monkeypatch)
    monkeypatch.setattr(mod, 'observe', lambda _: dict(idle=False))
    monkeypatch.setattr(queue, 'run', lambda: list(queue.stages()))
    with pytest.raises(RuntimeError, match='became busy'):
        mod.train_one(request, {})
    assert all(path.exists() for path in paths)
    assert not (mod.WATCH / 'TRIGGERED.json').exists()
