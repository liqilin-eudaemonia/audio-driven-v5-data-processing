import importlib.util
import json
from pathlib import Path
import sys
import types

import pytest

ENTRY = Path(__file__).resolve().parents[1] / 'scripts/p2_whisper8000_then_event_feature.py'
spec = importlib.util.spec_from_file_location('whisper8000_handoff', ENTRY)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    source = tmp_path / 'source'; target = tmp_path / 'target'; work = tmp_path / 'work'
    for p in (source / 'queue', target / 'queue', work): p.mkdir(parents=True)
    code = tmp_path / 'source.py'; code.write_text('unchanged')
    c = dict(training=dict(global_batch=28), loss='av_event_feature')
    paths = {k: tmp_path / (k + '.json') for k in ('source_config', 'target_config', 'target_cluster')}
    put(paths['source_config'], {})
    put(paths['target_config'], c)
    put(paths['target_cluster'], dict(nodes=[dict(gpus=list(range(28)))]))
    put(target / 'run.json', dict(experiment=c, world=28, source={str(code): m.sha(code)}))
    cp = target / 'step_005500.pt'; cp.write_bytes(b'complete target')
    put(cp.with_suffix('.complete.json'), dict(step=5500, sha256=m.sha(cp)))
    (target / 'metrics.jsonl').write_text('{"step":5514}\n')
    put(source / 'status.json', dict(step=8000, complete=False))
    put(source / 'queue/state.json', dict(phase='training'))
    put(source / 'queue/controller.json', {})
    s = dict(source_output=str(source), target_output=str(target), target_resume=str(cp))
    for k, p in paths.items(): s.update({k: str(p), k + '_sha256': m.sha(p)})
    put(work / 'spec.json', s)
    monkeypatch.setattr(m, 'alive_record', lambda *a: False)
    monkeypatch.setattr(m, 'pid_capacity', lambda cluster: [dict(free=True)])
    return source, target, work, cp


def test_complete_save_precedes_queue_stop_and_target_launch(fixture, monkeypatch):
    source, target, work, cp = fixture
    calls = []
    def finish_checkpoint(_):
        assert (source / 'STOP').exists()
        assert not (source / 'queue/STOP').exists()
        assert not calls
        saved = source / 'step_008000.pt'; saved.write_bytes(b'complete source')
        put(saved.with_suffix('.complete.json'), dict(step=8000, sha256=m.sha(saved)))
        put(source / 'queue/state.json', dict(phase='stopped'))
    monkeypatch.setattr(m.time, 'sleep', finish_checkpoint)
    def launch(cluster, commands, logs, **kw):
        assert (work / 'SOURCE_SAVED.json').exists()
        assert (source / 'queue/STOP').exists()
        assert commands == ['strict-resume', str(cp)]
        calls.append('launched')
        put(target / 'status.json', dict(step=10202, budget=10202, complete=True))
    monkeypatch.setitem(sys.modules, 'p2_j_event_feature_launch', types.SimpleNamespace(
        commands=lambda cluster, config, resume: ['strict-resume', str(resume)],
        resources=lambda cluster: [dict(free=True)]))
    monkeypatch.setitem(sys.modules, 'listen_cluster', types.SimpleNamespace(launch=launch))
    m.run(work)
    assert calls == ['launched']
    assert m.read(work / 'state.json')['phase'] == 'complete'
    assert (work / 'metrics_before_resume.jsonl').read_text() == '{"step":5514}\n'


def test_bad_resume_does_not_pause_current_training(fixture):
    source, target, work, cp = fixture
    cp.write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='step5500'):
        m.run(work)
    assert not (source / 'STOP').exists()


def test_objective_check_rejects_missing_auxiliary_gradient():
    row = dict(av=1., event=2., feature_video=3., feature_audio=4., weighted_total=1.275,
        grad_norm=dict(connector=1., events=1., features=1., text_lora=1.))
    m.verify_objective(row)
    row['grad_norm']['features'] = 0.
    with pytest.raises(ValueError, match='gradient'):
        m.verify_objective(row)


def test_pid_headroom_gate():
    assert not m.enough_pid_capacity(628942, '629145')
    assert not m.enough_pid_capacity(10000, '14095')
    assert m.enough_pid_capacity(10000, '14096')
    assert m.enough_pid_capacity(10000, 'max')
