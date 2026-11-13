import importlib.util
import json
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location('prepare_semantic', Path(__file__).resolve().parents[1] / 'scripts/prepare_listener_semantic_subset.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def make_subset(tmp_path, records):
    root = tmp_path / 'subset'
    (root / 'manifests').mkdir(parents=True)
    for name in ('targets.json', 'observations.json'):
        (root / 'manifests' / name).write_text(json.dumps([dict(sample_id='sample')]))
    (root / 'integrated_labels.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in records))
    return root


def record():
    return dict(sample_id='sample', duration_s=2, audio_postprocess={'recipe': 'unchanged'},
                events=[dict(event_id='a', modality='audio', start=0.4, end=0.8, description='Yeah', text='Yeah')])


def test_exact_record_and_no_overwrite(tmp_path):
    r = record()
    root = make_subset(tmp_path, [r])
    output = tmp_path / 'out'
    report = module.prepare(root, output)
    assert report['samples'] == 1
    assert json.loads((output / 'inputs.jsonl').read_text())['record'] == r
    with pytest.raises(FileExistsError):
        module.prepare(root, output)


@pytest.mark.parametrize('case', ['missing', 'duplicate', 'unprocessed'])
def test_incomplete_not_published(tmp_path, case):
    r = record()
    if case == 'unprocessed':
        del r['audio_postprocess']
    rows = [] if case == 'missing' else [r, r] if case == 'duplicate' else [r]
    root = make_subset(tmp_path, rows)
    output = tmp_path / 'out'
    with pytest.raises(ValueError):
        module.prepare(root, output)
    assert not (output / 'PREPARED.json').exists()
    assert not (output / 'inputs.jsonl').exists()
