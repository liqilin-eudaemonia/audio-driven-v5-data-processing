import importlib.util
import json
from pathlib import Path
import sys
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from listen_benchmark.semantic_prompt import digest


def setup_preview(tmp_path,monkeypatch,drop_audio=False):
    spec=importlib.util.spec_from_file_location('semantic_preview',ROOT/'scripts/build_listen_semantic_prompt_review.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    run=tmp_path/'run';(run/'results').mkdir(parents=True);(run/'prompts').mkdir()
    events=[dict(event_id='a1',modality='audio',start=0,end=.2,text='okay',description='okay'),
            dict(event_id='a2',modality='audio',start=1,end=1.2,text='okay',description='okay')]
    record=dict(sample_id='s',duration_s=2,events=events)
    jobs=[dict(job_id='job',record=record,provenance=dict(role='current_postprocessed'))]
    inputs=tmp_path/'inputs.jsonl';inputs.write_text(json.dumps(jobs[0])+'\n')
    protocol=dict(model='/private/ckpts/Qwen3-30B-A3B-Instruct-2507',input_sha256=digest(jobs))
    (run/'protocol.json').write_text(json.dumps(protocol))
    (run/'summary.json').write_text(json.dumps(dict(samples=1,accepted=1,rejected=[])))
    result=dict(sample_id='s',source_record=record,events=events[:-1] if drop_audio else events,
                provenance=dict(protocol_sha256=digest(protocol)))
    (run/'results/job.json').write_text(json.dumps(result))
    targets=tmp_path/'targets.json';targets.write_text('[]')
    monkeypatch.setattr(module,'TARGETS',targets)
    monkeypatch.setattr(module,'link_media',lambda *args:('',''))
    output=tmp_path/'preview'
    monkeypatch.setattr(sys,'argv',['builder','--run',str(run),'--inputs',str(inputs),'--output',str(output)])
    return module,output


def test_preview_counts_model_are_dynamic_and_audio_count_preserved(tmp_path,monkeypatch):
    module,output=setup_preview(tmp_path,monkeypatch)
    module.main()
    stats=json.loads((output/'summary.json').read_text())
    assert stats['samples']==1 and stats['accepted']==1 and stats['current']['audio']==2
    html=(output/'index.html').read_text()
    assert 'Qwen3-30B-A3B-Instruct-2507' in html
    assert 'Qwen3-VL-8B-Instruct' not in html


def test_preview_rejects_missing_tail_audio_previously_missed_by_zip(tmp_path,monkeypatch):
    module,_=setup_preview(tmp_path,monkeypatch,drop_audio=True)
    with pytest.raises(SystemExit,match='audio annotation changed'):module.main()
