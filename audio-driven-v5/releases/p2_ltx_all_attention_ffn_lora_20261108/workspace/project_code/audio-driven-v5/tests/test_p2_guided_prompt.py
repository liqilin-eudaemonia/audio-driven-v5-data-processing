"""Preservation and isolation tests for target-annotation-guided inference."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from p2_guided_prompt.formatting import render, validate_guidance, audio_response, intervals
from p2_guided_prompt.runtime import substituted_text, checked_guidance
from p2_listen.contracts import atomic_json, digest


def event(eid='e0',modality='audio',description='Affirmative vocalization',text='Yeah',start=.1,end=.5,**kwargs):
    return dict(event_id=eid,modality=modality,description=description,text=text,start=start,end=end,**kwargs)


def record(events=None):
    return dict(schema='listen-existing-result-fusion-verbatim-v2',sample_id='clip1',duration_s=3.,
                events=[event()] if events is None else events,unlabeled_time_policy='unknown',
                response_groups=[],training_admitted=False)


def test_original_record_and_content_are_preserved():
    raw=record([event(description=' 原描述：微弱的声音，可能表示理解。\n',text='嗯！')]);before=copy.deepcopy(raw);g=render(raw)
    assert raw==before==g['source_record']
    assert raw['events'][0]['description'] in g['prompt']
    assert '"嗯！"' in g['prompt']
    assert g['source_record_sha256']==digest(raw)
    assert validate_guidance(g)==g


@pytest.mark.parametrize('desc,text,expected',[
    ('Yeah','Yeah','"Yeah"'),(None,'嗯','"嗯"'),('Laughter',None,'Laughter'),
    ("Affirmative vocalization ('Mhm')",'Mhm',"Affirmative vocalization ('Mhm')"),
    ('Bright smile','Right','Bright smile；发声内容："Right"'),
    ('yeah','Yeah','yeah；发声内容："Yeah"'),
])
def test_audio_single_field_exact_and_quoted_dedup(desc,text,expected):
    assert audio_response(desc,text)==expected


def test_visual_laugh_does_not_invent_audio():
    g=render(record([event(modality='visual',description='Smiling broadly, indicating amusement or laughter.',text=None)]))
    assert '视觉回应：Smiling broadly, indicating amusement or laughter.' in g['prompt']
    assert not any(e['modality']=='audio' for e in g['entries'])
    assert '音频回应：' not in g['prompt']


def test_disjoint_intervals_do_not_fill_unknown_gaps():
    g=render(record([event(start=.1,end=2.1,intervals=[dict(start=.1,end=.2),dict(start=2.,end=2.1)])]))
    assert [(e['start'],e['end']) for e in g['entries']]==[(.1,.2),(2.,2.1)]
    assert '[0.1–2.1秒]' not in g['prompt']


def test_stable_sort_keeps_distinct_events_and_all_ties():
    raw=record([event('late',start=1.,end=2.),event('one'),event('two')]);g=render(raw)
    assert [e['source_event_id'] for e in g['entries']]==['one','two','late']
    assert g['prompt'].count('音频回应：')==3
    assert g['source_record']['events']==raw['events']


def test_no_rounding_of_timestamps():
    a=1.4890208333333332;g=render(record([event(start=a,end=2.0089791666666668)]))
    assert str(a) in g['prompt']
    assert g['entries'][0]['start']==a


def test_empty_unknown_has_no_fabricated_silence():
    g=render(record([]));assert not g['entries'];assert '保持静音' not in g['prompt']
    assert not g['policy']['unknown_to_silence']


@pytest.mark.parametrize('changes',[
    dict(start=-.1),dict(end=4.),dict(start=True),dict(end=float('nan')),
    dict(intervals=[]),dict(intervals=[dict(start=.5,end=1.),dict(start=.7,end=1.2)]),
    dict(modality='other'),dict(description=None,text=None),dict(modality='visual',text='Yeah'),
])
def test_invalid_annotation_fails_without_repair(changes):
    with pytest.raises(ValueError):render(record([event(**changes)]))


def test_duplicate_event_ids_rejected():
    with pytest.raises(ValueError):render(record([event(),event()]))


@pytest.mark.parametrize('field', ['prompt','entries','source_record'])
def test_guidance_tamper_rejected(field):
    g=render(record())
    if field=='prompt':g[field]+='new directive'
    elif field=='entries':g[field][0]['content']='new content'
    else:g[field]['events'][0]['text']='Right'
    with pytest.raises(ValueError):validate_guidance(g)


def test_self_consistent_annotation_replacement_rejected_by_frozen_protocol(tmp_path):
    first=render(record());replacement=render(record([event(text='Right')]))
    atomic_json(tmp_path/'prompts/clip1.json',replacement)
    with pytest.raises(ValueError):checked_guidance(tmp_path,dict(guidance={'clip1':digest(first)}),'clip1')


def test_substitution_restores_native_loader_even_after_error():
    original=lambda: 'fixed';replacement=lambda:'guided';native=SimpleNamespace(load_text=original)
    with pytest.raises(RuntimeError):
        with substituted_text(native,replacement):
            assert native.load_text()=='guided';raise RuntimeError('generation failure')
    assert native.load_text is original


def test_real_prepare_cli_and_selected_original_provenance(tmp_path):
    import subprocess
    from p2_listen.contracts import sha
    root=Path(__file__).resolve().parents[1]
    speaker=tmp_path/'speaker.wav';image=tmp_path/'first.png';speaker.write_bytes(b'fixture audio');image.write_bytes(b'fixture image')
    obs=dict(schema='p2-listen-observation-v1',sample_id='clip1',experiment_split='train',official_split='train',
             group_id='group1',grouping_ids=['person:one'],speaker_audio=str(speaker),listener_image=str(image),
             requested_duration_s=3.,geometry=dict(fps=25,height=512,width=512),
             media_hashes=dict(speaker_audio=sha(speaker),listener_image=sha(image)))
    atomic_json(tmp_path/'observations.json',[obs]);labels=tmp_path/'labels.jsonl';labels.write_text(json.dumps(record(),ensure_ascii=False)+'\n')
    result=subprocess.run([sys.executable,str(root/'scripts/p2_guided_prompt.py'),'prepare','--observations',str(tmp_path/'observations.json'),
                           '--annotations',str(labels),'--output',str(tmp_path/'out')],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    actual=json.loads((tmp_path/'out/prompts/clip1.json').read_text());assert actual==render(record())
    protocol=json.loads((tmp_path/'out/protocol.json').read_text());assert protocol['target_annotation_access']
    assert not protocol['formal_source_only_evaluation']


@pytest.mark.parametrize('tokens,expected',[(1,1024),(1024,1024),(1025,2048),(2048,2048),(2049,4096),(4096,4096)])
def test_lossless_context_bucket(tokens,expected):
    from p2_guided_prompt.encoding import sequence_length
    assert sequence_length(tokens)==expected


def test_context_overflow_never_truncates():
    from p2_guided_prompt.encoding import sequence_length
    with pytest.raises(OverflowError):sequence_length(4097)
    with pytest.raises(OverflowError):sequence_length(1025,1024)
