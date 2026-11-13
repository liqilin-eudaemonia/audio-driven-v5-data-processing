import copy
import pytest
from p2_guided_prompt_en.formatting import render, validate_guidance, is_english_script


def record():
    return dict(schema='listen-existing-result-fusion-verbatim-v2', sample_id='sample', duration_s=3.,
                events=[dict(event_id='v1', modality='visual', start=.1234567, end=2.1,
                             description='The listener smiles.', text=None),
                        dict(event_id='a1', modality='audio', start=1., end=1.5,
                             description=None, text='嗯')])


def test_english_prompt_retains_original_with_translation_audit():
    source=record(); before=copy.deepcopy(source); value=render(source)
    assert source==before==value['source_record']
    assert is_english_script(value['prompt'])
    assert 'The listener smiles.' in value['prompt']
    assert '[0.1234567–2.1 seconds]' in value['prompt']
    assert 'Audio response: "Mm-hmm"' in value['prompt']
    assert value['translations']==[dict(event_id='a1',field='text',original='嗯',english='Mm-hmm',method='user_authorized_text_translation_not_ASR_correction')]
    assert validate_guidance(value)==value


def test_english_original_and_spoken_content_are_unchanged():
    source=record();source['events'][1].update(description='A short acknowledgment.',text='Yeah')
    value=render(source)
    assert 'A short acknowledgment.; vocal content: "Yeah"' in value['prompt']
    assert not value['translations']


def test_unknown_non_english_and_audit_tampering_are_rejected():
    source=record();source['events'][1]['text']='未确认'
    with pytest.raises(ValueError,match='Unresolved'):render(source)
    value=render(record());value['translations'][0]['english']='Sure'
    with pytest.raises(ValueError):validate_guidance(value)
