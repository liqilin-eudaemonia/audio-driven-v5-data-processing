"""ResponseNet 适配的纯文本检查：不读音频、不跑切分。"""
import json
import segment
import numpy as np


def test_empty_listener_does_not_occupy_a_turn():
    doc = {'text': '', 'timestamps': {'words': [], 'segments': [
        {'text': '', 'start': 0.0, 'end': 30.9}]}}
    person = segment.person_from_official(doc)
    assert person['segments'] == []
    assert person['words'] == []
    assert person['vad'] == []


def test_speaker_words_use_start_end():
    doc = {'text': 'She was not.', 'timestamps': {
        'words': [
            {'text': 'She', 'start_time': 0.88, 'end_time': 1.28},
            {'text': 'was', 'start_time': 1.28, 'end_time': 1.44},
        ],
        'segments': [{'text': 'She was.', 'start': 0.8, 'end': 1.5}],
    }}
    person = segment.person_from_official(doc)
    assert [(w['text'], w['start'], w['end']) for w in person['words']] == [
        ('She', 0.88, 1.28), ('was', 1.28, 1.44)]
    assert person['segments'][0]['word_count'] == 2


def test_listener_speaking_regions_are_dropped():
    regions = [
        dict(start=1, end=4, speaker_index=0),
        dict(start=5, end=8, speaker_index=1),
        dict(start=9, end=12, speaker_index=0),
    ]
    kept = segment.keep_observation_subject(regions)
    assert [r['speaker_index'] for r in kept] == [0, 0]
    assert len(regions) - len(kept) == 1


def test_local_word_file_replaces_empty_official_text(tmp_path=None):
    import tempfile
    from pathlib import Path
    official = {
        'text': '',
        'timestamps': {'words': [], 'segments': [{'text': '', 'start': 0.0, 'end': 10.0}]},
    }
    local = {
        'text': 'hello there',
        'word_time_source': 'whisper_large_v3_on_official_separated_wav',
        'timestamps': {'words': [
            {'text': 'hello', 'start_time': 1.0, 'end_time': 1.4},
            {'text': 'there', 'start_time': 1.4, 'end_time': 1.8},
        ], 'segments': []},
    }
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        official_path = root / 'official.json'
        official_path.write_text(json.dumps(official))
        local_path = root / 'local' / 'abc' / 'speaker.json'
        local_path.parent.mkdir(parents=True)
        local_path.write_text(json.dumps(local))
        doc, meta = segment.document_for(official_path, 'abc', 'speaker', local_root=root / 'local')
        assert meta['source'] == 'whisper_large_v3_on_official_separated_wav'
        person = segment.person_from_official(doc)
        assert [word['text'] for word in person['words']] == ['hello', 'there']
        untouched = json.loads(official_path.read_text())
        assert untouched['timestamps']['words'] == []


def test_long_word_on_silence_shrinks_to_the_loud_part():
    sample_rate = 16000
    mono = np.zeros(int(6.2 * sample_rate), dtype=np.float32)
    mono[int(4.5 * sample_rate):int(6.0 * sample_rate)] = 0.05
    doc = {'text': 'hmm', 'timestamps': {'words': [
        {'text': 'hmm', 'start_time': 0.0, 'end_time': 6.16},
    ], 'segments': []}}
    repaired = segment.repair_long_words(doc, mono, sample_rate)
    words = repaired['timestamps']['words']
    assert words
    assert words[0]['start_time'] >= 4.3
    assert words[-1]['end_time'] <= 6.2
    assert all(word['end_time'] - word['start_time'] <= 1.5 for word in words)


def test_frozen_fingerprint_constant_matches_release():
    pinned = json.loads(segment.RELEASE.read_text())['module_sha256']
    assert segment.rule_sha256() == pinned
    assert pinned == '8a7f326fafbda95382a986ce19a634270e42e915db050fd07bf89afaa6ec7f0a'


if __name__ == '__main__':
    test_empty_listener_does_not_occupy_a_turn()
    test_speaker_words_use_start_end()
    test_listener_speaking_regions_are_dropped()
    test_local_word_file_replaces_empty_official_text()
    test_long_word_on_silence_shrinks_to_the_loud_part()
    test_frozen_fingerprint_constant_matches_release()
    print('ok')
