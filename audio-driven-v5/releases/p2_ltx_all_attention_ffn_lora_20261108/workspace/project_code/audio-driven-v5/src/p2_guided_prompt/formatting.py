"""Lossless annotation rendering: no inference, translation or interval merging."""
import copy
import json
import math
from p2_listen.contracts import digest, safe_id
from p3_listen.prompts import GEMMA

SCHEMA = 'p2-verbatim-guidance-v1'
INTRO = '\n\n以下回应安排的时间单位为秒，相对于片段起点。按描述生成视觉回应和音频回应；引号内是具体发声内容。'


def original_string(value, field):
    if value is not None and not isinstance(value, str):
        raise ValueError(f'{field} must be a string or null')
    return value if value is not None and value.strip() else None


def audio_response(description, text):
    description = original_string(description, 'description')
    text = original_string(text, 'text')
    if not description and not text:
        raise ValueError('Audio event has no original content')
    if not text:
        return description
    if not description or description == text:
        return json.dumps(text, ensure_ascii=False)
    # Only explicit whole quoted text counts as already displayed; substrings
    # such as Right in Bright must not suppress an independent transcript.
    if any(left + text + right in description for left, right in [('"', '"'), ("'", "'"), ('“', '”')]):
        return description
    return description + '；发声内容：' + json.dumps(text, ensure_ascii=False)


def intervals(event, duration):
    spans = event.get('intervals')
    if spans is None:
        spans = [dict(start=event['start'], end=event['end'])]
    if not isinstance(spans, list) or not spans:
        raise ValueError('Event has empty/invalid intervals; resolve upstream, do not invent support')
    result = []
    for span in spans:
        a, b = span['start'], span['end']
        if any(type(t) not in (int, float) or not math.isfinite(t) for t in (a, b)) or not 0 <= a < b <= duration:
            raise ValueError('Event interval outside clip support')
        if result and a < result[-1][1]:
            raise ValueError('An event has overlapping/unsorted effective intervals')
        result.append((a, b))
    return result


def render(record):
    sid = safe_id(record['sample_id']); duration = record['duration_s']
    if type(duration) not in (int, float) or not math.isfinite(duration) or duration <= 0:
        raise ValueError('Invalid clip duration')
    if record.get('schema') != 'listen-existing-result-fusion-verbatim-v2':
        raise ValueError('Expected existing integrated verbatim annotations')
    entries = []; ids = set()
    for ordinal, event in enumerate(record['events']):
        eid = safe_id(event['event_id'])
        if eid in ids:
            raise ValueError('Duplicate event ID; resolve upstream')
        ids.add(eid)
        modality = event['modality']; desc = original_string(event.get('description'), 'description')
        text = original_string(event.get('text'), 'text')
        if modality == 'visual':
            if not desc:
                raise ValueError('Visual event has no description')
            if text is not None:
                raise ValueError('Unexpected visual transcript; resolve modality upstream')
            content = desc; field = '视觉回应'
        elif modality == 'audio':
            content = audio_response(desc, text); field = '音频回应'
        else:
            raise ValueError('Unknown annotation modality')
        for a, b in intervals(event, duration):
            entries.append(dict(start=a, end=b, modality=modality, field=field, content=content,
                                source_event_id=eid, source_event_ordinal=ordinal))
    entries.sort(key=lambda e: (e['start'], e['end'], e['source_event_ordinal']))
    lines = [GEMMA, INTRO]; previous = None
    for e in entries:
        span = (e['start'], e['end'])
        if span != previous:
            # JSON number rendering is round-trip exact; never round boundaries.
            lines.extend(['', '[' + json.dumps(span[0]) + '–' + json.dumps(span[1]) + '秒]'])
        lines.append(e['field'] + '：' + e['content']); previous = span
    prompt = '\n'.join(lines)
    return dict(schema=SCHEMA, sample_id=sid, duration_s=duration, prompt=prompt,
                prompt_sha256=digest(prompt), source_record_sha256=digest(record),
                source_record=copy.deepcopy(record), entries=entries,
                policy=dict(original_text=True, original_effective_intervals=True,
                            extra_deduplication=False, unknown_to_silence=False,
                            new_audio_inferred_from_visual=False))


def validate_guidance(value):
    expected = render(value['source_record'])
    if value != expected:
        raise ValueError('Guidance differs from its original annotations or rendering policy')
    return value
