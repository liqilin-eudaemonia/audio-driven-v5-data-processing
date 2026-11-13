"""Existing English guidance layout with verbatim annotation fields in any script."""
import copy
import json
import math

from p2_guided_prompt_en.formatting import GEMMA, INTRO, audio_response, intervals, original_string
from p2_listen.contracts import digest, safe_id


def render_verbatim(record):
    sid = safe_id(record['sample_id']); duration = record['duration_s']
    if type(duration) not in (int, float) or not math.isfinite(duration) or duration <= 0:
        raise ValueError('Invalid clip duration')
    if record.get('schema') != 'listen-existing-result-fusion-verbatim-v2':
        raise ValueError('Expected existing integrated verbatim annotations')
    entries = []; ids = set()
    for ordinal, event in enumerate(record['events']):
        eid = safe_id(event['event_id'])
        if eid in ids: raise ValueError('Duplicate event ID; resolve upstream')
        ids.add(eid)
        modality = event['modality']
        desc = original_string(event.get('description'), 'description')
        text = original_string(event.get('text'), 'text')
        if modality == 'visual':
            if not desc or text is not None: raise ValueError('Invalid visual description/transcript')
            content = desc; field = 'Visual response'
        elif modality == 'audio':
            content = audio_response(desc, text); field = 'Audio response'
        else: raise ValueError('Unknown annotation modality')
        for a, b in intervals(event, duration):
            entries.append(dict(start=a, end=b, modality=modality, field=field, content=content,
                                source_event_id=eid, source_event_ordinal=ordinal))
    entries.sort(key=lambda e: (e['start'], e['end'], e['source_event_ordinal']))
    lines = [GEMMA, INTRO]; previous = None
    for e in entries:
        span = (e['start'], e['end'])
        if span != previous:
            lines.extend(['', '[' + json.dumps(span[0]) + '–' + json.dumps(span[1]) + ' seconds]'])
        lines.append(e['field'] + ': ' + e['content']); previous = span
    prompt = '\n'.join(lines)
    return dict(schema='p2-verbatim-guidance-v1', sample_id=sid, duration_s=duration, prompt=prompt,
                prompt_sha256=digest(prompt), source_record_sha256=digest(record),
                source_record=copy.deepcopy(record), entries=entries, translations=[],
                policy=dict(template='existing_english_guidance', original_fields_preserved=True,
                            translate=False, original_effective_intervals=True, extra_deduplication=False,
                            unknown_to_silence=False, new_audio_inferred_from_visual=False))
