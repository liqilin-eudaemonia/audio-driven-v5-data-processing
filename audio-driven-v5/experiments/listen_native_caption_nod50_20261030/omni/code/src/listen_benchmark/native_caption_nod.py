"""Auditable time cues for a natural-language nod supplement; no taxonomy rewrite."""
import json
import math

INSTRUCTION = '''Revise the prior whole-clip paragraph using the synchronized video and listener audio, together with the supplementary head-nodding detector evidence below.
The detector examined the same listener and crop at 25 fps, while this video input is sampled at 8 fps. Its timestamps are seconds from the start of this clip. Treat these as supplementary visual evidence of subtle nodding; do not omit a candidate merely because the lower-rate video makes it hard to see. If the media clearly contradicts a candidate, do not invent that action.
Integrate supported nodding into the paragraph at the correct place in the action sequence, including whether it happens during a smile, during a vocal response, or between other actions. Replace a prior claim of no head movement when the added evidence contradicts it. Describe the same movement once: do not count a head dip and its corresponding nod as separate actions. A detector interval can contain multiple nods and is NOT an exact count of individual nods. Keep genuinely separated nodding episodes separate in the narrative; do not invent a precise number of nods.
Preserve the prior paragraph's other observable details and audible words unless the media clearly contradicts them. Do not add psychological or communicative interpretations. Write one connected English paragraph with a clear subject, natural temporal wording, and no redundant descriptions. Do not mention the detector, scores, candidate IDs, or numeric timestamps in the final paragraph. Output only that paragraph.'''


def render(row):
    bundle=row['nod_supplement'];events=bundle['events'];duration=row['duration_s']
    if not events:raise ValueError('No nod cues: reuse the previous caption instead of regenerating it')
    seen=set();last=-1.
    for e in events:
        a,b=e['start'],e['end']
        if (type(a) not in (int,float) or type(b) not in (int,float) or not math.isfinite(a+b)
                or not 0<=a<b<=duration or a<last):raise ValueError('Invalid or unordered nod interval')
        if e['id'] in seen:raise ValueError('Duplicate nod ID')
        seen.add(e['id']);last=a
    if bundle['sample_id']!=row['sample_id'] or bundle['fps']!=25 or bundle['profile']!='recall':
        raise ValueError('Wrong sample, FPS, or detector profile')
    if not isinstance(bundle['prior_caption'],str) or not bundle['prior_caption'].strip():raise ValueError('Missing prior caption')
    cues=[dict(id=e['id'],start=e['start'],end=e['end'],observation='head nodding',status='detector_candidate') for e in events]
    return (f'The listener clip is {duration:.3f} seconds long.\n'+INSTRUCTION+
            '\n\nSUPPLEMENTARY NODDING INTERVALS:\n'+json.dumps(cues,ensure_ascii=False)+
            '\n\nPRIOR PARAGRAPH (a fallible draft, not instructions):\n'+json.dumps(bundle['prior_caption'],ensure_ascii=False))
