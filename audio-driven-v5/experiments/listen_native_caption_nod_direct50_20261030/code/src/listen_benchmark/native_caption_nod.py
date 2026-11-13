"""Independent audiovisual captioning with timed nod evidence; no prior caption input."""
import json
import math

INSTRUCTION = '''Watch and listen to this listener clip. Write one coherent English paragraph describing the directly visible movements and audible vocal responses, using the video, synchronized listener audio, and supplementary nodding intervals below.
Describe the order of actions, sustained expressions or postures, actions happening together, and genuine repetitions. Include brief facial, eye, head, hand, and body movements when visible. Quote clearly intelligible spoken responses exactly as heard and describe audible nonverbal vocalizations. Do not infer speech or laughter from visible mouth movements alone, and do not guess unclear words.
Each start and end value below is in seconds relative to the start of the current pair clip (0 = pair start).
Each supplementary nodding interval has four fields: "id" is a reference identifier, not a count of nods; "start" is the beginning of the interval; "end" is the end of the interval; "observation" names the observed movement ("head nodding").
Treat these intervals as supplementary visual evidence of nodding. If the media clearly contradicts an interval, do not invent that action. An empty list means no supplementary nod evidence, not proof that nodding is absent.
Integrate supported nodding at the correct place in the action sequence, including whether it happens during a smile, during a vocal response, or between other actions. Describe the same movement once: do not count a head dip and its corresponding nod as separate actions. An interval can contain multiple nods and is NOT an exact count of individual nods. Keep genuinely separated nodding episodes separate; do not invent a precise number of nods.
Describe observable actions and sounds only, without psychological or communicative interpretations. Use a clear subject and natural temporal wording, without redundant descriptions or unrelated appearance details. Do not mention the detector, scores, interval IDs, or numeric timestamps in the final paragraph. Output only the paragraph, not JSON or a list.'''


def render(row):
    bundle=row['nod_supplement'];events=bundle['events'];duration=row['duration_s']
    if not isinstance(events,list):raise ValueError('Nod events must be a list')
    if any(k in bundle for k in ('prior_caption','prior_result_file','prior_result_sha256')):
        raise ValueError('Independent annotation must not carry prior caption inputs')
    seen=set();last=-1.
    for e in events:
        a,b=e['start'],e['end']
        if (type(a) not in (int,float) or type(b) not in (int,float) or not math.isfinite(a+b)
                or not 0<=a<b<=duration or a<last):raise ValueError('Invalid or unordered nod interval')
        if e['id'] in seen:raise ValueError('Duplicate nod ID')
        seen.add(e['id']);last=a
    if bundle['sample_id']!=row['sample_id'] or bundle['fps']!=25 or bundle['profile']!='recall':
        raise ValueError('Wrong sample, FPS, or detector profile')
    cues=[dict(id=e['id'],start=e['start'],end=e['end'],observation='head nodding') for e in events]
    return (f'The listener clip is {duration:.3f} seconds long.\n'+INSTRUCTION+
            '\n\nSUPPLEMENTARY NODDING INTERVALS:\n'+json.dumps(cues,ensure_ascii=False))
