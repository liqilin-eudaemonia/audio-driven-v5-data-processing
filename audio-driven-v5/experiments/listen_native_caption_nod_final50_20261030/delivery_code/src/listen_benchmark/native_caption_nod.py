"""Independent audiovisual captioning with timed nod evidence; no prior caption input."""
import json
import math

PROMPT_VERSION = 'native-av-natural-nod-v1'

INSTRUCTION = '''Watch and listen to this listener clip. Write one coherent English paragraph describing directly visible actions and audible listener responses across the whole clip.
Describe actions in order, sustained states, simultaneous actions, and genuine repetitions. Include visible facial, eye, head, hand, and body movements. Describe each sustained state or movement once; retain separate occurrences when they actually recur. Use a clear subject and natural temporal connections.
Quote clearly intelligible spoken responses exactly as heard, and describe audible nonverbal vocalizations. Do not guess unclear words or infer sounds from mouth movements. listener_audio_is_silent is computed from the supplied waveform: true means every sample is zero, so describe visible actions only and do not claim speech, audible laughter, or other vocalizations. false does not by itself establish that any vocal response occurs.
Before writing, check every supplementary nodding interval against its place in the clip. Integrate supported nodding into the action sequence, including what happens at the same time. Do not invent an action if the video clearly contradicts it. An empty list does not prove nodding is absent.
Interval keys: "id" identifies a candidate interval, not an individual nod; "start" and "end" are seconds relative to the current pair's start (0 = pair start); "observation" names the movement. An interval can contain multiple nods. Do not infer nod counts from interval counts, or describe a head dip and the same nod as two separate actions.
Describe physical movements and expressions, not thoughts, intentions, feelings, agreement, or communicative meanings. Avoid interpretations such as indicating, suggesting, thoughtful, contemplative, or acknowledgment. Do not add unrelated appearance details.
Output only the paragraph describing the person. Do not explain the input, detector, supplementary data, silence flag, interval IDs, or numeric timestamps. Do not output JSON, a list, or your checking process.'''


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
    silent=row['audio_preflight']['full_silence']
    if type(silent) is not bool:raise ValueError('Silence evidence must be a boolean')
    return (f'The listener clip is {duration:.3f} seconds long.\n'+INSTRUCTION+
            '\n\nlistener_audio_is_silent: '+json.dumps(silent)+
            '\n\nSUPPLEMENTARY NODDING INTERVALS:\n'+json.dumps(cues,ensure_ascii=False))
