"""Versioned observational judgments; no timing alignment or mental-state claims."""
import json
from .contracts import CLASSES

VERSION = 'annotation-pipeline-presence-visual4-audio3-20261108-v1'

DEFINITIONS = {
    'nodding': 'Visible up/down nod-like head movement; not just a stationary tilted head or a one-way look downward. Do not infer agreement.',
    'head_shaking': 'Visible back/forth head shaking; not just turning once to look sideways. Do not infer disagreement.',
    'smiling': 'A visible smile or laughing facial appearance. An open mouth or speech mouth movement alone is insufficient. It need not make a sound.',
    'frowning': 'A visible frown, with brow furrowing/lowering. Do not infer sadness or anger from appearance alone.',
    'audible_laughter': 'Clearly audible laughing or chuckling, including audible ha/heh laughing sounds. Never infer it from a face or transcript alone.',
    'brief_verbal_response': 'A distinct intelligible brief word/short-phrase response, e.g. yeah, okay, right, no, wow. A sustained full utterance is not automatically a brief response; only a separately audible brief response counts.',
    'nonlexical_response': 'A clearly audible nonlexical response such as mhm, uh-huh, hmm, uh, ah or oh. Exclude laughter, ordinary breathing, cough and background noise. Do not count the same uh-huh as both this class and a verbal response.',
}


def schema(mod):
    return {'type': 'object', 'additionalProperties': False, 'required': ['classes', 'long_speech_observed'],
        'properties': {'classes': {'type': 'object', 'additionalProperties': False, 'required': CLASSES[mod],
            'properties': {c: {'type': 'object', 'additionalProperties': False, 'required': ['state', 'evidence'],
                'properties': {'state': {'type': 'string', 'enum': ['present', 'absent', 'unknown']},
                               'evidence': {'type': 'string'}}} for c in CLASSES[mod]}},
            'long_speech_observed': {'type': 'boolean'}}}


ASR_SCHEMA = {'type': 'object', 'additionalProperties': False, 'required': ['transcript', 'nonlexical_syllables', 'uncertain'],
    'properties': {'transcript': {'type': 'string'}, 'nonlexical_syllables': {'type': 'array', 'items': {'type': 'string'}},
                   'uncertain': {'type': 'boolean'}}}
ASR_PROMPT = '''Listen to the supplied listener audio only. Produce candidate audible words and nonlexical syllables for a later independent check against THIS SAME waveform. Transcribe intelligible spoken words exactly as heard; do not translate or invent words. Put mhm, uh-huh, hmm, uh, ah and oh in nonlexical_syllables. Do not turn laughter into the greeting Hey. Leave transcript empty when no words are clearly intelligible. Ordinary breath, noise and cough are not words. If uncertain, set uncertain=true. These are fallible cues, not ground truth. Return only the required JSON.'''


def prompt(mod, duration, candidates=None, asr=None):
    base = f'''Observe this entire {duration:.3f}-second listener {'silent video' if mod == 'visual' else 'audio waveform'}.
Classify each requested observable category independently as present, absent or unknown.
Repeated occurrences of the same category count only as present. Multiple different categories can coexist.
Use present only when directly supported. Use absent only when you can inspect the entire supplied material and confidently rule it out. Use unknown for ambiguity or missing/unclear evidence. A class not mentioned in a description is NOT absent.
Do not infer intentions, agreement, emotional appropriateness, or causes. Do not invent behavior or sounds to fill quiet intervals. Evidence must briefly describe what is observed, including why a negative or unknown is justified.
This is a standalone anonymous media observation. No reference event schedule or expected answer exists.
'''
    if mod == 'visual':
        base += 'You are given video WITHOUT audible sound. Classify only visible behavior; no inference about audible laughter, speech or vocalization. If the target listener is ambiguous among multiple people, use unknown rather than borrowing another person\'s behavior. Set long_speech_observed=false.\n'
        if candidates:
            base += 'Independent annotation-pipeline movement candidates below may be WRONG or incomplete. Verify in video; empty candidates are not absence, intervals are not repetition counts.\n'
            base += json.dumps(candidates, ensure_ascii=False) + '\n'
    else:
        base += 'You are given AUDIO ONLY. Do not infer visible smiles, nods, or mouth movements. A clearly silent waveform contains none of the three sounds. Set long_speech_observed=true only for audible sustained longer speech. Short/long speech ambiguity should make brief_verbal_response unknown. Do not impose an unstated seconds/word-count cutoff.\n'
        if asr:
            base += 'Fallible ASR cues derived only from this same waveform. Listen again; ignore incorrect cues and never force their labels:\n' + json.dumps(asr, ensure_ascii=False) + '\n'
    for c in CLASSES[mod]: base += c + ': ' + DEFINITIONS[c] + '\n'
    return base + 'Return only the required JSON. Do not return a confidence probability or omit any class.'


def parse_label_response(value, mod):
    if not isinstance(value, dict) or set(value) != {'classes', 'long_speech_observed'}:
        raise ValueError('Incomplete structured detection response')
    if type(value['long_speech_observed']) is not bool: raise ValueError('Invalid long_speech flag')
    if not isinstance(value['classes'], dict) or set(value['classes']) != set(CLASSES[mod]):
        raise ValueError('All modality classes must be explicit')
    for item in value['classes'].values():
        if not isinstance(item, dict) or set(item) != {'state', 'evidence'}: raise ValueError('Invalid class fields')
        if item['state'] not in {'present', 'absent', 'unknown'}: raise ValueError('Invalid class state')
        if not isinstance(item['evidence'], str) or not item['evidence'].strip(): raise ValueError('Missing observation evidence')
    return value
