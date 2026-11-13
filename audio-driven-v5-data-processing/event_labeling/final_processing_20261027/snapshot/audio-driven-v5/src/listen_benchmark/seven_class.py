"""Map immutable open observations to seven-class candidates, never to reviewed negatives.

This is a deterministic description mapper, not a new media detector.  Decisions
require modality-specific source evidence as well as the normalized observation.
"""
from copy import deepcopy
import math
import re

VERSION = 'p2-listen-event-v2-visual6-audio1'
MAPPING_VERSION = 'open-to-visual6-audio1-v1'
TAXONOMY = ('head_nod', 'head_shake', 'smile', 'laugh', 'frown', 'surprise',
            'listener_vocalization')
NAMES = dict(zip(TAXONOMY, ('点头', '摇头', '微笑', '大笑', '皱眉', '惊讶表情', '音频回应／发声')))
PATTERNS = {
    'head_nod': r'\bnod(?:s|ding|ded)?\b|点头',
    'head_shake': r'\b(?:shak(?:e|es|ing)|shook)\s+(?:(?:his|her|their|the)\s+)?head\b|\bhead[- ]shak(?:e|es|ing)\b|摇头',
    'smile': r'\bsmil(?:e|es|ing|ed)\b|微笑',
    'laugh': r'\b(?:laugh(?:s|ing|ed|ter)?|chuckl(?:e|es|ing|ed)|giggl(?:e|es|ing|ed))\b|大笑|笑出声',
    'frown': r'\bfrown(?:s|ing|ed)?\b|\bfurrow(?:s|ing|ed)?\s+(?:(?:the|her|his)\s+)?(?:eye)?brows?\b|皱眉',
    'surprise': r'\bsurpris(?:e|ed)\b|惊讶',
}
ALIASES = {
    'head_nod': {'nod', 'nod head', 'head nod'},
    'head_shake': {'shake head', 'head shake', 'head shaking'},
    'smile': {'smile', 'maintain smile'},
    'laugh': {'laugh', 'chuckle', 'giggle'},
    'frown': {'frown', 'furrow eyebrows', 'furrow brow'},
    'surprise': {'surprise', 'show surprise', 'look surprised', 'surprised expression'},
}
SPEECH = r'\b(?:say(?:s|ing)?|said|speak(?:s|ing)?|spoke|utter(?:s|ed|ing)?|vocaliz(?:e|es|ing|ation)|hum(?:s|ming)?)\b|发声|说出|应答|附和'
UNCERTAIN = r'\b(?:possibly|perhaps|maybe|might|could|implied|uncertain|unclear|appears? to|seems? to)\b|可能|似乎|不确定'
NEGATION = r"\b(?:not|no|never|without|isn't|doesn't|didn't)\b|没有|并未|不是|未见"
TRANSITION = r'\b(?:transition\w*|fade\w*|ceas\w*|stop\w*)\b|逐渐消失|转为|停止'
REPEATED = r'\b(?:repeatedly|multiple (?:times|points|nods|cycles)|several (?:times|nods|cycles))\b'
BREATH = r'\b(?:breath\w*|sigh\w*|sniffl\w*|clear(?:s|ing|ed)? (?:the )?throat|cough\w*)\b|呼吸|叹气|吸鼻|清嗓|咳嗽'


def facts(text):
    # Interpretation clauses cannot turn a generic motion into a named reaction.
    return re.split(r',?\s+\b(?:indicating|suggesting|reflecting|signaling|reinforcing|possibly indicating)\b',
                    text, maxsplit=1, flags=re.I)[0]


def explicit(text, pattern):
    """A conservative positive mention; reject negated/hedged local clauses."""
    text = facts(text)
    for match in re.finditer(pattern, text, re.I):
        prefix = re.split(r'[,;.!]|\b(?:but|while|and)\b', text[:match.start()], flags=re.I)[-1]
        if not re.search(NEGATION + '|' + UNCERTAIN, prefix, re.I):
            return True
    return False


def surprise_cues(text):
    patterns = [r'\b(?:eyes?\s+(?:are\s+)?(?:wide\w*|open)|wide\w*\s+eyes?)\b',
                r'\b(?:eyebrows?\s+(?:are\s+)?rais\w*|rais\w*\s+(?:(?:the|her|his)\s+)?eyebrows?)\b',
                r'\b(?:mouth\s+(?:is\s+)?open\w*|open\w*\s+(?:(?:the|her|his)\s+)?mouth)\b']
    return sum(explicit(text, p) for p in patterns)


def interval(event, duration):
    start, end = event.get('start'), event.get('end')
    if not (all(type(v) in (int, float) and math.isfinite(v) for v in (start, end))
            and 0 <= start < end <= duration):
        raise ValueError('Invalid event interval')


def decide(event, origin, sources):
    """Return an auditable decision for exactly one existing open event."""
    signature = origin.get('signature', '').split(' | ')[0].strip().lower()
    behavior = facts(event['behavior'])
    raw = [facts(s['behavior']) for s in sources]
    flags = list(event.get('review_flags', []))
    reasons, classes = [], []
    anchored = origin.get('boundary_method') == 'asr_word_anchor'
    if event['modality'] == 'visual':
        classes = [k for k, aliases in ALIASES.items()
                   if signature in aliases or explicit(behavior, PATTERNS[k])]
        if not classes:
            return dict(status='out_of_scope', candidate_types=[],
                        reasons=['no_explicit_visual_class'], review_flags=flags)
        if len(classes) != 1:
            reasons.append('composite_or_conflicting_class_requires_review')
        for cls in classes:
            support = [t for t in raw if explicit(t, PATTERNS[cls])]
            if not support:
                reasons.append('normalized_class_not_supported_by_original_description')
            elif len(support) != len(raw):
                reasons.append('mixed_source_semantics_require_review')
            if cls == 'surprise' and not any(surprise_cues(t) >= 2 for t in support):
                reasons.append('surprise_requires_original_multicue_evidence')
            if cls in ('head_nod', 'head_shake') and any(re.search(REPEATED, t, re.I) for t in [behavior] + raw):
                reasons.append('multiple_cycles_without_individual_boundaries')
        if re.search(TRANSITION, behavior, re.I) or ' | change=' in origin.get('signature', ''):
            reasons.append('state_transition_boundaries_require_review')
    else:
        if anchored:
            classes = ['listener_vocalization']
            flags += ['word_is_not_necessarily_complete_response', 'listener_ownership_unverified']
        else:
            audio_pattern = SPEECH + '|' + PATTERNS['laugh']
            vocal = explicit(behavior, audio_pattern) or signature in {'vocalize', 'laugh', 'chuckle', 'giggle'}
            bodily = any(re.search(BREATH, t, re.I) for t in [behavior] + raw)
            if not vocal and not bodily:
                return dict(status='out_of_scope', candidate_types=[],
                            reasons=['no_explicit_audio_vocalization'], review_flags=flags)
            classes = ['listener_vocalization']
            if bodily:
                reasons.append('breathing_or_bodily_sound_response_requires_review')
            if not any(explicit(t, audio_pattern) for t in raw):
                reasons.append('normalized_class_not_supported_by_original_description')
            flags += ['listener_ownership_unverified', 'response_function_unverified']
            if 'multiple_word_anchors_overlap' in flags or 'word_timing_conflicts_with_asr' in flags:
                reasons.append('audio_word_alignment_conflict')
    if not anchored and ('source_interpretation_uncertain' in flags
                         or any(re.search(UNCERTAIN, t, re.I) for t in raw)):
        reasons.append('source_observation_uncertain')
    if event.get('unverified_internal_gaps'):
        reasons.append('unverified_gap_in_event_support')
    if 'normalization_missing_or_invalid_original_retained' in flags:
        reasons.append('normalization_incomplete')
    return dict(status='needs_review' if reasons else 'mapped_candidate',
                candidate_types=classes, reasons=sorted(set(reasons)),
                review_flags=sorted(set(flags + ['taxonomy_mapping_unverified'])))


def map_sample(label, provenance):
    """Preserve every input event, including out-of-scope/ambiguous observations."""
    duration = label['duration_s']
    if type(duration) not in (int, float) or not math.isfinite(duration) or duration <= 0:
        raise ValueError('Invalid duration')
    if label.get('time_origin') != 'clip_start_seconds':
        raise ValueError('Unsupported time origin')
    events = label['events']
    ids = [e['event_id'] for e in events]
    if len(ids) != len(set(ids)) or set(ids) != set(provenance['events']):
        raise ValueError('Open-event/provenance identity mismatch')
    candidates = provenance['source_candidates']
    sources = {c['id']: c for c in candidates}
    if len(sources) != len(candidates):
        raise ValueError('Duplicate source candidate IDs')
    decisions, mapped = [], []
    for event in events:
        interval(event, duration)
        if event['modality'] not in ('visual', 'audio') or not isinstance(event['behavior'], str):
            raise ValueError('Invalid modality/description')
        origin = provenance['events'][event['event_id']]
        source_ids = origin['source_candidate_ids']
        if not source_ids or any(i not in sources for i in source_ids):
            raise ValueError('Missing original source evidence')
        selected = [sources[i] for i in source_ids]
        if any(s['modality'] != event['modality'] for s in selected):
            raise ValueError('Cross-modal provenance mismatch')
        if origin.get('boundary_method') == 'asr_word_anchor':
            anchors = [s for s in selected if s['source'] == 'asr_aligner']
            if (len(anchors) != 1 or event['modality'] != 'audio'
                    or (event['start'], event['end'], event.get('text')) !=
                    (anchors[0]['start'], anchors[0]['end'], anchors[0]['behavior'])):
                raise ValueError('ASR anchor text/boundaries changed')
        decision = decide(event, origin, selected)
        decision.update(open_event_id=event['event_id'], open_event=deepcopy(event),
                        source_candidate_ids=list(source_ids), origin=deepcopy(origin),
                        source_candidates=deepcopy(selected))
        decisions.append(decision)
        if decision['status'] == 'mapped_candidate':
            target = deepcopy(event)
            target.update(type=decision['candidate_types'][0],
                          event_id='mapped-' + event['event_id'], source_open_event_id=event['event_id'],
                          mapping_status='mapped_candidate', verification='unverified_model_candidate',
                          review_flags=decision['review_flags'], human_verified=False,
                          event_unit='word' if origin.get('boundary_method') == 'asr_word_anchor' else 'source_event')
            mapped.append(target)
    return {
        'schema': 'listen-seven-class-candidates-v1', 'version': VERSION,
        'mapping_version': MAPPING_VERSION, 'sample_id': label['sample_id'],
        'duration_s': duration, 'time_origin': 'clip_start_seconds', 'taxonomy': list(TAXONOMY),
        'events': mapped, 'coverage': {k: [] for k in TAXONOMY},
        'unknown': {k: [[0, duration]] for k in TAXONOMY},
        'K_train': [], 'human_ground_truth': False, 'training_admitted': False,
        'unlabeled_time_policy': 'unknown',
        'coverage_semantics': 'Reviewed support only; mapped candidates do not certify positive or negative coverage.',
        'decisions': decisions, 'upstream_exclusions': deepcopy(provenance.get('excluded', [])),
        'upstream_audit': deepcopy(provenance.get('audit', [])),
    }


def policy():
    return dict(version=VERSION, mapping_version=MAPPING_VERSION, taxonomy=list(TAXONOMY),
                names=NAMES, patterns=PATTERNS, signature_aliases={k: sorted(v) for k, v in ALIASES.items()},
                immutable_open_data=True, changes_timestamps=False, merges_occurrences=False,
                requires_source_modality=True, inferred_negative_coverage=False,
                word_level_candidates_are_not_utterance_segmentation=True,
                surprise_requires_original_multicue_evidence=True,
                implementation='src/listen_benchmark/seven_class.py')
