"""Reconcile overlapping observations into audio response episodes, not word counts.

Categories remain model-authored. This layer matches literal content/identical
descriptions and existing time support; it does not reclassify behavior or infer
new silence, speech boundaries, or repetition counts from text alone.
"""
from copy import deepcopy
import re
import unicodedata
from .open_merge import quoted_content

VERSION = 'listen-response-episodes-v1'
EPS = 1e-8


def identity(event):
    text = event.get('text')
    if not text:
        quotes = quoted_content(event['behavior'])
        if len(quotes) == 1:
            text = next(iter(quotes))
    if text:
        # Case/punctuation are transcription variants, not evidence of new words.
        value = unicodedata.normalize('NFKC', text).casefold()
        value = ' '.join(re.findall(r'\w+', value))
        if value:
            return ('literal_content', value)
    return ('same_description', ' '.join(event['behavior'].casefold().split()).rstrip('.'))


def intersects(event, members, touch=False):
    a, b = min(e['start'] for e in members), max(e['end'] for e in members)
    return min(event['end'], b) - max(event['start'], a) >= -EPS if touch else min(event['end'], b) > max(event['start'], a) + EPS


def components(events, touch=False):
    groups = []
    for event in sorted(events, key=lambda e: (e['start'], e['end'], e['event_id'])):
        if groups and intersects(event, groups[-1], touch):
            groups[-1].append(event)
        else:
            groups.append([event])
    return groups


def consolidate_responses(sample):
    """Keep raw observations elsewhere; derive one record per supported response bout.

    ASR observations that overlap/touch form continuous same-content support.
    A broad non-ASR observation may attach to one anchor component, but cannot
    bridge two components separated by silence/unobserved time. Ambiguous support
    remains explicitly unresolved, rather than being counted as a third response.
    """
    original = sample['events']
    ids = [e['event_id'] for e in original]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate observation IDs')
    duration = sample['duration_s']
    for e in original:
        if not (0 <= e['start'] < e['end'] <= duration):
            raise ValueError('Invalid observation interval')
    pool, retained = {}, []
    for e in original:
        if (e['modality'] == 'audio' and e.get('mapping_status') == 'mapped_candidate'
                and e.get('categories') == ['listener_vocalization']):
            pool.setdefault(identity(e), []).append(e)
        else:
            retained.append(deepcopy(e))
    groups, unresolved = [], []
    for key, events in pool.items():
        anchors = [e for e in events if e.get('event_unit') == 'word']
        # Derive components BEFORE adding loose model intervals so those intervals
        # cannot expand/chain the components across distinct anchored responses.
        anchor_groups = components(anchors, touch=True)
        attached = [list(g) for g in anchor_groups]
        free = []
        for e in events:
            if e.get('event_unit') == 'word':
                continue
            matches = [i for i, g in enumerate(anchor_groups) if intersects(e, g)]
            if len(matches) == 1:
                attached[matches[0]].append(e)
            elif len(matches) > 1:
                unresolved.append(dict(observation=deepcopy(e),
                    reason='one_model_interval_overlaps_separate_anchored_responses',
                    candidate_anchor_groups=[[x['event_id'] for x in anchor_groups[i]] for i in matches]))
            else:
                free.append(e)
        groups.extend((key, g, True) for g in attached)
        groups.extend((key, g, False) for g in components(free))
    trace = {}
    for key, members, anchored in groups:
        members = sorted(members, key=lambda e: (e['start'], e['end'], e['event_id']))
        anchors = [e for e in members if e.get('event_unit') == 'word']
        support = anchors if anchored else members
        # Select one existing natural-language description, never concatenate every
        # "yeah" observation into a transcript implying repetitions.
        representative = support[0]
        event = deepcopy(representative)
        member_ids = [e['event_id'] for e in members]
        event.update(event_id='episode-' + min(member_ids),
            start=min(e['start'] for e in support), end=max(e['end'] for e in support),
            source_open_event_id=representative['event_id'], member_event_ids=member_ids,
            event_unit='vocal_response_episode', merge_version=VERSION,
            boundary_method='continuous_same_content_asr_support' if anchored else 'overlapping_same_content_model_support',
            behavior_variants=list(dict.fromkeys(e['behavior'] for e in members)),
            word_observations=[{k: e[k] for k in ('event_id', 'start', 'end', 'text')} for e in anchors],
            word_observations_role='unaltered_alignment_evidence_not_a_transcript_or_repetition_count',
            repetition_count=None, human_verified=False,
            review_flags=sorted({f for e in members for f in e.get('review_flags', [])}
                                | {'response_episode_unverified'}),
            source_descriptions=[dict(source, observation_id=e['event_id'])
                                 for e in members for source in e.get('source_descriptions', [])])
        event['overlapping_visual_event_ids'] = [e['event_id'] for e in original
            if e['modality'] == 'visual' and e.get('mapping_status') == 'mapped_candidate'
            and min(e['end'], event['end']) > max(e['start'], event['start'])]
        trace[event['event_id']] = dict(member_event_ids=member_ids, observations=deepcopy(members),
            identity=list(key), boundary_support_event_ids=[e['event_id'] for e in support],
            group_claim='One continuous response episode; internal word repetition is not inferred.')
        retained.append(event)
    retained.sort(key=lambda e: (e['start'], e['end'], e['modality'], e['event_id']))
    assigned = [i for t in trace.values() for i in t['member_event_ids']]
    assigned += [e['event_id'] for e in retained if e['modality'] != 'audio' or e.get('event_unit') != 'vocal_response_episode']
    assigned += [x['observation']['event_id'] for x in unresolved]
    if len(assigned) != len(set(assigned)) or set(assigned) != set(ids):
        raise AssertionError('Lost or duplicated source observations')
    result = deepcopy(sample)
    result.update(schema='listen-response-episode-labels-v1', merge_version=VERSION,
                  events=retained, unresolved_observations=unresolved,
                  observation_count=len(original), observations_file='observations.jsonl')
    return result, trace
