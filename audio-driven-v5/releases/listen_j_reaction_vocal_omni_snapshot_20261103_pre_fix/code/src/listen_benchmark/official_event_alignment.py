"""Align existing events to official timing without rewriting their language.

This is an auditable, conservative text/time matcher, not another model pass.
The waveform gate and this event boundary policy are deliberately separate.
"""
import copy
import math
import re
import unicodedata
from collections import Counter

POLICY = dict(version='official-event-alignment-v1', word_boundary_tolerance_s=.2,
              phrase_max_internal_gap_s=.3, vad_min_iou=.5,
              word_matching='normalized_literal_only_unique_one_to_one',
              conflicting_text='preserve_our_text_and_intersect',
              padding_is_not_event_duration=True, new_events_from_official=False)


def tokens(text):
    # Punctuation/case only; no translation or synonym substitution.
    return tuple(re.findall(r'[^\W_]+', unicodedata.normalize('NFKC', text or '').casefold()))


def utterance(event):
    if event.get('text'):
        return tokens(event['text']), 'original_text_field'
    quoted = re.findall(r'''["'“‘]([^"'”’]+)["'”’]''', event.get('description', ''))
    if len(quoted) == 1:
        return tokens(quoted[0]), 'quoted_original_description'
    return (), 'no_literal_utterance'


def overlap(a, b):
    return max(0., min(a['end'], b['end']) - max(a['start'], b['start']))


def intersection(interval, support):
    return [dict(start=max(interval['start'], x['start']), end=min(interval['end'], x['end']))
            for x in support if overlap(interval, x) > 1e-9]


def align_events(events, official, gate):
    """Return derived events plus one audit row for every original audio event.

    `intervals` is authoritative when an event has disconnected support. It does
    not assert that fragments are different utterances. `start/end` is its envelope.
    """
    sr = gate['sample_rate']; duration = gate['samples'] / sr
    support = [dict(start=a/sr, end=b/sr) for a,b in gate['positive_intervals_samples']]
    audio = [e for e in events if e.get('modality') == 'audio']
    if len({e['event_id'] for e in events}) != len(events):
        raise ValueError('Duplicate event IDs')
    for e in audio:
        if not all(isinstance(e.get(k), (int,float)) and math.isfinite(e[k]) for k in ('start','end')) or e['end'] <= e['start']:
            raise ValueError('Invalid source event interval')
    def clipped(kind):
        out=[]
        for x in official[kind]:
            a,b=max(0.,x['start']),min(duration,x['end'])
            if b>a:out.append(dict(x,start=a,end=b,source_interval=dict(start=x['start'],end=x['end'])))
        return sorted(out,key=lambda x:(x['start'],x['end'],x['id']))
    words=clipped('words'); vad=clipped('vad')
    base={e['event_id']:intersection(e,support) for e in audio}
    proposals={}; attempts={}; reasons={}
    for e in audio:
        eid=e['event_id']; wanted,origin=utterance(e); options=[]
        if base[eid] and wanted:
            for i in range(len(words)):
                seen=(); chunk=[]
                for w in words[i:]:
                    if chunk and (w.get('parent') != chunk[0].get('parent') or w['start']-chunk[-1]['end']>POLICY['phrase_max_internal_gap_s']):break
                    seen+=tokens(w.get('text'));chunk.append(w)
                    if seen!=wanted[:len(seen)]:break
                    if seen==wanted:
                        a,b=chunk[0]['start'],chunk[-1]['end']
                        shifts=[abs(a-e['start']),abs(b-e['end'])]
                        if max(shifts)<=POLICY['word_boundary_tolerance_s']+1e-8:
                            options.append(dict(start=a,end=b,words=copy.deepcopy(chunk),boundary_shifts_s=shifts,literal_source=origin))
                        break
        attempts[eid]=options
        if len(options)==1:proposals[eid]=options[0]
        elif len(options)>1:reasons[eid]='multiple_matching_word_occurrences'
    claims=Counter(w['id'] for p in proposals.values() for w in p['words'])
    # Never let two source events collapse onto one official word occurrence.
    for eid,p in list(proposals.items()):
        if any(claims[w['id']]>1 for w in p['words']):
            del proposals[eid];reasons[eid]='official_word_claimed_by_multiple_events'
    ordered=sorted(audio,key=lambda e:(e['start'],e['end'],e['event_id']))
    crossing=set()
    for i,e in enumerate(ordered):
        if e['event_id'] not in proposals:continue
        for later in ordered[i+1:]:
            if later['event_id'] in proposals and proposals[e['event_id']]['start']>proposals[later['event_id']]['start']:
                crossing.update([e['event_id'],later['event_id']])
    for eid in crossing:del proposals[eid];reasons[eid]='word_assignment_would_reverse_event_order'

    output=[]; excluded=[]; audit=[]
    for e in events:
        if e.get('modality')!='audio':output.append(copy.deepcopy(e));continue
        eid=e['event_id']; ranges=base[eid]; evidence=[]; method='model_intersection'; reason=reasons.get(eid,'no_unique_literal_word_match')
        local_words=[copy.deepcopy(w) for w in words if overlap(e,w)>0]
        if not ranges:
            method='excluded';reason='outside_official_audio_gate'
        elif eid in proposals:
            p=proposals[eid];ranges=intersection(p,support);evidence=p['words'];method='official_words';reason='unique_literal_words_with_close_boundaries'
        else:
            # This selects timing evidence only, never changes text or categories.
            nonlexical=not utterance(e)[0] and bool(re.search(r'\b(laugh\w*|chuckl\w*|giggl\w*|non[- ]speech)\b|笑声',e.get('description',''),re.I))
            vs=[v for v in vad if overlap(e,v)>0]
            if nonlexical and len(vs)==1:
                v=vs[0];shared=[x for x in audio if overlap(x,v)>0]
                ov=overlap(e,v);iou=ov/((min(duration,e['end'])-max(0,e['start']))+(v['end']-v['start'])-ov)
                if len(shared)==1 and not any(overlap(w,v)>0 for w in words) and iou>=POLICY['vad_min_iou']:
                    ranges=[dict(start=v['start'],end=v['end'])];evidence=[copy.deepcopy(v)];method='official_vad';reason='isolated_nonlexical_event_single_vad_no_words'
                else:reason='vad_shared_with_other_events_or_words_or_insufficient_overlap'
            elif nonlexical and len(vs)>1:reason='multiple_vad_regions_preserve_event_internal_boundaries'
        source=dict(start=e['start'],end=e['end'])
        changed=ranges!=[source]
        row=dict(event_id=eid,description=e.get('description'),text=e.get('text'),source_interval=source,
                 intervals=ranges,method=method,reason=reason,boundary_changed=changed,
                 official_evidence=evidence,overlapping_official_words=local_words,
                 literal_match_candidates=attempts[eid],text_rewritten=False,
                 boundary_uncertain=method=='model_intersection',
                 text_not_literal_match=bool(utterance(e)[0] and local_words and method!='official_words'))
        audit.append(row)
        if method=='excluded':excluded.append(dict(original_event=copy.deepcopy(e),alignment=row));continue
        final=copy.deepcopy(e)
        final.update(start=ranges[0]['start'],end=ranges[-1]['end'],intervals=ranges,
                     source_interval=source,source_boundary_method=e.get('boundary_method'),
                     boundary_method=method,boundary_alignment=row,
                     disjoint_support=len(ranges)>1)
        output.append(final)
    return dict(schema='official-aligned-events-v1',policy=POLICY,events=output,
                excluded_audio_events=excluded,audio_event_alignment=audit,
                audio_events_retained=sum(e.get('modality')=='audio' for e in output),
                text_rewritten=False,visual_events_unchanged=True,training_applied=False,
                interval_usage='Use intervals, not start/end envelope, for audio supervision. Fragments are not new event counts.')
