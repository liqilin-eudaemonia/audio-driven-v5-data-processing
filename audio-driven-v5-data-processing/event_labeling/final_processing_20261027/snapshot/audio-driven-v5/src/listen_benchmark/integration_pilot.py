"""Conservative independent events and co-occurrence links, with separate evidence."""
from copy import deepcopy
from collections import defaultdict
from .response_review import literal, deduplicate_tracks
from .open_merge import quoted_content

VERSION='listen-integration-pilot-v1'


def text_identity(e):
    if e.get('text'):return literal(e['text'])
    q=quoted_content(e['behavior'])
    return literal(next(iter(q))) if len(q)==1 else None


def sources(decision):
    return [dict(s,candidate_id=c['id']) for c in decision.get('source_candidates',[]) for s in c.get('sources',[])]


def passes(e,decisions):
    return {str(s.get('source'))+'::'+str(s.get('window_id')) for s in sources(decisions.get(e['event_id'],{})) if s.get('window_id')}


def key(e,decisions):
    if e['modality']=='audio' and text_identity(e):return ('audio',text_identity(e))
    d=decisions.get(e['event_id'],{});signature=d.get('origin',{}).get('signature')
    return (e['modality'],tuple(e.get('categories',[])),signature or literal(e['behavior']))


def close(a,b,tol=.2):
    return (min(a['end'],b['end'])>max(a['start'],b['start']) and
            max(abs(a['start']-b['start']),abs(a['end']-b['end']))<=tol+1e-8)


def duplicate(a,b,decisions):
    pa,pb=passes(a,decisions),passes(b,decisions)
    return bool(pa and pb and pa.isdisjoint(pb) and key(a,decisions)==key(b,decisions) and close(a,b))


def clusters(events,decisions):
    """Complete-link clusters. Ambiguous matches and shared passes remain separate."""
    result=[]
    for e in sorted(events,key=lambda x:(x['start'],x['end'],x['event_id'])):
        matches=[c for c in result if all(duplicate(e,x,decisions) for x in c)]
        if len(matches)==1:matches[0].append(e)
        else:result.append([e])
    return result


def make_event(members,index):
    # Choose an observed medoid; never expand to the union of intervals.
    rep=min(members,key=lambda e:(sum(abs(e['start']-x['start'])+abs(e['end']-x['end']) for x in members),e['event_id']))
    return dict(event_id=f'event-{index:04d}',modality=rep['modality'],description=rep['behavior'],
        text=rep.get('text'),categories=rep.get('categories',[]),start=rep['start'],end=rep['end'],
        event_unit='word_candidate' if rep.get('event_unit')=='word' else 'described_action_candidate',
        source_observation_ids=[e['event_id'] for e in members],support_observation_ids=[],
        original_descriptions=list(dict.fromkeys(e['behavior'] for e in members)),
        status=dict(content='model_candidate',count='unverified',boundary='model_candidate'),
        review_flags=sorted({f for e in members for f in e.get('review_flags',[])}),
        boundary_method='observed_medoid_no_union',training=dict(admitted=False,precise_boundary_loss=False,count_loss=False),
        duplicate_observations_removed=len(members)-1)


def integrate(sample,decisions_list,tracks,correction=None):
    decisions={d['open_event_id']:d for d in decisions_list};original=sample['events']
    mapped=[e for e in original if e['mapping_status']=='mapped_candidate']
    visual=[e for e in mapped if e['modality']=='visual']
    words=[e for e in mapped if e['modality']=='audio' and e.get('event_unit')=='word']
    audio=[e for e in mapped if e['modality']=='audio' and e.get('event_unit')!='word']
    events=[];evidence=[]
    for group in clusters(visual,decisions)+clusters(words,decisions):events.append(make_event(group,len(events)))
    word_events=[e for e in events if e['modality']=='audio']
    free=[];timing_conflicts=[]
    for obs in audio:
        identity=text_identity(obs)
        matches=[e for e in word_events if identity and identity==literal(e.get('text') or '')
                 and min(obs['end'],e['end'])>max(obs['start'],e['start'])]
        if matches:
            # Descriptions may cover several tokens; attach without changing counts/bounds.
            evidence.append(dict(observation_id=obs['event_id'],relation='descriptive_support_not_an_extra_occurrence',
                                 candidate_event_ids=[e['event_id'] for e in matches],count_resolved=False))
            for e in matches:e['support_observation_ids'].append(obs['event_id'])
        elif identity and any(identity==literal(e.get('text') or '') for e in word_events):
            timing_conflicts.append(dict(observation_id=obs['event_id'],reason='same_word_outside_asr_support_possible_extra_occurrence'))
        else:free.append(obs)
    for group in clusters(free,decisions):events.append(make_event(group,len(events)))
    unresolved=[dict(observation_id=e['event_id'],reason=e['mapping_status']) for e in original if e['mapping_status']!='mapped_candidate']+timing_conflicts
    # Multiple records in one model pass are evidence of count disagreement, not duplicate proof.
    by_id={e['event_id']:e for e in original}
    for e in events:
        p=defaultdict(set)
        for oid in e['support_observation_ids']:
            for origin in passes(by_id[oid],decisions):p[origin].add(oid)
        if any(len(v)>1 for v in p.values()):e['review_flags'].append('same_pass_support_may_contain_multiple_occurrences')
    token_clusters=deduplicate_tracks(tracks)
    token_links=[]
    for tr in tracks:
        for t in tr['tokens']:
            matches=[e['event_id'] for e in word_events if literal(e.get('text') or '')==literal(t['text']) and
                     t['time_valid'] and close(t,e)]
            token_links.append(dict(token_id=t['id'],candidate_event_ids=matches,
                status='boundary_agreement_candidate' if len(matches)==1 else 'unresolved_word_evidence',
                invalid_time_flags=t['flags']))
    before_correction=deepcopy(events)
    if correction:
        replace_ids=set(correction['replace_observation_ids'])
        removed=[e for e in events if set(e['source_observation_ids']+e['support_observation_ids'])&replace_ids]
        removed_ids={e['event_id'] for e in removed};events=[e for e in events if e['event_id'] not in removed_ids]
        evidence=[x for x in evidence if x['observation_id'] not in replace_ids]
        for x in token_links:
            if any(i in removed_ids for i in x['candidate_event_ids']):
                x['candidate_event_ids']=[i for i in x['candidate_event_ids'] if i not in removed_ids]
                x['status']='human_correction_requires_new_time_assignment'
        for x in evidence:x['candidate_event_ids']=[i for i in x['candidate_event_ids'] if i not in removed_ids]
        for i,c in enumerate(correction['events']):
            e=dict(event_id=f'human-content-{i:03d}',modality='audio',categories=['listener_vocalization'],
                source_observation_ids=[],support_observation_ids=[],original_descriptions=[],review_flags=[],
                boundary_method='human_feedback_with_explicit_uncertainty',
                training=dict(admitted=False,precise_boundary_loss=False,count_loss=False),duplicate_observations_removed=0)
            e.update(deepcopy(c));e['feedback_reference']=correction['feedback_reference'];events.append(e)
        evidence.extend(dict(observation_id=i,relation='preserved_under_human_correction_not_interval_assignment',
                             candidate_event_ids=[e['event_id'] for e in events if e.get('feedback_reference')],count_resolved=False) for i in sorted(replace_ids))
    # Evidence accounting: every original observation remains visible, including unsupported items.
    covered=set(e['observation_id'] for e in unresolved+evidence)
    covered.update(i for e in events for i in e['source_observation_ids'])
    if covered!=set(by_id):raise AssertionError(('Lost original observations',set(by_id)-covered,covered-set(by_id)))
    events.sort(key=lambda e:(e.get('start') is None,e.get('start') if e.get('start') is not None else float('inf'),
                             e.get('end') if e.get('end') is not None else float('inf'),e.get('sequence_index',0),e['event_id']))
    active_ids={e['event_id'] for e in events}
    assert all(set(x['candidate_event_ids'])<=active_ids for x in evidence+token_links)
    for e in events:
        e['source_boundary_methods']={i:decisions.get(i,{}).get('origin',{}).get('boundary_method') for i in e['source_observation_ids']}
        if any(v and 'union' in v for v in e['source_boundary_methods'].values()):
            e['review_flags'].append('inherited_interval_union_requires_boundary_review')
    groups=[]
    # One audio event anchors a possible co-occurrence group. No transitive merging.
    for audio_event in [e for e in events if e['modality']=='audio']:
        peers=[]
        if audio_event.get('start') is not None and audio_event.get('end') is not None:
            peers=[e['event_id'] for e in events if e['modality']=='visual' and e.get('start') is not None
                   and min(e['end'],audio_event['end'])>max(e['start'],audio_event['start'])]
        members=[audio_event['event_id']]+peers
        groups.append(dict(group_id=f'group-{len(groups):03d}',member_event_ids=members,
            relation='possible_temporal_cooccurrence' if peers else 'audio_event_reference',
            continuous_activity_claim=False,training_mask=None))
    grouped={i for g in groups for i in g['member_event_ids']}
    for e in events:
        if e['event_id'] not in grouped:
            groups.append(dict(group_id=f'group-{len(groups):03d}',member_event_ids=[e['event_id']],
                relation='visual_event_reference',continuous_activity_claim=False,training_mask=None))
    return dict(schema=VERSION,sample_id=sample['sample_id'],duration_s=sample['duration_s'],events=events,
        response_groups=groups,evidence_links=evidence,unresolved_observations=unresolved,
        alignment_tracks=tracks,token_clusters=token_clusters,token_event_links=token_links,
        correction=correction,pre_correction_events=before_correction,
        original_observation_count=len(original),raw_observation_ids=list(by_id),
        training_admitted=False,negative_coverage={'visual':[],'audio':[]},unlabeled_time_policy='unknown')
