"""Validate open-vocabulary merge proposals, preserving occurrences and provenance."""
from collections import Counter
import math
import re


def overlap(a,b):
    return min(a['end'],b['end'])>max(a['start'],b['start'])


def boundary_distance(a,b):
    return abs(a['start']-b['start'])+abs(a['end']-b['end'])


def consolidate(candidates, proposals, duration, tolerance=.2):
    """Proposals are suggestions, never authoritative timestamps or verified labels."""
    by_id={e['id']:e for e in candidates}
    if len(by_id)!=len(candidates):raise ValueError('Duplicate input candidate IDs')
    for e in candidates:
        if not (all(type(e[k]) in (int,float) and math.isfinite(e[k]) for k in ('start','end')) and 0<=e['start']<e['end']<=duration):
            raise ValueError('Invalid input candidate interval')
    events=[];excluded=[];audit=[];covered=set()

    def emit(members,behavior,mode,flags):
        selected=[by_id[m] for m in members];anchor=[e for e in selected if e['source']=='asr_aligner']
        gaps=[]
        if anchor:
            chosen=anchor[0];start,end=chosen['start'],chosen['end'];method='asr_anchor'
            flags.append('asr_transcription_unverified')
        elif mode=='continuous_state':
            ordered=sorted(selected,key=lambda e:e['start']);start=ordered[0]['start'];end=ordered[0]['end']
            for e in ordered[1:]:
                if e['start']>end:gaps.append([end,e['start']])
                end=max(end,e['end'])
            method='union_of_reported_continuous_state_support'
            if gaps:flags.append('continuity_across_small_gap_unverified')
        else:
            chosen=min(selected,key=lambda e:(sum(boundary_distance(e,x) for x in selected),-e.get('support',1),e['start'],e['id']))
            start,end=chosen['start'],chosen['end'];method='source_interval' if len(selected)==1 else 'observed_boundary_medoid'
        if mode!='continuous_state' and any(abs(e['start']-start)>tolerance+1e-8 or abs(e['end']-end)>tolerance+1e-8 for e in selected):
            flags.append('source_boundary_disagreement')
        result={'start':start,'end':end,'modality':selected[0]['modality'],'behavior':behavior,
                'source_candidate_ids':members,'merge_mode':mode,'boundary_method':method,
                'review_flags':sorted(set(flags)),'human_verified':False}
        if anchor:result['text']=anchor[0]['behavior']
        if gaps:result['unverified_internal_gaps']=gaps
        events.append(result);covered.update(members)

    for modality in ['visual','audio']:
        pool={k for k,e in by_id.items() if e['modality']==modality};proposal=proposals.get(modality)
        if not pool:continue
        if not proposal or proposal.get('finish_reason')!='stop' or not isinstance(proposal.get('parsed'),dict):
            audit.append({'modality':modality,'problem':'missing_or_incomplete_proposal'})
            continue
        parsed=proposal['parsed'];groups=parsed.get('groups',[]);drops=parsed.get('excluded',[])
        if not isinstance(groups,list) or not isinstance(drops,list):continue
        references=Counter(m for g in groups+drops if isinstance(g,dict) for m in g.get('members',[]) if isinstance(m,str))
        for group in groups:
            if not isinstance(group,dict):continue
            ids=group.get('members',[]);mode=group.get('mode');behavior=group.get('behavior')
            reason=None
            if (not isinstance(ids,list) or not ids or any(not isinstance(m,str) or m not in pool or references[m]!=1 for m in ids)):
                reason='invalid_or_repeated_member_reference'
            elif not isinstance(behavior,str) or not behavior.strip():reason='empty_description'
            elif mode not in {'singleton','same_occurrence','continuous_state'}:reason='invalid_merge_mode'
            elif mode=='singleton' and len(ids)!=1:reason='singleton_has_multiple_members'
            else:
                selected=[by_id[m] for m in ids];anchors=[e for e in selected if e['source']=='asr_aligner']
                if len(anchors)>1:reason='distinct_word_occurrences_cannot_merge'
                elif anchors and mode=='continuous_state':reason='word_occurrence_is_not_continuous_state'
                elif anchors and not re.search(r'(?<!\w)'+re.escape(anchors[0]['behavior'])+r'(?!\w)',behavior,re.I):reason='original_word_not_preserved'
                elif mode=='same_occurrence' and any(not overlap(a,b) for i,a in enumerate(selected) for b in selected[i+1:]):reason='nonoverlapping_occurrences_cannot_merge'
                elif mode=='continuous_state':
                    ordered=sorted(selected,key=lambda e:e['start']);end=ordered[0]['end']
                    for e in ordered[1:]:
                        if e['start']>end+tolerance+1e-8:reason='gap_breaks_continuity';break
                        end=max(end,e['end'])
            if reason:
                audit.append({'modality':modality,'proposal':group,'problem':reason});continue
            flags=['semantic_rewrite_unverified']
            if group.get('uncertain'):flags.append('model_reported_uncertainty')
            emit(ids,behavior.strip(),mode,flags)
        for drop in drops:
            if not isinstance(drop,dict):continue
            ids=drop.get('members',[])
            if not ids or any(not isinstance(m,str) or m not in pool or references[m]!=1 or m in covered for m in ids):
                audit.append({'modality':modality,'proposal':drop,'problem':'invalid_exclusion_reference'});continue
            if any(by_id[m]['source']=='asr_aligner' for m in ids):
                audit.append({'modality':modality,'proposal':drop,'problem':'asr_anchor_exclusion_requires_review'});continue
            excluded.append({'source_candidate_ids':ids,'reason':str(drop.get('reason','')),'status':'model_proposed_non_event','human_verified':False})
            covered.update(ids)
    for e in candidates:
        if e['id'] not in covered:
            behavior='Says '+json_quote(e['behavior'])+'.' if e['source']=='asr_aligner' else e['behavior']
            emit([e['id']],behavior,'singleton',['proposal_rejected_or_missing_original_retained'])
    events.sort(key=lambda e:(e['start'],e['end'],e['modality']))
    for index,event in enumerate(events):event['event_id']=f'event-{index:04d}'
    assigned=[m for e in events+excluded for m in e['source_candidate_ids']]
    if Counter(assigned)!=Counter(by_id.keys()):raise AssertionError('Lost or duplicated source candidates')
    return {'events':events,'excluded':excluded,'audit':audit,'input_candidates':len(candidates),
            'human_ground_truth':False,'training_admitted':False,'unlabeled_time_policy':'unknown',
            'negative_coverage':{'visual':[],'audio':[]}}


def json_quote(text):
    import json
    return json.dumps(text,ensure_ascii=False)


def clean_description(text):
    """Remove explanation clauses, without mapping observations to an action vocabulary."""
    text=re.split(r',\s*(?:indicating|suggesting|reflecting|possibly|likely)\b',text,flags=re.I)[0]
    text=re.split(r'\s+while\s+(?:maintaining\s+)?(?:attentive|engaged|thoughtful)\b',text,flags=re.I)[0]
    text=re.sub(r',?\s*\battentive\s+(?=expression\b)',' ',text,flags=re.I)
    return text.strip().rstrip('.')+'.'


def normalized_key(text):
    return re.sub(r'\s+',' ',text.casefold()).strip().rstrip('.')


def quoted_content(text):
    return {normalized_key(a or b) for a,b in re.findall(r"[\"“]([^\"”]+)[\"”]|(?:^|[\s:])'([^']+)'",text)}


def semantic_guard_key(signature,behavior):
    """Retain literal word identity and stated direction inside free-form merge keys."""
    key=normalized_key(signature);quotes=quoted_content(behavior)
    if quotes:key+=' | literal='+','.join(sorted(quotes))
    if re.search(r'\b(head|gaze|look|lean|turn|move|tilt)\b',key):
        directions=set(re.findall(r'\b(upward|up|downward|down|left|right|backward|back|forward|sideways)\b',behavior.casefold()))
        directions={dict(upward='up',downward='down',backward='back').get(x,x) for x in directions}
        key+=' | direction='+','.join(sorted(directions))
    return key


def atomic_consolidate(candidates, normalizations, duration, tolerance=.2, rules=None):
    """Normalize composite observations first; only then reconcile time intervals."""
    rules=rules or {};aliases=rules.get('signature_aliases',{})
    by_id={e['id']:e for e in candidates};atoms=[];excluded=[];anchors=[];audit=[]
    if len(by_id)!=len(candidates):raise ValueError('Duplicate source IDs')
    for e in candidates:
        if not (all(type(e[k]) in (int,float) and math.isfinite(e[k]) for k in ('start','end')) and 0<=e['start']<e['end']<=duration):raise ValueError('Invalid source interval')
        common={'start':e['start'],'end':e['end'],'modality':e['modality'],'source':e['source'],'source_candidate_id':e['id']}
        if e['source']=='asr_aligner':
            anchors.append(dict(common,atom_id=e['id']+':word',signature='say '+normalized_key(e['behavior']),behavior='Says '+json_quote(e['behavior'])+'.',text=e['behavior'],temporal_kind='occurrence',flags=['asr_transcription_unverified']))
            continue
        record=normalizations.get(e['behavior']);value=record.get('value') if record else None
        valid=(isinstance(value,dict) and isinstance(value.get('atoms'),list) and all(isinstance(a,dict) and isinstance(a.get('signature'),str) and a['signature'].strip() and isinstance(a.get('behavior'),str) and a['behavior'].strip() and a.get('temporal_kind') in ('state','occurrence') for a in value['atoms']))
        if valid and not value['atoms'] and value.get('excluded_reason'):
            excluded.append({'source_candidate_id':e['id'],'reason':value['excluded_reason'],'normalization_id':record['id'],'status':'proposed_non_event','human_verified':False});continue
        if not valid or not value['atoms']:
            atoms.append(dict(common,atom_id=e['id']+':fallback',signature='unprocessed '+e['id'],behavior=clean_description(e['behavior']),temporal_kind='occurrence',flags=['normalization_missing_or_invalid_original_retained']))
            audit.append({'source_candidate_id':e['id'],'problem':'normalization_missing_or_invalid'});continue
        seen=set();observed=value['atoms'];primary=None;folded=[]
        present={aliases.get(a['signature'],a['signature']) for a in observed}
        for possible,details in rules.get('supporting_detail_relations',{}).items():
            if possible in present:
                if primary is None or primary in details:primary=possible
        if primary:
            details=rules['supporting_detail_relations'][primary]
            folded=[a for a in observed if aliases.get(a['signature'],a['signature']) in details
                    and (a['temporal_kind']=='state' or aliases.get(a['signature'],a['signature'])=='smile')]
        for i,a in enumerate(observed):
            if a in folded:continue
            canonical=aliases.get(a['signature'],a['signature'])
            context_rule=next((r for r in rules.get('contextual_aliases',[]) if a['signature'] in r['signatures'] and re.search(r['behavior_pattern'],a['behavior'],re.I)),None)
            if context_rule:canonical=context_rule['canonical']
            key=semantic_guard_key(canonical,a['behavior'])
            kind=rules.get('temporal_overrides',{}).get(canonical,a['temporal_kind'])
            behavior=rules.get('descriptions',{}).get(canonical,a['behavior'])
            if context_rule:
                kind=context_rule.get('temporal_kind',kind);behavior=context_rule.get('behavior',behavior)
            # A change towards a different state must not extend the preceding state.
            if re.search(r'\b(transition\w*|fade\w*|cease\w*|stop\w*)\b',a['behavior'],re.I):
                kind='occurrence';key+=' | change='+normalized_key(a['behavior']);behavior=a['behavior']
            # Multiple identical atoms in a single description are one observation,
            # not evidence of multiple independently timed occurrences.
            if (key,kind) in seen:continue
            seen.add((key,kind))
            atoms.append(dict(common,atom_id=e['id']+f':a{i}',signature=key,behavior=clean_description(behavior),temporal_kind=kind,normalization_id=record['id'],supporting_details=folded if canonical==primary else [],flags=['normalization_unverified']+(['source_interpretation_uncertain'] if a.get('uncertain') else [])))
    # Coherent, contiguous low-level mouth fragments are kept as one observed
    # process, retaining every internal transition in provenance. Isolated gestures,
    # word anchors, blinks, and head cycles do not enter this operation.
    process_keys=set(rules.get('compound_process_predicates',[]));process=[];other=[]
    for a in atoms:
        (process if a['modality']=='visual' and a['signature'] in process_keys else other).append(a)
    bouts=[]
    for a in sorted(process,key=lambda x:(x['start'],x['end'])):
        if bouts and a['start']<=max(x['end'] for x in bouts[-1])+tolerance+1e-8:bouts[-1].append(a)
        else:bouts.append([a])
    for bout in bouts:
        # Require at least three observations and more than one kind of fragment.
        if len(bout)>=3 and len({a['signature'] for a in bout})>1:
            for a in bout:
                a['original_atomic_behavior']=a['behavior'];a['original_signature']=a['signature']
                a.update(signature='observed continuous mouth movements',behavior=rules['compound_process_description'],temporal_kind='state')
                a['flags'].append('microgesture_process_summary')
        other.extend(bout)
    atoms=other
    # Every ASR word remains its own event; attach only an explicitly matching word
    # report with overlapping support. No invented alignment from nearby timestamps.
    anchor_groups=[[a] for a in anchors];rest=[]
    for atom in atoms:
        raw=atom['behavior']
        quoted=quoted_content(raw)
        matches=[i for i,a in enumerate(anchors) if atom['modality']=='audio' and overlap(atom,a)
                 and (normalized_key(a['text']) in quoted or atom['signature']==a['signature'])]
        if len(matches)==1:anchor_groups[matches[0]].append(atom)
        else:
            if len(matches)>1:atom['flags'].append('multiple_word_anchors_overlap')
            if not matches and atom['modality']=='audio' and any(normalized_key(a['text']) in quoted for a in anchors):
                atom['flags'].append('word_timing_conflicts_with_asr')
            rest.append(atom)
    buckets={}
    for a in rest:buckets.setdefault((a['modality'],a['signature'],a['temporal_kind']),[]).append(a)
    groups=[]
    for (_,_,kind),values in buckets.items():
        clusters=[]
        for a in sorted(values,key=lambda e:(e['start'],e['end'])):
            if kind=='state':
                found=next((g for g in reversed(clusters) if a['start']<=max(x['end'] for x in g)+tolerance+1e-8),None)
            else:
                found=next((g for g in clusters if all(overlap(a,b) and abs(a['start']-b['start'])<=tolerance+1e-8 and abs(a['end']-b['end'])<=tolerance+1e-8 for b in g)),None)
            if found is None:clusters.append([a])
            else:found.append(a)
        groups.extend(clusters)
    groups.extend(anchor_groups);events=[];trace={}
    for g in groups:
        anchor=next((a for a in g if a['source']=='asr_aligner'),None);flags={f for a in g for f in a['flags']};gaps=[]
        if anchor:
            representative=anchor;start,end=anchor['start'],anchor['end'];method='asr_word_anchor'
            if any(abs(a['start']-start)>tolerance+1e-8 or abs(a['end']-end)>tolerance+1e-8 for a in g):flags.add('source_boundary_disagreement')
        elif g[0]['temporal_kind']=='state':
            ordered=sorted(g,key=lambda a:a['start']);start=ordered[0]['start'];end=ordered[0]['end']
            for a in ordered[1:]:
                if a['start']>end:gaps.append([end,a['start']])
                end=max(end,a['end'])
            representative=min(g,key=lambda a:(len(a['behavior']),a['atom_id']));method='reported_continuous_support_union'
            if gaps:flags.add('continuity_across_small_gap_unverified')
        else:
            representative=min(g,key=lambda a:(sum(boundary_distance(a,b) for b in g),a['atom_id']));start,end=representative['start'],representative['end'];method='observed_occurrence_interval'
        event={'start':start,'end':end,'modality':representative['modality'],'behavior':representative['behavior'],'review_flags':sorted(flags)}
        if anchor:event['text']=anchor['text']
        if gaps:event['unverified_internal_gaps']=gaps
        events.append((event,{'source_candidate_ids':sorted({a['source_candidate_id'] for a in g}),'atom_ids':[a['atom_id'] for a in g],'boundary_method':method,'signature':representative['signature'],'temporal_kind':representative['temporal_kind'],'atoms':g}))
    events.sort(key=lambda pair:(pair[0]['start'],pair[0]['end'],pair[0]['modality']))
    result=[]
    for i,(event,origin) in enumerate(events):
        event['event_id']=f'event-{i:04d}';result.append(event);trace[event['event_id']]=origin
    represented={i for origin in trace.values() for i in origin['source_candidate_ids']}
    discarded={e['source_candidate_id'] for e in excluded}
    if represented&discarded or represented|discarded!=set(by_id):raise AssertionError('Unaccounted source candidate')
    all_atoms=[a['atom_id'] for a in atoms+anchors];used=[i for x in trace.values() for i in x['atom_ids']]
    if Counter(all_atoms)!=Counter(used):raise AssertionError('Lost or duplicated atomic observations')
    return {'events':result,'trace':trace,'excluded':excluded,'audit':audit,'input_candidates':len(candidates),
            'atomic_observations':len(atoms)+len(anchors),'human_ground_truth':False,'training_admitted':False,
            'unlabeled_time_policy':'unknown','negative_coverage':{'visual':[],'audio':[]}}
