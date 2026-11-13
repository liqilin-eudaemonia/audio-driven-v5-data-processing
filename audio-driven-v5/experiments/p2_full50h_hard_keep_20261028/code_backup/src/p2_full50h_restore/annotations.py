from copy import deepcopy
from listen_benchmark.semantic_prompt import compile_plan,validate_proposal,digest


def compiled_events(record,answer):
    validate_proposal(record,answer)
    source={e['event_id']:e for e in record['events']};events=[]
    for e in answer['visual_events']:
        original=source[e['timing_source_id']]
        events.append(dict(event_id='semantic-'+e['id'],modality='visual',description=e['description'],text=None,
            start=original['start'],end=original['end'],intervals=deepcopy(original.get('intervals',[dict(start=original['start'],end=original['end'])])),
            timing_source_id=e['timing_source_id'],source_event_ids=list(dict.fromkeys(q['source_event_id'] for q in e['evidence'])),evidence=deepcopy(e['evidence'])))
    return events


def recover(record,plan):
    """Validate connected source groups independently; failed groups stay verbatim."""
    visual={e['event_id']:e for e in record['events'] if e['modality']=='visual'}
    original=lambda ids:[deepcopy(e) for e in record['events'] if e['modality']=='visual' and e['event_id'] in ids]
    audio=[deepcopy(e) for e in record['events'] if e['modality']=='audio']
    if not isinstance(plan,dict) or set(plan)!={'decisions','conflicts'} or not isinstance(plan['decisions'],list) or not isinstance(plan['conflicts'],list):
        return deepcopy(record['events']),dict(mode='original_only',fallback_source_ids=list(visual),safe_source_ids=[],reason='No complete structurally valid proposal')
    decisions={};bad=set();edges={sid:set() for sid in visual}
    for d in plan['decisions']:
        if not isinstance(d,dict) or d.get('source_event_id') not in visual:continue
        sid=d['source_event_id']
        if sid in decisions:bad.add(sid)
        decisions[sid]=d
        targets=d.get('targets',[])
        if not isinstance(targets,list):bad.add(sid);continue
        for t in targets:
            if not isinstance(t,str) or t not in visual:bad.add(sid);continue
            edges[sid].add(t);edges[t].add(sid)
    # Cross-group conflict claims must be validated together, never silently dropped.
    invalid_conflict=False
    for c in plan['conflicts']:
        if not isinstance(c,dict):invalid_conflict=True;continue
        refs=c.get('source_event_ids',[])+c.get('selected_event_ids',[]) if isinstance(c.get('source_event_ids'),list) and isinstance(c.get('selected_event_ids'),list) else []
        if not refs or any(not isinstance(x,str) or x not in visual for x in refs):invalid_conflict=True;continue
        for sid in refs[1:]:edges[refs[0]].add(sid);edges[sid].add(refs[0])
    if invalid_conflict:
        return deepcopy(record['events']),dict(mode='original_only',fallback_source_ids=list(visual),safe_source_ids=[],reason='Invalid conflict references')
    seen=set();events=[];safe=[];fallback=[];details=[]
    for sid in visual:
        if sid in seen:continue
        group=set();todo=[sid]
        while todo:
            x=todo.pop()
            if x in group:continue
            group.add(x);todo.extend(edges[x]-group)
        seen|=group
        subset=dict(record,events=original(group))
        try:
            if group & bad or group-decisions.keys():raise ValueError('Missing/duplicate/malformed source decision')
            candidate=dict(decisions=[decisions[x] for x in visual if x in group],conflicts=[c for c in plan['conflicts'] if set(c['source_event_ids']) & group])
            answer=compile_plan(subset,candidate)
            derived=compiled_events(subset,answer)
            inactive=[d['source_event_id'] for d in candidate['decisions'] if d['action'] in ('uncertain','non_observable')]
            # Even inactive original events are retained under the no-drop policy.
            derived+=original(inactive)
            if {e['event_id'] for e in derived} & ({e['event_id'] for e in events}|{e['event_id'] for e in audio}|(set(visual)-group)):
                raise ValueError('Derived ID collision; preserve original group')
            events+=derived;safe.extend(x for x in visual if x in group and x not in inactive);fallback.extend(inactive)
            details.append(dict(source_ids=sorted(group),status='hard_checked',inactive_preserved=inactive))
        except (ValueError,KeyError,TypeError) as exc:
            events+=original(group);fallback.extend(x for x in visual if x in group);details.append(dict(source_ids=sorted(group),status='original_preserved',reason=str(exc)))
    events+=audio;events.sort(key=lambda e:(e['start'],e['end'],e['event_id']))
    if set(safe)|set(fallback)!=set(visual) or set(safe)&set(fallback):raise AssertionError('Source coverage lost')
    if len({e['event_id'] for e in events})!=len(events):raise ValueError('Duplicate output IDs')
    return events,dict(mode='safe_components_with_original_fallback',safe_source_ids=safe,fallback_source_ids=fallback,groups=details)
