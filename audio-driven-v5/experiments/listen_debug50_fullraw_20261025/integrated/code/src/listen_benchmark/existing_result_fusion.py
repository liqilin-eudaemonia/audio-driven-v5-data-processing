"""CPU-only fusion of existing observations; never reads human review or media."""
from copy import deepcopy
from collections import defaultdict
import re
from .response_review import literal
from .open_merge import quoted_content
from .integration_pilot import integrate

VERSION='listen-existing-result-fusion-verbatim-v2'

def content_key(text):return ''.join(re.findall(r'\w+',text.casefold()))

def compatible(a,b,tolerance=.2):
    return (content_key(a['text'])==content_key(b['text']) and a['pass_id']!=b['pass_id']
            and min(a['end'],b['end'])>max(a['start'],b['start'])
            and max(abs(a['start']-b['start']),abs(a['end']-b['end']))<=tolerance+1e-8)

def group_candidates(candidates):
    groups=[]
    for c in sorted(candidates,key=lambda c:(c['start'],c['end'],c['pass_id'])):
        fits=[g for g in groups if all(compatible(c,x) for x in g)]
        if len(fits)==1:fits[0].append(c)
        else:groups.append([c])
    result=[]
    for i,g in enumerate(groups):
        rep=min(g,key=lambda c:(sum(abs(c['start']-x['start'])+abs(c['end']-x['end']) for x in g),c['edge'],c['end']-c['start']))
        d=deepcopy(rep);d['members']=g;d['candidate_id']='timing-'+str(i)
        length=d['end']-d['start'];limit=max(1,len(re.findall(r'\w+',d['text'])))
        d['score']=1.5+.25*len({x['pass_id'] for x in g})+.3*len({x['source'] for x in g})
        d['score']-=.6*all(x['edge'] for x in g)+max(0,length-.65*limit)*1.5
        d['score']-=.8*any(x.get('partial_phrase') for x in g)
        result.append(d)
    return result

def transcript_units(track):
    """Use punctuation and the existing token sequence, not a fixed response lexicon."""
    tokens=track['tokens'];clauses=[x.strip() for x in re.split(r'[,，.。!?！？;；]+',track['transcript']) if x.strip()]
    cursor=0;units=[]
    for clause in clauses:
        key=content_key(clause);part=[];joined=''
        for token in tokens[cursor:]:
            part.append(token);joined+=content_key(token['text'])
            if joined==key:break
            if not key.startswith(joined):part=[];break
        if not part or joined!=key:return [dict(text=t['text'],tokens=[t]) for t in tokens]
        cursor+=len(part)
        # Repeated identical lexical units remain distinct even without punctuation.
        if len(part)>1 and len({content_key(t['text']) for t in part})==1:
            units.extend(dict(text=t['text'],tokens=[t]) for t in part)
        else:units.append(dict(text=clause,tokens=part))
    if cursor!=len(tokens):return [dict(text=t['text'],tokens=[t]) for t in tokens]
    return units

def timed_observations(tracks,decisions,duration):
    cs=[]
    for tr in tracks:
        for t in tr['tokens']:
            if not t['time_valid']:continue
            cs.append(dict(text=t['text'],start=t['start'],end=t['end'],pass_id='asr:'+tr['pass_id'],source='asr',
                evidence_id=t['id'],edge=t['start']<=tr['window_start']+.01 or t['end']>=tr['window_end']-.01))
        for unit in transcript_units(tr):
            if len(unit['tokens'])<=1:continue
            valid=[t for t in unit['tokens'] if t['time_valid']]
            if not valid:continue
            # A phrase's observed extent must not span a multi-second alignment jump.
            ordered=sorted(valid,key=lambda t:t['start'])
            if any(b['start']-a['end']>.5 for a,b in zip(ordered,ordered[1:])):continue
            start,end=ordered[0]['start'],max(t['end'] for t in valid)
            cs.append(dict(text=unit['text'],start=start,end=end,pass_id='asr:'+tr['pass_id'],source='asr_phrase',
                evidence_id=[t['id'] for t in unit['tokens']],edge=start<=tr['window_start']+.01 or end>=tr['window_end']-.01,
                partial_phrase=len(valid)<len(unit['tokens'])))
    seen=set()
    for d in decisions:
        for c in d.get('source_candidates',[]):
            if c.get('source')!='omni_audio':continue
            q=quoted_content(c['behavior'])
            if len(q)!=1:continue
            text=next(iter(q))
            for raw in c.get('sources',[]):
                start,end=raw['start'],raw['end'];wid=raw.get('window_id','full');identity=(wid,start,end,text)
                if identity in seen or not 0<=start<end<=duration:continue
                seen.add(identity);w=raw.get('window',{})
                cs.append(dict(text=text,start=start,end=end,pass_id='omni:'+wid,source='omni',evidence_id=c['id'],
                    edge=start<=w.get('start_s',0)+.01 or end>=w.get('end_s',duration)-.01))
    return cs

def select_path(units,candidates):
    """Monotonic maximum-support assignment using only observed positive intervals."""
    beam=[(0.,0.,[])]
    for unit in units:
        pool=[c for c in candidates if content_key(c['text'])==content_key(unit['text'])]
        next_beam=[]
        for score,last,path in beam:
            next_beam.append((score-100,last,path+[None]))
            for c in pool:
                if c['start']>=last-1e-7:
                    valid=[t for t in unit['tokens'] if t['time_valid']]
                    prior=0.
                    if valid:
                        a=min(t['start'] for t in valid);b=max(t['end'] for t in valid)
                        if b-a<=.8:prior=1.5*abs(c['start']-a)
                    next_beam.append((score+c['score']-prior,c['end'],path+[c]))
        # Preserve timing alternatives; maximizing count before quality prevents wide
        # records from swallowing later repeated words. No synthetic split points.
        best={}
        for item in next_beam:
            k=round(item[1],8)
            if k not in best or item[0]>best[k][0]:best[k]=item
        beam=sorted(best.values(),key=lambda x:-x[0])[:256]
    return max(beam,key=lambda x:x[0])[2] if units else []

def restore_source_wording(event, observations):
    """Select an existing model description verbatim; keep every source variant."""
    ids=set(event['source_observation_ids'])
    records=[]
    for o in observations:
        if o['event_id'] not in ids:continue
        sources=o.get('source_descriptions') or [dict(description=o['behavior'],source='input_observation')]
        for source in sources:
            records.append(dict(source,observation_id=o['event_id']))
    if records:
        event['description']=records[0]['description']
        event['original_descriptions']=list(dict.fromkeys(r['description'] for r in records))
        event['description_sources']=records
        event['description_source']=records[0]

def fuse(sample,decisions,tracks):
    base=integrate(sample,decisions,tracks,None);original=sample['events'];duration=sample['duration_s']
    observations=timed_observations(tracks,decisions,duration)
    asr=[c for c in observations if c['source'].startswith('asr')]
    outside_support=[];eligible=[]
    for c in observations:
        anchors=[x for x in asr if content_key(x['text'])==content_key(c['text'])]
        if c['source']=='omni' and anchors and not any(min(c['end'],x['end'])>max(c['start'],x['start']) for x in anchors):
            outside_support.append(c)
        else:eligible.append(c)
    candidates=group_candidates(eligible)
    reference=next((t for t in tracks if t['full_clip'] and t['tokens']),None)
    if reference is None:
        reference=max((t for t in tracks if t['tokens']),key=lambda t:(len(t['tokens']),t['window_end']-t['window_start']),default=None)
    units=transcript_units(reference) if reference else []
    audio_base=[e for e in base['events'] if e['modality']=='audio' and e['event_unit']!='word_candidate']
    decision_map={d['open_event_id']:d for d in decisions}
    chosen=select_path(units,candidates);events=[e for e in base['events'] if e['modality']=='visual'];failures=[]
    for i,(u,c) in enumerate(zip(units,chosen)):
        if c is None:
            failures.append(dict(unit_index=i,text=u['text'],reason='no_order_consistent_observed_positive_interval'));continue
        events.append(dict(event_id=f'auto-speech-{i:03d}',modality='audio',description=u['text'],text=u['text'],
            start=c['start'],end=c['end'],event_unit='spoken_response_candidate',categories=['listener_vocalization'],
            source_observation_ids=[],support_observation_ids=[],original_descriptions=[u['text']],description_source=dict(kind='ASR_original_text',pass_id=reference['pass_id'],token_ids=[t['id'] for t in u['tokens']]),source_boundary_methods={},
            status=dict(content='model_candidate',count='model_sequence',boundary='automatically_selected'),
            review_flags=[],
            boundary_method='monotonic_assignment_of_existing_observed_intervals',timing_evidence=c,
            training=dict(admitted=False,precise_boundary_loss=False,count_loss=False),duplicate_observations_removed=0))
    # Keep nonverbal/open model descriptions; lexical descriptions are evidence
    # for the selected word sequence, not additional repetitions.
    represented={content_key(u['text']) for u in units};alternatives=[]
    for e in audio_base:
        quotes=quoted_content(e['description'])
        if len(quotes)==1 and represented:
            alternatives.append(dict(event=deepcopy(e),reason='lexical_model_description_is_sequence_evidence_not_extra_word'))
            continue
        signatures={decision_map.get(i,{}).get('origin',{}).get('signature') for i in e['source_observation_ids']}
        speech=[x for x in events if x['modality']=='audio']
        overlap=[x for x in speech if min(x['end'],e['end'])>max(x['start'],e['start'])]
        if represented and signatures=={'vocalize'}:
            alternatives.append(dict(event=deepcopy(e),reason='generic_vocal_description_support' if overlap else 'generic_vocal_description_timing_conflict',related_event_ids=[x['event_id'] for x in overlap]))
            continue
        if any((min(x['end'],e['end'])-max(x['start'],e['start']))/(e['end']-e['start'])>=.5 for x in overlap):
            alternatives.append(dict(event=deepcopy(e),reason='overlapping_audio_descriptions_preserved_without_resolution',related_event_ids=[x['event_id'] for x in overlap]))
            continue
        events.append(e)
    for e in events:
        if 'timing_evidence' not in e:
            restore_source_wording(e, original)
    for alternative in alternatives:
        restore_source_wording(alternative['event'], original)
    events.sort(key=lambda e:(e['start'],e['end'],e['event_id']))
    links=[];unresolved=[]
    for o in original:
        if o['modality']=='visual':
            if not any(o['event_id'] in e['source_observation_ids'] for e in events):unresolved.append(dict(observation_id=o['event_id'],reason=o['mapping_status']))
            continue
        related=[e['event_id'] for e in events if e['modality']=='audio' and min(e['end'],o['end'])>max(e['start'],o['start'])]
        links.append(dict(observation_id=o['event_id'],relation='existing_audio_evidence_not_extra_occurrence',candidate_event_ids=related,count_resolved=False))
    groups=[]
    for e in events:
        if e['modality']!='audio':continue
        visual=[v['event_id'] for v in events if v['modality']=='visual' and min(e['end'],v['end'])>max(e['start'],v['start'])]
        groups.append(dict(group_id=f'group-{len(groups):03d}',member_event_ids=[e['event_id']]+visual,relation='possible_temporal_cooccurrence',continuous_activity_claim=False,training_mask=None))
    grouped={i for g in groups for i in g['member_event_ids']}
    for e in events:
        if e['event_id'] not in grouped:groups.append(dict(group_id=f'group-{len(groups):03d}',member_event_ids=[e['event_id']],relation='single_event_reference',continuous_activity_claim=False,training_mask=None))
    base.update(schema=VERSION,events=events,response_groups=groups,evidence_links=links,unresolved_observations=unresolved,
        token_event_links=[],correction=None,automatic_fusion=dict(reference_pass=reference['pass_id'] if reference else None,
            units=units,alignment_failures=failures,content_substitutions=[],phonetic_laughter_links=[],content_policy='verbatim_sources_no_semantic_replacement',
            candidate_intervals=candidates,outside_asr_timing_support=outside_support,unresolved_audio_alternatives=alternatives,human_feedback_read=False,media_read=False,gpu_used=False))
    assert all(0<=e['start']<e['end']<=duration for e in events)
    return base
