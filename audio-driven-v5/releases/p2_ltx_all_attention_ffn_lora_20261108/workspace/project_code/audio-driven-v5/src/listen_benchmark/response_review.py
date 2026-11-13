"""Conservative response review: preserve token order; never turn groups into labels."""
from copy import deepcopy
import re
import unicodedata

VERSION = 'listen-response-review-v2'


def literal(text):
    return ' '.join(re.findall(r'\w+', unicodedata.normalize('NFKC',text).casefold()))


def alignment_track(record, duration):
    window=record.get('input_evidence',{}).get('window',{})
    offset=window.get('start_s',0.)
    stop=window.get('end_s',duration)
    full=not bool(window)
    tokens=[]
    for i,w in enumerate(record.get('words',[])):
        start,end=offset+w['start_time'],offset+w['end_time']
        flags=[]
        if end<=start:flags.append('zero_or_negative_duration')
        if start<offset-1e-7 or end>stop+1e-7:flags.append('outside_source_window')
        if start<0 or end>duration+1e-7:flags.append('outside_parent_clip')
        tokens.append(dict(id=record['sample_id']+':word:'+str(i),pass_id=record['sample_id'],
            token_index=i,text=w['text'],start=start,end=end,local_start=w['start_time'],local_end=w['end_time'],
            time_valid=not flags,flags=flags))
    return dict(pass_id=record['sample_id'],full_clip=full,window_start=offset,window_end=stop,
                transcript=record.get('text',''),language=record.get('language'),status=record.get('status'),tokens=tokens)


def possible_same_token(a,b,tolerance=.2):
    return (a['pass_id']!=b['pass_id'] and a['time_valid'] and b['time_valid']
        and literal(a['text'])==literal(b['text'])
        and min(a['end'],b['end'])>max(a['start'],b['start'])
        and max(abs(a['start']-b['start']),abs(a['end']-b['end']))<=tolerance+1e-8)


def deduplicate_tracks(tracks,tolerance=.2):
    """Unique cross-pass complete-link matching; same-pass tokens never collapse.

    Matching is a candidate equivalence, not acoustic verification or a confidence
    probability. An ambiguous/broad token is left unlinked instead of bridging.
    """
    clusters=[]
    for track in sorted(tracks,key=lambda t:(not t['full_clip'],t['pass_id'])):
        tokens=track['tokens'];edges={}
        for i,t in enumerate(tokens):
            edges[i]=[j for j,c in enumerate(clusters)
                      if all(possible_same_token(t,x,tolerance) for x in c)]
        incoming={j:[i for i,js in edges.items() if j in js] for j in range(len(clusters))}
        for i,t in enumerate(tokens):
            candidates=edges[i]
            if len(candidates)==1 and len(incoming[candidates[0]])==1:
                clusters[candidates[0]].append(deepcopy(t))
            else:clusters.append([deepcopy(t)])
    return [dict(cluster_id='token-cluster-'+str(i),representative=deepcopy(c[0]),members=c,
                 equivalence='candidate_cross_window_duplicate' if len(c)>1 else 'unlinked_observation',
                 human_verified=False) for i,c in enumerate(clusters)]


def review_group(episode,observations,tracks,decisions):
    members=[deepcopy(e) for e in observations if e['event_id'] in episode['member_event_ids']]
    text=episode.get('text')
    full=next((t for t in tracks if t['full_clip']),None)
    refs=[]
    if full and text:
        refs=[deepcopy(t) for t in full['tokens'] if literal(t['text'])==literal(text)
              and t['start']<=episode['end']+.2 and t['end']>=episode['start']-.2]
    hypotheses=[]
    for t in refs:
        hypotheses.append(dict(reference_token_id=t['id'],start=t['start'] if t['time_valid'] else None,
            end=t['end'] if t['time_valid'] else None,reported_start=t['start'],reported_end=t['end'],
            description=t['text'],time_valid=t['time_valid'],flags=t['flags'],human_verified=False))
    ids=set(episode['member_event_ids'])
    source_details=[deepcopy(d) for d in decisions if d['open_event_id'] in ids]
    same_pass={}
    for d in source_details:
        for c in d.get('source_candidates',[]):
            if c.get('source')!='asr_aligner':continue
            for s in c.get('sources',[]):same_pass.setdefault(s.get('window_id','unknown'),set()).add(d['open_event_id'])
    flags=['group_envelope_is_not_continuous_activity','requires_audio_review']
    if any(len(v)>1 for v in same_pass.values()):flags.append('multiple_tokens_from_same_asr_pass')
    if any(not t['time_valid'] for t in refs):flags.append('unlocalized_reference_tokens')
    if not refs:flags.append('no_matching_full_clip_word_reference')
    return dict(group_id=episode['event_id'],candidate_span=dict(start=episode['start'],end=episode['end']),
        original_description=episode['behavior'],member_event_ids=episode['member_event_ids'],
        observations=members,source_details=source_details,reference_word_candidates=hypotheses,
        reference_word_count=len(refs) if refs else None,actual_occurrence_count=None,
        confirmed_occurrences=[],review_flags=flags,human_verified=False,
        training=dict(admitted=False,precise_boundary_loss=False,repetition_count_loss=False,
                      group_envelope_is_positive_mask=False))
