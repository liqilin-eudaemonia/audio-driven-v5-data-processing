"""Natural 2--10 s source intervals; model frame adaptation is a separate stage."""
import math
import re
from listen_policy_v6 import DANGLING, listener_screen, require_upright, reviewed_noise_hits, seconds

POLICY = dict(revision=7, minimum_crop_side=448, min_seconds=2., primary_min_seconds=3.,
    max_seconds=10., preferred_min_seconds=6., natural_pause_seconds=.65,
    onset_near_seconds=.5, frame_alignment_filter=False,
    orientation_policy='reject_non_upright_or_uncertain_raw_video',
    crop_scope='fixed_original_pixel_crop_per_final_child_no_padding_no_enlargement',
    onset_policy='separate_reserve_if_listener_activity_in_first_0.5_seconds',
    audio_policy='unchanged_QC3_and_whole_source_normalization')


def boundary_activity(start,end,person):
    vad=person['vad']
    crossing=any(a < start-1e-7 and b > start+1e-7 for a,b in vad)
    near=any(a < min(end,start+POLICY['onset_near_seconds']) and b > start for a,b in vad)
    feedback=[s for s in person['segments'] if s['feedback'] and s['end']>start
              and s['start']<min(end,start+POLICY['onset_near_seconds'])]
    return dict(onset_activity=near, onset_crossing=crossing,
        onset_transcribed_feedback=bool(feedback),
        onset_kind='carry_in_activity' if crossing else 'early_activity' if near else 'none',
        end_crossing=any(a<end-1e-7 and b>end+1e-7 for a,b in vad),
        end_activity=any(a<end and b>max(start,end-.5) for a,b in vad))


def natural_clips(start,end,words,speaker_vad=(),listener_activity=()):
    """Optimize full coverage at word gaps, then naturalness and balanced lengths.

    A skipped interval is recorded explicitly. No cut lies inside a timed word;
    normal adjacent children share a boundary, preserving intervening pauses.
    """
    if not math.isfinite(start+end) or end<=start:raise ValueError('Invalid interval')
    rejected=[]
    overlapping=sorted([w for w in words if w['end']>start and w['start']<end
                        and w['end']>w['start']],key=lambda w:(w['start'],w['end']))
    ws=[w for w in overlapping if w['start']>=start-1e-7 and w['end']<=end+1e-7]
    for w in overlapping:
        if w not in ws:
            rejected.append(dict(start=max(start,w['start']),end=min(end,w['end']),
                rejection='speaker_word_crosses_floor_or_exclusion_edge'))
    if not ws:return [],rejected+[dict(start=start,end=end,rejection='no_complete_speaker_word')]
    a,b=ws[0]['start'],max(w['end'] for w in ws)
    nodes={a:('speaker_start',0),b:('speaker_end',0)}
    running_end=a
    for left,right in zip(ws,ws[1:]):
        running_end=max(running_end,left['end'])
        if right['start']<running_end-1e-7:continue
        gap=max(0.,right['start']-running_end)
        terminal=bool(re.search(r'[.!?。！？][\s\"\u201d\u2019\)]*$',left.get('text','')))
        token=re.sub('[^a-z]','',left.get('text','').lower())
        reason='sentence_end' if terminal else 'natural_pause' if gap>=.65 and token not in DANGLING else 'word_gap'
        rank={'sentence_end':0.,'natural_pause':1.,'word_gap':3.}[reason]
        points=[(running_end+right['start'])/2]
        if gap>2:points.extend([running_end,right['start']])
        for p in points:
            if a<p<b:nodes[p]=(reason,rank)
    points=sorted(nodes);n=len(points)
    best=[None]*n;previous=[None]*n;best[0]=(0.,0.,0)
    for i,x in enumerate(points[:-1]):
        if best[i] is None:continue
        # This permits recovery around an exceptional >10s word/timestamp or short tail.
        candidate=(round(best[i][0]+points[i+1]-x,7),best[i][1],best[i][2])
        if best[i+1] is None or candidate<best[i+1]:
            best[i+1]=candidate;previous[i+1]=(i,False)
        for j in range(i+1,n):
            y=points[j];duration=y-x
            if duration>10+1e-7:break
            if duration<2-1e-7:continue
            if not any(w['start']>=x-1e-7 and w['end']<=y+1e-7 for w in ws):continue
            cuts_listener=any(v<y<z for v,z in listener_activity) if j<n-1 else False
            penalty=1+nodes[y][1]+max(0,6-duration)*.5+max(0,3-duration)*3+6*cuts_listener
            candidate=(best[i][0],best[i][1]+penalty,best[i][2]+1)
            if best[j] is None or candidate<best[j]:
                best[j]=candidate;previous[j]=(i,True)
    segments=[];j=n-1
    while j:
        i,keep=previous[j];segments.append((points[i],points[j],keep));j=i
    clips=[]
    for x,y,keep in reversed(segments):
        if not keep:
            reason='below_2s_or_unsegmentable_word_timing'
            if rejected and rejected[-1].get('rejection')==reason and abs(rejected[-1]['end']-x)<1e-7:
                rejected[-1]['end']=y
            else:rejected.append(dict(start=x,end=y,rejection=reason))
            continue
        contained=[w for w in ws if w['start']>=x-1e-7 and w['end']<=y+1e-7]
        clips.append(dict(start=round(x,9),end=round(y,9),duration_s=round(y-x,9),
            unique_seconds=round(y-x,9),speech_start=min(w['start'] for w in contained),
            speech_end=max(w['end'] for w in contained),end_reason=nodes[y][0],
            listener_event_cut=any(v<y<z for v,z in listener_activity),
            alignment_silence_seconds=0.,frame_adaptation_required=True,
            text=' '.join(w.get('text','') for w in contained)))
    return clips,rejected
