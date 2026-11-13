"""Experimental candidate integration; never convert model consensus into ground truth."""
import math
import re
from statistics import median


def legal(start, end, duration):
    return (type(start) in (int, float) and type(end) in (int, float)
            and math.isfinite(start) and math.isfinite(end) and 0 <= start < end <= duration)


def cluster_candidates(candidates, tolerance=.2):
    """Conservatively group exact-text, overlapping predictions from different windows."""
    clusters=[]
    for candidate in sorted(candidates,key=lambda x:(x['start'],x['end'])):
        key=(candidate['source'],re.sub(r'\W','',candidate['behavior']).casefold())
        found=next((c for c in clusters if c['key']==key and all(
            x['window_id']!=candidate['window_id']
            and abs(x['start']-candidate['start'])<=tolerance+1e-8
            and abs(x['end']-candidate['end'])<=tolerance+1e-8
            and min(x['end'],candidate['end'])>max(x['start'],candidate['start'])
            for x in c['sources'])),None)
        if found is None:clusters.append({'key':key,'sources':[candidate]})
        else:found['sources'].append(candidate)
    results=[]
    for cluster in clusters:
        sources=cluster['sources'];first=sources[0]
        results.append(dict(start=median(x['start'] for x in sources),end=median(x['end'] for x in sources),
                            modality=first['modality'],behavior=first['behavior'],source=first['source'],
                            support=len(sources),sources=sources,valid=True,human_verified=False))
    return sorted(results,key=lambda x:(x['start'],x['end'],x['source']))


def integrate(visual_windows,audio_windows,word_windows,duration):
    candidates=[];invalid_words=[]
    for source,modality,windows in [('qwen3_vl','visual',visual_windows),('omni_audio','audio',audio_windows)]:
        for window in windows:
            for event in window['events']:
                if event.get('valid') and event.get('modality')==modality and legal(event['start'],event['end'],duration):
                    candidates.append(dict(start=event['start'],end=event['end'],modality=modality,
                                           behavior=event['behavior'],source=source,window_id=window['id'],
                                           window=window['window']))
    for window in word_windows:
        bounds=window['window'];offset=bounds['start_s'];size=bounds['end_s']-offset
        for word in window['words']:
            start,end=word['start_time'],word['end_time']
            if not legal(start,end,size):
                invalid_words.append(dict(word,window=bounds));continue
            candidates.append(dict(start=start+offset,end=end+offset,modality='audio',behavior=word['text'],
                                   source='asr_aligner',window_id=window['sample_id'],window=bounds))
    events=cluster_candidates(candidates)
    for index,event in enumerate(events):event['id']=f'candidate-{index:04d}'
    # Temporal overlap is an association only, never a merged event or a semantic claim.
    links=[{'visual_id':v['id'],'audio_id':a['id'],'overlap_start':max(v['start'],a['start']),
            'overlap_end':min(v['end'],a['end']),'relation':'temporal_overlap_only'}
           for v in events if v['modality']=='visual'
           for a in events if a['modality']=='audio' and min(v['end'],a['end'])>max(v['start'],a['start'])]
    return dict(events=events,links=links,invalid_words=invalid_words,human_ground_truth=False,
                status='candidates_require_review',deduplication='exact_text_and_pairwise_boundaries_within_0.2s_only')
