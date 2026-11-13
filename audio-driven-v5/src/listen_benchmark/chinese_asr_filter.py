"""User policy: keep single Han characters, mute multi-Han ASR events.

This is a text policy, not a claim that the source waveform is proven noise.
Original records are retained in exclusion/audit entries.
"""
import copy
import math
import unicodedata
import numpy as np
from listen_benchmark.official_audio_gate import merge

POLICY = dict(version='multi-han-asr-mute-v1',min_han_characters=2,
              scope='audio_ASR_original_text',single_han='keep',
              mute_scope='current_event_intervals_only',fade_seconds=.01,
              fade_placement='outside_exact_muted_ranges',training_applied=False)


def han_count(text):
    return sum(unicodedata.name(c,'').startswith(('CJK UNIFIED IDEOGRAPH','CJK COMPATIBILITY IDEOGRAPH'))
               for c in unicodedata.normalize('NFKC',text or ''))


def should_remove(event):
    return (event.get('modality')=='audio' and
            event.get('description_source',{}).get('kind')=='ASR_original_text' and
            han_count(event.get('text',''))>=POLICY['min_han_characters'])


def mask_ranges(mask,sr):
    changes=np.diff(np.pad((mask>0).astype(np.int8),(1,1)))
    return [dict(start=int(a)/sr,end=int(b)/sr) for a,b in zip(np.flatnonzero(changes==1),np.flatnonzero(changes==-1))]


def filter_labels(label,gate):
    result=copy.deepcopy(label);sr=gate['sample_rate'];n=gate['samples']
    removed=[e for e in label['events'] if should_remove(e)]
    cut=[]
    for e in removed:
        for x in e['intervals']:
            a,b=x['start'],x['end']
            if not math.isfinite(a+b) or b<=a:raise ValueError('Invalid current event interval')
            cut.append([max(0,math.floor(a*sr+1e-7)),min(n,math.ceil(b*sr-1e-7))])
    cut=merge(cut);mask=np.ones(n,dtype=np.float32);fade=round(POLICY['fade_seconds']*sr)
    for a,b in cut:
        mask[a:b]=0
        # Keep every requested mute sample zero; fade adjacent surviving audio.
        k=min(fade,a)
        if k>=2:mask[a-k:a]*=np.cos(np.linspace(0,np.pi/2,k,dtype=np.float64)).astype(np.float32)**2
        k=min(fade,n-b)
        if k>=2:mask[b:b+k]*=np.sin(np.linspace(0,np.pi/2,k,dtype=np.float64)).astype(np.float32)**2
    support=np.zeros(n,dtype=np.float32)
    for a,b in gate['positive_intervals_samples']:support[a:b]=1
    support*=mask;valid=mask_ranges(support,sr)
    removed_ids={e['event_id'] for e in removed};output=[];newly_excluded=[]
    audits={a['event_id']:a for a in result['audio_event_alignment']}
    for e in result['events']:
        if e.get('modality')!='audio':output.append(e);continue
        eid=e['event_id'];old=copy.deepcopy(e);a=audits[eid]
        if eid in removed_ids:
            a.update(previous_intervals=copy.deepcopy(e['intervals']),previous_method=a['method'],method='excluded',
                     reason='multi_character_chinese_asr',intervals=[],boundary_changed=True,
                     language_filter=dict(han_characters=han_count(e.get('text')),policy=POLICY['version']))
            newly_excluded.append(dict(original_event=old,alignment=copy.deepcopy(a)))
            continue
        kept=[]
        for x in e['intervals']:
            kept.extend(dict(start=max(x['start'],v['start']),end=min(x['end'],v['end'])) for v in valid
                        if min(x['end'],v['end'])>max(x['start'],v['start'])+1e-9)
        if kept!=e['intervals']:
            a.update(previous_intervals=copy.deepcopy(e['intervals']),previous_method=a['method'],
                     method='language_mask_intersection' if kept else 'excluded',
                     reason='overlap_with_chinese_asr_mute',intervals=kept,boundary_changed=True)
            if not kept:
                newly_excluded.append(dict(original_event=old,alignment=copy.deepcopy(a)));continue
            e.update(start=kept[0]['start'],end=kept[-1]['end'],intervals=kept,
                     boundary_method='language_mask_intersection',boundary_alignment=copy.deepcopy(a),disjoint_support=len(kept)>1)
        output.append(e)
    result['events']=output;result['excluded_audio_events'].extend(newly_excluded)
    result['audio_events_retained']=sum(e.get('modality')=='audio' for e in output)
    result['language_filter']=dict(policy=POLICY,removed_event_ids=sorted(removed_ids),
                                   removed_texts=[e.get('text') for e in removed],
                                   mute_intervals_samples=cut,positive_intervals=valid,
                                   adjacent_fade_seconds=POLICY['fade_seconds'])
    return result,mask,support
