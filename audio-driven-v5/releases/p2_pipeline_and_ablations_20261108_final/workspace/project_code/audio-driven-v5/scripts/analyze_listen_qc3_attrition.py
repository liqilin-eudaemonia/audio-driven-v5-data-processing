"""Read-only attribution and counterfactual coverage. Never change screening rules."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
import math
import os
from pathlib import Path
import re
import tarfile
import time

os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
import preview_listen_v6 as preview
from p3_listen.common import read,write,sha
from screen_listen_qc_sample import metadata
from listen_policy_v6 import natural_clips,merged,DANGLING,POLICY

ROOT=preview.ROOT
FULL=ROOT/'data_corpora/seamless_listen_qc3_full'
OUT=ROOT/'experiments/p3_listen_qc3_attrition'


def bounds(words):
    units=[];first=0
    for i,w in enumerate(words):
        nxt=words[i+1] if i+1<len(words) else None
        terminal=bool(re.search(r'[.!?。！？][\s\"\u201d\u2019\)]*$',w.get('text','')))
        token=re.sub(r'[^a-z]','',w.get('text','').lower())
        pause=nxt is not None and nxt['start']-w['end']>=.65 and token not in DANGLING
        if terminal or pause or nxt is None:
            units.append(dict(start=words[first]['start'],end=w['end'],first=first,last=i,
                              reason='sentence_end' if terminal else 'natural_pause' if pause else 'transcript_end'))
            first=i+1
    return units


def trace(start,end,words,speaker_vad,listener_vad,align=True,avoid_listener=True,local=False):
    words=sorted([w for w in words if w['end']>w['start'] and
                  (not local or w['start']>=start-1e-7 and w['end']<=end+1e-7)],key=lambda w:w['start'])
    units=[];bad=[]
    for u in bounds(words):
        if u['end']>start and u['start']<end:
            if u['start']>=start-1e-7 and u['end']<=end+1e-7:units.append(u)
            else:bad.append(dict(u,cause='crosses_parent_boundary'))
    cursor=0;out=[]
    while cursor<len(units):
        first=units[cursor];choices=[];attempts=Counter()
        for j in range(cursor,len(units)):
            u=units[j]
            if j>cursor and u['first']!=units[j-1]['last']+1:break
            a,b=first['start'],u['end']
            if b-a>9.96+1e-7:break
            if b-a<4:continue
            attempts['length_ok']+=1
            extra=max(0.,(8*math.ceil(((b-a)*25-1-1e-7)/8)+1)/25-(b-a)) if align else 0.
            left=max(start,words[first['first']-1]['end'] if first['first'] else start)
            right=min(end,words[u['last']+1]['start'] if u['last']+1<len(words) else end)
            for x,y in speaker_vad:
                if y<=a:left=max(left,y)
                elif x<a<y:left=a
                if x>=b:right=min(right,x)
                elif x<b<y:right=b
            left=max(left,out[-1]['end'] if out else start)
            tail=min(extra,max(0.,right-b));head=extra-tail
            if head>a-left+1e-7:continue
            attempts['alignment_ok']+=1
            x,y=a-head,b+tail
            if avoid_listener and any(v<x<z or v<y<z for v,z in listener_vad):continue
            choices.append((j,dict(start=round(x,9),end=round(y,9),speech_start=a,speech_end=b,
                                   unique_seconds=b-a,duration_s=b-a+extra,reason=u['reason'])))
        if choices:
            j,clip=max(choices,key=lambda p:(p[1]['reason']=='sentence_end',p[1]['duration_s']))
            out.append(clip);cursor=j+1
        else:
            cause=('indivisible_unit_over_9_96s' if first['end']-first['start']>9.96+1e-7 else
                   'too_short_to_reach_4s' if not attempts['length_ok'] else
                   'no_silence_for_frame_alignment' if not attempts['alignment_ok'] else
                   'listener_vad_crosses_boundary')
            bad.append(dict(first,cause=cause));cursor+=1
    return out,bad


def word_coverage(start,end,words,minimum):
    """Upper bound: <=9.96s whole-word groups, no sentence/encoding criterion."""
    ws=[w for w in words if w['end']>w['start'] and w['start']>=start-1e-7 and w['end']<=end+1e-7]
    out=[];i=0
    while i<len(ws):
        options=[]
        for j in range(i,len(ws)):
            if ws[j]['end']-ws[i]['start']>9.96+1e-7:break
            if j+1==len(ws) or ws[j]['end']<=ws[j+1]['start']+1e-7:options.append(j)
        if not options:i+=1;continue
        j=options[-1];a,b=ws[i]['start'],ws[j]['end']
        if b-a>=minimum:out.append([a,b])
        i=j+1
    return sum(b-a for a,b in merged(out))


def init():
    if (OUT/'snapshot.json').exists():return
    cat=read(FULL/'catalog.json');items=[]
    for item in cat['rows']:
        p=FULL/'results'/(item['pair']+'.json')
        if p.exists() and read(p).get('complete'):items.append(item)
    write(OUT/'snapshot.json',dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        rows=items,scope='Completed pair records at snapshot time; not a random sample or final full corpus.',
        screening_contract_sha256=sha(FULL/'contract.json'),analysis_code_sha256=sha(Path(__file__))))
    print('SNAPSHOT',len(items),flush=True)


def analyze(item):
    path=OUT/'pairs'/(item['pair']+'.json')
    if path.exists():return read(path)['ok']
    try:
        audit=read(FULL/'audit'/(item['pair']+'.json'))
        result=read(FULL/'results'/(item['pair']+'.json'))
        oldpath=preview.OLD/'plans'/item['split']/(item['pair']+'.json')
        old=read(oldpath) if oldpath.exists() else {}
        sources=old.get('sources')
        if not audit['ok']:
            write(path,dict(pair=item['pair'],ok=True,source_ok=False,
                  pair_seconds=min(s['duration'] for s in sources) if sources else None,
                  rejected=audit['rejected'],parents=[]))
            return True
        records=[];geometry={}
        if sources is None:
            loaded=[preview.base.load_person(item[k]) for k in ('a','b')]
            sources=[v[2] for v in loaded];geometry={i:v[1] for i,v in enumerate(loaded)}
            records=[v[0] for v in loaded]
        else:
            for source in sources:
                m=metadata(source['source']['archive'],source['source']['member'])
                records.append(dict(id=source['id'],transcript=m.get('metadata:transcript',[]),vad=m.get('metadata:vad',[])))
        people=[preview.base.rules.prepare(r,i) for i,r in enumerate(records)]
        ann=preview.base.rules.annotate(dict(pair=item['pair'],subset=item['label'],sources=sources),
                                       dict(records=records),defer_untranscribed=True)
        crops={r['parent']:r['crop'] for r in audit['rejected'] if 'crop' in r}
        for r in audit['clips']:
            parent=int(re.search(r'_p(\d+)_\d+$',r['sample_id']).group(1))
            crops[parent]=r['listener_crop']
        parentrows=[];losses=Counter();stages=Counter()
        stages['source_seconds']=ann['duration']
        stages['tracked_floor_seconds']=sum(t['end']-t['start'] for t in ann['turns'])
        stages['listener_excluded_seconds']=sum(b-a for t in ann['turns'] for a,b in t['excluded_intervals'])
        stages['short_or_few_words_seconds']=sum(c['duration'] for t in ann['turns'] for c in t['listening_intervals']
                                                if c['export_status']!='candidate')
        stages['annotated_candidate_seconds']=sum(c['duration'] for c in ann['clips'])
        actual_by_parent={}
        for r in audit['clips']:
            parent=int(re.search(r'_p(\d+)_\d+$',r['sample_id']).group(1))
            actual_by_parent.setdefault(parent,[]).append(r)
        for ci,c in enumerate(ann['clips']):
            a,b=c['start'],c['end'];si=c['speaker_index'];li=c['listener_index'];p=people[si]
            ws=[w for w in p['words'] if w['start']<b-.001 and w['end']>a+.001]
            if not ws:continue
            ca=max(a,min(w['start'] for w in ws));cb=min(b,max(w['end'] for w in ws))
            pr=dict(parent=ci,start=ca,end=cb,seconds=cb-ca)
            crop=crops.get(ci)
            if crop is None:
                if li not in geometry:
                    source=sources[li]
                    with tarfile.open(source['source']['archive'],'r:') as tar:
                        with np.load(tar.extractfile(source['id']+'.npz'),allow_pickle=False) as z:
                            k=z['boxes_and_keypoints:keypoints']
                    geometry[li]=dict(source,arrays={'boxes_and_keypoints:keypoints':k})
                try:crop=preview.base.rules.full_turn_crop(geometry[li],ca,cb)
                except ValueError as e:
                    pr['crop_error']=str(e);parentrows.append(pr);continue
            pr['crop']=crop;parentrows.append(pr)
            if crop['size']<512 or crop['wide_crop']:continue
            strict,bad=trace(a,b,p['words'],p['vad'],people[li]['vad'])
            expected,_=natural_clips(a,b,p['words'],p['vad'],people[li]['vad'])
            assert [(r['start'],r['end']) for r in strict]==[(r['start'],r['end']) for r in expected],item['pair']
            pr['boundary_kept_seconds']=sum(r['unique_seconds'] for r in strict)
            pr['actual_after_manual_listener_seconds']=sum(r['unique_seconds'] for r in actual_by_parent.get(ci,[]))
            pr['boundary_rejections']=[dict(cause=r['cause'],seconds=max(0,min(cb,r['end'])-max(ca,r['start'])),
                                          start=max(ca,r['start']),end=min(cb,r['end'])) for r in bad]
            scenarios={}
            for label,align,avoid,local in [
                    ('no_frame_alignment',False,True,False),
                    ('allow_listener_edge_overlap',True,False,False),
                    ('native_semantic_units',False,False,False),
                    ('turn_local_semantic_units',False,False,True)]:
                clips,_=trace(a,b,p['words'],p['vad'],people[li]['vad'],align,avoid,local)
                scenarios[label]=sum(r['unique_seconds'] for r in clips)
            scenarios['whole_word_upper_bound_min4']=word_coverage(a,b,p['words'],4)
            scenarios['whole_word_upper_bound_no_minimum']=word_coverage(a,b,p['words'],0)
            pr['counterfactual_seconds']=scenarios
        # Audio outcomes are copied from the fixed screened source records.
        audio=Counter();audio_counts=Counter()
        for r in result['clips']:
            key=r['split']+'/'+r['effective_decision']
            audio[key]+=r['unique_seconds'];audio_counts[key]+=1
        write(path,dict(pair=item['pair'],ok=True,source_ok=True,parents=parentrows,
            stages=dict(stages),audio_seconds=dict(audio),audio_counts=dict(audio_counts)))
        return True
    except Exception as e:
        write(path,dict(pair=item['pair'],ok=False,error=type(e).__name__+': '+str(e)))
        return False


def run(node,workers):
    snap=read(OUT/'snapshot.json')
    assert snap['analysis_code_sha256']==sha(Path(__file__))
    rows=snap['rows'][node::2];ok=0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i,value in enumerate(pool.map(analyze,rows,chunksize=4),1):
            ok+=int(value)
            if i%100==0 or i==len(rows):
                report=dict(node=node,completed=i,total=len(rows),ok=ok,errors=i-ok)
                write(OUT/f'progress_{node}.json',report);print(report,flush=True)


def summary():
    snap=read(OUT/'snapshot.json');stages=Counter();audio=Counter();audio_counts=Counter()
    bins=[0,256,320,384,448,512,640,768,1024,float('inf')]
    hist=[dict(minimum=bins[i],maximum=bins[i+1] if math.isfinite(bins[i+1]) else None,
               count=0,seconds=0,nonwide_count=0,nonwide_seconds=0) for i in range(len(bins)-1)]
    thresholds={str(v):dict(count=0,seconds=0) for v in [256,320,384,448,480,512,640]}
    counts=Counter();loss_count=Counter();loss_seconds=Counter();alternatives=Counter()
    source_reject=Counter();source_reject_seconds=Counter();source_unknown=Counter();sizes=[];errors=[]
    for item in snap['rows']:
        p=OUT/'pairs'/(item['pair']+'.json')
        if not p.exists():continue
        r=read(p);counts['processed_pairs']+=1
        if not r['ok']:errors.append(r);continue
        if not r['source_ok']:
            counts['source_rejected_pairs']+=1
            reason=str(r['rejected'][0]['reason']);source_reject[reason]+=1
            if r['pair_seconds'] is None:source_unknown[reason]+=1
            else:source_reject_seconds[reason]+=r['pair_seconds']
            continue
        counts['source_valid_pairs']+=1;stages.update(r['stages']);audio.update(r['audio_seconds']);audio_counts.update(r['audio_counts'])
        for pr in r['parents']:
            counts['parent_candidates']+=1;stages['parent_speaker_span_seconds']+=pr['seconds']
            if 'crop' not in pr:
                counts['crop_geometry_error']+=1;stages['crop_geometry_error_seconds']+=pr['seconds'];continue
            c=pr['crop'];size=c['size'];sizes.append(size)
            h=next(h for h in hist if size>=h['minimum'] and (h['maximum'] is None or size<h['maximum']))
            h['count']+=1;h['seconds']+=pr['seconds']
            if not c['wide_crop']:
                h['nonwide_count']+=1;h['nonwide_seconds']+=pr['seconds']
                for t,s in thresholds.items():
                    if size>=int(t):s['count']+=1;s['seconds']+=pr['seconds']
            reason='crop_below_512' if size<512 else 'wide_crop' if c['wide_crop'] else 'crop_pass'
            counts[reason]+=1;stages[reason+'_seconds']+=pr['seconds']
            if reason!='crop_pass':continue
            stages['boundary_kept_seconds']+=pr['boundary_kept_seconds']
            stages['after_manual_listener_seconds']+=pr['actual_after_manual_listener_seconds']
            for b in pr['boundary_rejections']:
                loss_count[b['cause']]+=1;loss_seconds[b['cause']]+=b['seconds']
            alternatives.update(pr['counterfactual_seconds'])
    data=dict(snapshot_utc=snap['utc'],snapshot_pairs=len(snap['rows']),counts=dict(counts),
        errors=errors,stages_seconds=dict(stages),audio_seconds=dict(audio),audio_counts=dict(audio_counts),
        crop_histogram=hist,crop_threshold_nonwide_parent_opportunities=thresholds,
        crop_size_percentiles={str(p):float(np.percentile(sizes,p)) for p in [5,10,25,50,75,90,95]} if sizes else {},
        boundary_loss_unit_counts=dict(loss_count),boundary_loss_unit_seconds=dict(loss_seconds),
        counterfactual_seconds=dict(alternatives),source_rejected_pair_counts=dict(source_reject),
        source_rejected_pair_seconds_known=dict(source_reject_seconds),source_rejected_missing_durations=dict(source_unknown),
        caveats=['Parent opportunities precede temporal/audio filtering, not final usable hours.',
                 'Counterfactual coverage is diagnostic only; encoding/sentence quality not validated.',
                 'Boundary losses are clamped semantic-unit spans; gaps account for residual waterfall loss.',
                 'Source duration counted once per paired conversation; no summing two participant tracks.'])
    write(OUT/'summary.json',data)
    print(json.dumps(data,ensure_ascii=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['init','run','summary'])
    p.add_argument('--node',type=int,choices=[0,1],default=0);p.add_argument('--workers',type=int,default=16)
    a=p.parse_args()
    if a.stage=='init':init()
    elif a.stage=='run':run(a.node,a.workers)
    else:summary()
