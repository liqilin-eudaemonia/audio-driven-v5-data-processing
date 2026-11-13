#!/usr/bin/env python3
"""Rule-only floor tracking. No inference model and no fixed-length turn cuts."""
import argparse
import collections
import concurrent.futures
import json
import math
import re
from pathlib import Path

import numpy as np
import soundfile as sf
import build_seamless_listening_preview as media

V1=media.ROOT/'data_previews/seamless_face_listening_v1'
OUT=media.ROOT/'data_previews/seamless_rule_turns_v2'
RULES={
    'version':'rule-floor-v2.0','lookahead_seconds':8.0,
    'takeover_content_words':8,'takeover_content_seconds':1.6,
    'dominance_ratio':1.5,'incumbent_sparse_words':3,
    'incumbent_sparse_seconds':1.0,'switch_cooldown_seconds':2.5,
    'yielded_takeover_words':4,'yielded_takeover_content_seconds':0.65,
    'short_feedback_max_seconds':3.0,'short_feedback_max_words':7,
    'excluded_event_margin_seconds':0.15,'minimum_export_seconds':4.0,
    'minimum_speaker_words':4,'turn_end_tail_seconds':0.2,
}
FEEDBACK_PHRASES={
    'i see','i know','i agree','i understand','i get it','got it','of course',
    'that is true','thats true','that is right','thats right','that makes sense',
    'makes sense','thats cool','that is cool','i love that','thats good',
    'oh my god','oh my gosh','oh really','for sure','all right','alright',
    'mm hmm','uh huh','mhm','mmhmm','uhhuh','hmm','hm',
}
FEEDBACK_WORDS={'yeah','yep','yes','yup','right','okay','ok','oh','ah','uh','huh',
                'mm','hmm','mhm','mmhmm','uhhuh','wow','exactly','absolutely','sure','totally'}


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');media.dump(tmp,value);tmp.replace(path)


def tokens(text):
    return re.findall(r"[a-z]+",text.lower().replace("'",''))


def feedback(text,duration):
    ts=tokens(text)
    return bool(ts) and duration<=RULES['short_feedback_max_seconds'] and len(ts)<=7 and (
        ' '.join(ts) in FEEDBACK_PHRASES or all(t in FEEDBACK_WORDS for t in ts))


def prepare(record,person):
    segments=[];words=[]
    for i,s in enumerate(record['transcript']):
        if not isinstance(s.get('start'),(int,float)) or not isinstance(s.get('end'),(int,float)):continue
        a,b=max(0,float(s['start'])),float(s['end'])
        if b<=a:continue
        text=s.get('transcript','');ws=[]
        for w in s.get('words',[]):
            if isinstance(w.get('start'),(int,float)) and isinstance(w.get('end'),(int,float)) and w['end']>=w['start']:
                ws.append({'start':max(0,float(w['start'])),'end':float(w['end']),'text':w.get('word','')})
        is_feedback=feedback(text,b-a)
        for w in ws:w.update(person=person,segment=f'{person}_{i:03d}',feedback=is_feedback)
        segments.append(dict(id=f'{person}_{i:03d}',person=person,start=a,end=b,text=text,words=ws,
                             feedback=is_feedback,word_count=len(ws) or len(tokens(text))))
        words.extend(ws)
    return dict(segments=segments,words=sorted(words,key=lambda w:w['start']),
                vad=media.merge([[v['start'],v['end']] for v in record['vad'] if v['end']>v['start']]))


def evidence(person,a,b,content_only=True):
    ws=[w for w in person['words'] if a<=w['start']<b and (not content_only or not w['feedback'])]
    intervals=[[max(a,w['start']),min(b,w['end'])] for w in ws]
    return dict(words=len(ws),content_seconds=round(media.overlap(intervals,a,b),3),
                vad_seconds=round(media.overlap(person['vad'],a,b),3))


def question_insertion(s,owner,people):
    if '?' not in s['text'] or s['end']-s['start']>4.5 or s['word_count']>10:return False
    after=evidence(people[owner],s['end'],s['end']+6)
    return after['words']>=5


def track_floors(people,duration):
    events=sorted([s for p in people for s in p['segments']],key=lambda s:(s['start'],s['person']))
    starts=[s for s in events if not s['feedback'] and s['word_count']>=4]
    if not starts:return [],[]
    first=starts[0];owner=first['person'];floor_start=first['start']
    turns=[dict(speaker_index=owner,start=round(floor_start,3),start_anchor=first['id'],
                start_reason='first_substantive_utterance',takeover_evidence=None)]
    decisions=[]
    for s in events:
        if s['start']<floor_start or s['person']==owner:continue
        if s['feedback']:continue
        if question_insertion(s,owner,people):
            decisions.append(dict(segment=s['id'],decision='clarification_candidate',reason='short_question_then_incumbent_continues'))
            continue
        a=s['start'];b=min(duration,a+RULES['lookahead_seconds'])
        challenger=evidence(people[s['person']],a,b);incumbent=evidence(people[owner],a,b)
        sustained=challenger['words']>=8 and challenger['content_seconds']>=1.6
        sparse=incumbent['words']<=3 and incumbent['content_seconds']<=1.0
        yielded=challenger['words']>=4 and challenger['content_seconds']>=.65 and incumbent['words']<=1 and incumbent['content_seconds']<=.3
        dominance=challenger['content_seconds']>=1.5*max(.25,incumbent['content_seconds']) and challenger['words']>=incumbent['words']+4
        cooldown=a-floor_start>=RULES['switch_cooldown_seconds']
        if ((sustained and (sparse or dominance)) or yielded) and cooldown:
            turns[-1]['end']=round(a,3);turns[-1]['end_reason']='other_person_takeover_onset'
            owner=s['person'];floor_start=a
            ev=dict(challenger=challenger,incumbent=incumbent,horizon_end=round(b,3),
                    matched_rule='incumbent_yields' if sparse or yielded else 'challenger_dominates')
            turns.append(dict(speaker_index=owner,start=round(a,3),start_anchor=s['id'],
                              start_reason='rule_takeover',takeover_evidence=ev))
            decisions.append(dict(segment=s['id'],decision='takeover',reason=ev['matched_rule']))
        else:
            decisions.append(dict(segment=s['id'],decision='no_takeover',
                                  reason='insufficient_sustained_dominance',challenger=challenger,incumbent=incumbent))
    last=max([s['end'] for s in events]+[v[1] for p in people for v in p['vad']])
    turns[-1]['end']=round(min(duration,last+.2),3);turns[-1]['end_reason']='conversation_end'
    return turns,decisions


def subtract(a,b,excluded):
    parts=[];start=a
    for x,y in media.merge(excluded):
        x=max(a,x);y=min(b,y)
        if y<=start:continue
        if x>start:parts.append([start,x])
        start=max(start,y)
    if start<b:parts.append([start,b])
    return parts


def annotate(pair,item,defer_untranscribed=False):
    people=[prepare(r,i) for i,r in enumerate(item['records'])]
    duration=min(s['duration'] for s in pair['sources'])
    turns,decisions=track_floors(people,duration)
    clips=[]
    for ti,turn in enumerate(turns):
        turn['turn_id']=f'T{ti+1:03d}';si=turn['speaker_index'];li=1-si;a,b=turn['start'],turn['end']
        turn['listener_index']=li;turn['events']=[]
        for s in people[li]['segments']:
            if s['end']<=a or s['start']>=b:continue
            x,y=max(a,s['start']),min(b,s['end'])
            if y-x<=.001:continue
            if s['start']<a:
                label='overlap_from_previous_turn';why='previous_speaker_continues_after_takeover'
            elif s['feedback']:
                label='feedback_candidate';why='conservative_feedback_phrase_match'
            elif question_insertion(s,si,people):
                label='clarification_candidate';why='short_question_then_floor_holder_continues'
            else:
                overlap=media.overlap(people[si]['vad'],x,y)
                continuation=evidence(people[si],y,min(duration,y+6))['words']
                label='interruption_candidate' if overlap>.15 or continuation>=4 else 'uncertain_speech'
                why='non_feedback_speech_without_confirmed_takeover'
            turn['events'].append(dict(start=round(x,3),end=round(y,3),text=s['text'],
                                       label=label,reason=why,source_segment=s['id'],
                                       exclude=label!='feedback_candidate'))
        # VAD-only sounds are visible as uncertain vocalizations, not forced to speech/backchannel.
        text_intervals=[[s['start'],s['end']] for s in people[li]['segments']]
        for x,y in people[li]['vad']:
            x,y=max(a,x),min(b,y)
            if y<=x or media.overlap(text_intervals,x,y)>.2*(y-x):continue
            turn['events'].append(dict(start=round(x,3),end=round(y,3),text='[VAD only]',
                                       label='untranscribed_vocalization',reason='audio_activity_without_aligned_transcript',
                                       exclude=y-x>3 and not defer_untranscribed))
        turn['events'].sort(key=lambda e:e['start'])
        excluded=[[max(a,e['start']-.15),min(b,e['end']+.15)] for e in turn['events'] if e['exclude']]
        turn['excluded_intervals']=media.merge(excluded)
        clean=subtract(a,b,excluded);turn['listening_intervals']=[]
        for pi,(x,y) in enumerate(clean):
            ws=[w for w in people[si]['words'] if x<=w['start']<y]
            item_interval=dict(start=round(x,3),end=round(y,3),duration=round(y-x,3))
            if y-x<4 or len(ws)<4:
                item_interval['export_status']='too_short_or_insufficient_speaker_words';turn['listening_intervals'].append(item_interval);continue
            item_interval['export_status']='candidate';turn['listening_intervals'].append(item_interval)
            ev=[e for e in turn['events'] if e['end']>x and e['start']<y and not e['exclude']]
            lw=[w for w in people[li]['words'] if w['end']>x and w['start']<y]
            whole=abs(x-a)<.001 and abs(y-b)<.001
            clip=dict(start=round(x,3),end=round(y,3),duration=round(y-x,3),speaker_index=si,listener_index=li,
                      speaker=item['records'][si]['id'],listener=item['records'][li]['id'],
                      turn_id=turn['turn_id'],part_index=pi+1,parent_turn_start=a,parent_turn_end=b,
                      entire_floor_turn=whole,start_reason=turn['start_reason'] if x==a else 'after_excluded_listener_event',
                      end_reason=turn['end_reason'] if y==b else 'before_excluded_listener_event',
                      boundary_evidence=turn['takeover_evidence'],parent_events=turn['events'],
                      kind='feedback_candidate' if any(e['label']=='feedback_candidate' for e in ev) else 'vocal_candidate' if ev else 'quiet_candidate',
                      listener_events=ev,speaker_text=' '.join(w['text'] for w in ws),listener_text=' '.join(w['text'] for w in lw),
                      speaker_vad_fraction=round(media.overlap(people[si]['vad'],x,y)/(y-x),3),
                      listener_vad_fraction=round(media.overlap(people[li]['vad'],x,y)/(y-x),3),
                      speaker_vad=[[max(x,v)-x,min(y,z)-x] for v,z in people[si]['vad'] if z>x and v<y],
                      listener_vad=[[max(x,v)-x,min(y,z)-x] for v,z in people[li]['vad'] if z>x and v<y])
            clips.append(clip)
    return dict(pair=pair['pair'],subset=pair['subset'],split='test',sources=pair['sources'],
                duration=duration,rules=RULES,turns=turns,decisions=decisions,clips=clips)


def full_turn_crop(r,a,b):
    """Fixed window spanning a whole turn; large displacement is flagged, not recut."""
    k=r['arrays']['boxes_and_keypoints:keypoints'][int(a*r['fps']):math.ceil(b*r['fps']),23:91]
    valid=(k[:,:,2]>.4).sum(axis=1)>=45
    if not len(valid) or valid.mean()<.8:raise ValueError('Face landmark validity below 80%')
    k=k[valid];x=np.where(k[:,:,2]>.4,k[:,:,0],np.nan);y=np.where(k[:,:,2]>.4,k[:,:,1],np.nan)
    xl,xh=np.nanpercentile(x,[3,97],axis=1);yl,yh=np.nanpercentile(y,[3,97],axis=1)
    fw=float(np.median(xh-xl));fh=float(np.median(yh-yl))
    left=float(np.percentile(xl,1))-.4*fw;right=float(np.percentile(xh,99))+.4*fw
    top=float(np.percentile(yl,1))-.55*fh;bottom=float(np.percentile(yh,99))+.3*fh
    size=int(math.ceil(max(right-left,bottom-top,fw*2.2,fh*2.2)/2)*2)
    size=min(size,r['width']//2*2,r['height']//2*2)
    if size<60:raise ValueError('Implausible face size')
    return dict(x=int(np.clip((left+right-size)/2,0,r['width']-size))//2*2,
                y=int(np.clip((top+bottom-size)/2,0,r['height']-size))//2*2,size=size,
                valid_fraction=round(float(valid.mean()),4),face_width=round(fw,1),face_height=round(fh,1),
                wide_crop=size>3.2*max(fw,fh))


def render_clip(pair,c,people,n):
    s,l=people[c['speaker_index']],people[c['listener_index']];a,b=c['start'],c['end'];duration=b-a
    d=OUT/'media'/pair/f'clip_{n:03d}';d.mkdir(parents=True,exist_ok=True)
    if (d/'metadata.json').exists():return json.loads((d/'metadata.json').read_text())
    boxes=[full_turn_crop(r,a,b) for r in [s,l]]
    for name,r in [('speaker',s),('listener',l)]:
        sf.write(d/f'{name}.wav',r['audio'][round(a*r['sr']):round(b*r['sr'])],r['sr'],subtype='PCM_16')
        media.run(['ffmpeg','-v','error','-nostdin','-y','-i',str(d/f'{name}.wav'),'-c:a','aac','-b:a','128k',str(d/f'{name}.m4a')])
    media.render_face_media(s,l,boxes,a,b,d)
    video_inputs=[];filters=[]
    for i,(r,box) in enumerate(zip([s,l],boxes)):
        video_inputs+=['-threads','2','-ss',str(a),'-t',str(duration),'-i',media.uri(r['source'])]
        filters.append(f'[{i}:v]setpts=PTS-STARTPTS,drawbox=x={box["x"]}:y={box["y"]}:w={box["size"]}:h={box["size"]}:color=0x58e0b4:t=5,scale=320:360:force_original_aspect_ratio=decrease,pad=320:360:(ow-iw)/2:(oh-ih)/2,fps=15,setsar=1[c{i}]')
    filters.append('[c0][c1]hstack=inputs=2[v]')
    media.run(['ffmpeg','-v','error','-nostdin','-y',*video_inputs,'-filter_complex_threads','1','-filter_complex',';'.join(filters),
               '-map','[v]','-an','-t',str(duration),'-c:v','libx264','-threads','2','-preset','veryfast','-crf','25',
               '-pix_fmt','yuv420p','-movflags','+faststart',str(d/'context.mp4')])
    c.update(path=str(d.relative_to(OUT)),face_resolution=512,media_revision=3,segmentation_revision=2,
             crop_boxes=dict(zip(['speaker','listener'],boxes)),
             emotion={role:{key.split(':')[-1]:media.curve(r,a,b,key) for key in ['movement:emotion_valence','movement:emotion_arousal']}
                      for role,r in [('speaker',s),('listener',l)]})
    save(d/'metadata.json',c)
    for name in ['speaker','listener']:(d/f'{name}.wav').unlink()
    return c


def render_conversation(pair,people):
    d=OUT/'media'/pair;d.mkdir(parents=True,exist_ok=True)
    path=d/'conversation.mp4'
    if path.exists():return str(path.relative_to(OUT))
    duration=min(r['duration'] for r in people);inputs=[];filters=[]
    for i,r in enumerate(people):
        inputs+=['-threads','2','-i',media.uri(r['source'])]
        filters.append(f'[{i}:v]setpts=PTS-STARTPTS,scale=320:360:force_original_aspect_ratio=decrease,pad=320:360:(ow-iw)/2:(oh-ih)/2,fps=10,setsar=1[c{i}]')
        sf.write(d/f'conversation_{i}.wav',r['audio'],r['sr'],subtype='PCM_16')
    inputs+=['-i',str(d/'conversation_0.wav'),'-i',str(d/'conversation_1.wav')]
    filters+=['[c0][c1]hstack=inputs=2[v]','[2:a][3:a]amix=inputs=2:duration=shortest:normalize=0,volume=.5[a]']
    media.run(['ffmpeg','-v','error','-nostdin','-y',*inputs,'-filter_complex_threads','1','-filter_complex',';'.join(filters),
               '-map','[v]','-map','[a]','-t',str(duration),'-c:v','libx264','-threads','2','-preset','veryfast','-crf','27',
               '-pix_fmt','yuv420p','-c:a','aac','-b:a','128k','-movflags','+faststart',str(d/'conversation.pending.mp4')])
    (d/'conversation.pending.mp4').replace(path)
    for i in range(2):(d/f'conversation_{i}.wav').unlink()
    return str(path.relative_to(OUT))


def render_pair(ann,item):
    pair=ann['pair'];p=OUT/'media'/pair/'pair.json'
    if p.exists():return json.loads(p.read_text())
    print('START',pair,len(ann['clips']),'segments',flush=True)
    people=[media.load_record(r) for r in item['records']];clips=[];errors=[]
    conversation_path=render_conversation(pair,people)
    for n,c in enumerate(ann['clips'],1):
        try:
            clips.append(render_clip(pair,c,people,n));print('CLIP',pair,n,c['duration'],flush=True)
        except Exception as e:
            errors.append(dict(turn_id=c['turn_id'],start=c['start'],end=c['end'],error=str(e)))
            print('SKIP',pair,n,str(e)[:150],flush=True)
    result={k:ann[k] for k in ['pair','subset','split','sources','duration','turns']}
    result.update(clips=clips,errors=errors,segmentation_revision=2,conversation_path=conversation_path,
                  conversation_roles=[r['id'] for r in people])
    save(p,result);print('DONE',pair,len(clips),'clips',flush=True);return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--render',action='store_true');ap.add_argument('--workers',type=int,default=4)
    args=ap.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    old=json.loads((V1/'manifest.json').read_text());selection=json.loads((V1/'selection.json').read_text())
    records={p['pair']:p for p in selection['attempts']}
    anns=[annotate(p,records[p['pair']]) for p in old['pairs']]
    for ann in anns:save(OUT/'annotations'/f'{ann["pair"]}.json',ann)
    stats={'pairs':len(anns),'floor_turns':sum(len(p['turns']) for p in anns),'listening_segments':sum(len(p['clips']) for p in anns),
           'duration_seconds':round(sum(c['duration'] for p in anns for c in p['clips']),3),
           'events':dict(collections.Counter(e['label'] for p in anns for t in p['turns'] for e in t['events']))}
    save(OUT/'annotation_summary.json',stats);print('ANNOTATED',json.dumps(stats),flush=True)
    if not args.render:return
    manifest=dict(seed=media.SEED,segmentation_revision=2,method='rules_only_no_model',rules=RULES,
                  source_pairs=50,pairs=[],counts={},annotation_summary=stats)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs={pool.submit(render_pair,p,records[p['pair']]):p['pair'] for p in anns}
        for future in concurrent.futures.as_completed(jobs):
            result=future.result();manifest['pairs'].append(result);manifest['pairs'].sort(key=lambda p:p['pair'])
            manifest['counts']={'naturalistic':len(manifest['pairs'])}
            save(OUT/'manifest.json',manifest)
            print('PROGRESS',len(manifest['pairs']),sum(len(p['clips']) for p in manifest['pairs']),flush=True)
    print('FINISHED',flush=True)


if __name__=='__main__':main()
