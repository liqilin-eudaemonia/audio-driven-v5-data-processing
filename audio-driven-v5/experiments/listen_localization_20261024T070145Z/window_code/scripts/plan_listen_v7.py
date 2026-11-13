"""V7 source-time planner; natural durations and per-child original-pixel geometry."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import shutil
import sys
import time

import prepare_listen_300h as base
from listen_policy_v7 import POLICY,natural_clips,listener_screen,require_upright,reviewed_noise_hits,boundary_activity
import listen_turns_v7 as rules
from p3_listen.common import read,write,digest
from p3_listen import media

ROOT=base.ROOT
OLD=base.OUT
OUT=ROOT/'data_corpora/seamless_listen_v7_full'
REPORTED='naturalistic_V03_S0985_I00000432'


def inspect_pair(item,defer_audio_uncertainty=False):
    name=item['pair']; rejected=[];clips=[];orientations=[]
    review_path=ROOT/'data_reviews/listener_noise_rejections.json'
    reviews=read(review_path)['rows'] if review_path.exists() else []
    try:
        loaded=[base.load_person(item[k]) for k in ('a','b')]
        records=[x[0] for x in loaded];geom=[x[1] for x in loaded];sources=[x[2] for x in loaded]
        people=[rules.prepare(r,i) for i,r in enumerate(records)]
        # Reject the whole pair if either raw participant video is rotated or uncertain.
        for g,s in zip(geom,sources):
            probe=json.loads(base.media.run(['ffprobe','-v','error','-select_streams','v:0',
                '-show_streams','-of','json',base.media.uri(s['source'])]))['streams'][0]
            rotations=[float(probe.get('tags',{}).get('rotate',0))]+[
                float(d.get('rotation',0)) for d in probe.get('side_data_list',[])]
            orientations.append(require_upright(g['arrays']['boxes_and_keypoints:keypoints'],
                g['width'],g['height'],next((r for r in rotations if r%360),0)))
        annotated=rules.annotate(dict(pair=name,subset=item['label'],sources=sources),dict(records=records),
                                     defer_untranscribed=defer_audio_uncertainty)
        for ci,c in enumerate(annotated['clips']):
            a,b=c['start'],c['end'];si=c['speaker_index'];li=c['listener_index']
            parts,bad=natural_clips(a,b,people[si]['words'],people[si]['vad'],people[li]['vad'])
            rejected.extend(dict(parent=ci,**v) for v in bad)
            for j,w in enumerate(parts):
                try:crop=rules.full_turn_crop(geom[li],w['start'],w['end'])
                except ValueError as e:
                    rejected.append(dict(parent=ci,start=w['start'],end=w['end'],reason=str(e)));continue
                if crop['size']<448 or crop['wide_crop']:
                    rejected.append(dict(parent=ci,start=w['start'],end=w['end'],reason='original_crop_below_448' if crop['size']<448 else 'wide_crop',crop=crop));continue
                hits=reviewed_noise_hits(c['listener'],w['start'],w['end'],reviews)
                if hits:
                    rejected.append(dict(parent=ci,start=w['start'],end=w['end'],
                        reason='user_confirmed_source_noise',review_samples=[h['sample_id'] for h in hits]));continue
                screen=listener_screen(people[li],w['start'],w['end'])
                if defer_audio_uncertainty:
                    # Missing words and activity density are review evidence, not noise truth.
                    screen['deferred_reasons']=[r for r in screen['reasons'] if r!='listener_substantive_speech']
                    screen['reasons']=[r for r in screen['reasons'] if r=='listener_substantive_speech']
                    screen['accepted']=not screen['reasons']
                if not screen['accepted']:
                    rejected.append(dict(parent=ci,start=w['start'],end=w['end'],reason=screen['reasons'],screen=screen));continue
                row=dict(sample_id=f'{name}_v7_p{ci:03d}_{j:02d}',pair=name,split=item['split'],group_id=item['group_id'],
                    policy_revision=7,**w,fps=25,width=512,height=512,
                    speaker_id=c['speaker'],listener_id=c['listener'],speaker_index=si,listener_index=li,
                    parent_start=a,parent_end=b,listener_crop=crop,orientations=orientations,listener_screen=screen,
                    boundary_activity=boundary_activity(w['start'],w['end'],people[li]),
                    duration_tier='ge3s' if w['duration_s']>=3-1e-7 else '2to3s',
                    listener_events=[e for e in c['listener_events'] if e['end']>w['start'] and e['start']<w['end']],
                    sources={role:sources[c[role+'_index']]['source'] for role in ('speaker','listener')})
                clips.append(row)
        result=dict(pair=name,clips=clips,rejected=rejected,orientations=orientations,ok=True,policy=POLICY,
            source_seconds=min(s['duration'] for s in sources),floor_turns=annotated['turns'],
            parent_candidates=len(annotated['clips']))
    except Exception as e:
        result=dict(pair=name,clips=[],rejected=[dict(reason=str(e))],ok=False,policy=POLICY)
    write(OUT/'audit'/(name+'.json'),result)
    print(json.dumps(dict(pair=name,accepted=len(result['clips']),rejected=len(result['rejected']),
                         error=result['rejected'] if not result['ok'] else None)),flush=True)
    return result

