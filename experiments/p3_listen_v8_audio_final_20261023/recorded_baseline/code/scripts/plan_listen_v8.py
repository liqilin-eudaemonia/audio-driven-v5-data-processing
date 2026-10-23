"""Independent V8 planner. Frozen V7 sources, outputs and QC remain untouched."""
import json
import math
from pathlib import Path
import tarfile
import numpy as np
import soundfile as sf

import prepare_listen_300h as base
import listen_turns_v8 as rules
from listen_policy_v8 import POLICY,sentence_units,group_units,listener_compatibility,acoustic_cut
from listen_policy_v7 import boundary_activity
from listen_policy_v6 import require_upright,reviewed_noise_hits

ROOT=base.ROOT


def video_stream(source):
    probe=json.loads(base.media.run(['ffprobe','-v','error','-select_streams','v:0',
        '-show_streams','-of','json',base.media.uri(source)]))
    streams=probe['streams']
    if not streams:
        raise ValueError('source_missing_video_stream: '+json.dumps(source,sort_keys=True))
    return streams[0]


def load_person(record):
    """Keep the frozen legacy loader; diagnose its empty-video IndexError."""
    try:
        return base.load_person(record)
    except IndexError:
        # Reprobe only failures. Do not relabel unrelated indexing bugs as
        # source failures, or swallow subprocess / I/O errors.
        with tarfile.open(record['archive'],'r:') as tar:
            member=tar.getmember(record['id']+'.mp4')
            source=dict(archive=record['archive'],member=member.name,
                offset=member.offset_data,byte_size=member.size)
        video_stream(source)
        raise


def raw_rms(source):
    """10 ms energy from original PCM, not normalized or preview AAC."""
    member=str(Path(source['member']).with_suffix('.wav'))
    with tarfile.open(source['archive'],'r:') as tar:
        with sf.SoundFile(tar.extractfile(member)) as f:
            sr=f.samplerate
            if sr%100:raise ValueError('Unsupported RMS timebase')
            hop=sr//100;parts=[]
            while True:
                wave=f.read(hop*10000,dtype='float32',always_2d=True)
                if not len(wave):break
                mono=wave.mean(axis=1);n=len(mono)//hop
                if n:parts.append(np.sqrt(np.mean(mono[:n*hop].reshape(n,hop).astype(np.float64)**2,axis=1)))
    return np.concatenate(parts) if parts else np.empty(0)


def inspect_pair(item):
    loaded=[load_person(item[k]) for k in ('a','b')]
    records=[x[0] for x in loaded];geom=[x[1] for x in loaded];sources=[x[2] for x in loaded]
    people=[rules.prepare(r,i) for i,r in enumerate(records)]
    limit=min(s['duration'] for s in sources);orientations=[]
    for g,s in zip(geom,sources):
        probe=video_stream(s['source'])
        angles=[float(probe.get('tags',{}).get('rotate',0))]+[float(d.get('rotation',0)) for d in probe.get('side_data_list',[])]
        orientations.append(require_upright(g['arrays']['boxes_and_keypoints:keypoints'],g['width'],g['height'],next((a for a in angles if a%360),0)))
    floors,decisions=rules.track_floors(people,limit)
    takeover_ids={f['start_anchor'] for f in floors}
    for p in people:
        for s in p['segments']:s['formal_takeover']=s['id'] in takeover_ids
    candidates=[];unresolved=[]
    for si in (0,1):
        eligible=[]
        for u in sentence_units(people[si]['words']):
            nominated=[f for f in floors if f['speaker_index']==si and any(
                f['start']<=w['start']<f['end'] for w in u['words'])]
            if not nominated:continue
            comp=listener_compatibility(people[1-si],u['start'],u['end'])
            if comp['substantive']:
                unresolved.append(dict(speaker_index=si,start=u['start'],end=u['end'],text=u['text'],
                    reason='complete_sentence_crosses_listener_substantive_speech',evidence=comp));continue
            eligible.append(u)
        groups,bad=group_units(eligible,people[1-si]);unresolved.extend(dict(speaker_index=si,**r) for r in bad)
        candidates.extend(dict(g,speaker_index=si) for g in groups)
    candidates.sort(key=lambda r:(r['start'],r['end']))
    conflicts=set()
    for i,c in enumerate(candidates):
        for j in range(i+1,len(candidates)):
            d=candidates[j]
            if d['start']>=c['end']-1e-6:break
            conflicts.update((i,j))
    clean=[]
    for i,c in enumerate(candidates):
        if i in conflicts:unresolved.append(dict(start=c['start'],end=c['end'],speaker_index=c['speaker_index'],reason='ambiguous_overlapping_roles',text=c['text']))
        else:clean.append(c)
    rms=[raw_rms(s['source']) for s in sources] if clean else []
    reviews=json.loads((ROOT/'data_reviews/listener_noise_rejections.json').read_text())['rows']
    clips=[]
    for i,u in enumerate(clean):
        si=u['speaker_index'];li=1-si
        left=clips[-1]['end'] if clips else 0.
        right=clean[i+1]['start'] if i+1<len(clean) else limit
        next_formal=[f['start'] for f in floors if f['speaker_index']==li and f['start']>=u['end']-1e-6]
        if next_formal:right=min(right,min(next_formal))
        cut,error=acoustic_cut(u,people,rms,si,limit,left,right)
        if error:
            unresolved.append(dict(start=u['start'],end=u['end'],speaker_index=si,reason=error,text=u['text']));continue
        try:
            crop=rules.full_turn_crop(geom[li],cut['start'],cut['end'])
            if crop['size']<448 or crop['wide_crop']:raise ValueError('original_crop_below_448' if crop['size']<448 else 'wide_crop')
        except ValueError as e:
            unresolved.append(dict(start=cut['start'],end=cut['end'],speaker_index=si,reason=str(e),text=u['text']));continue
        hits=reviewed_noise_hits(records[li]['id'],cut['start'],cut['end'],reviews)
        events=[dict(start=s['start'],end=s['end'],text=s['text'],label='feedback_candidate' if s['feedback'] else 'short_speech_unconfirmed')
                for s in people[li]['segments'] if s['end']>cut['start'] and s['start']<cut['end']]
        row=dict(sample_id=f"{item['pair']}_v8_s{si}_{i:03d}",pair=item['pair'],split=item['split'],group_id=item['group_id'],
            policy_revision=8,**cut,speaker_index=si,listener_index=li,speaker_id=records[si]['id'],listener_id=records[li]['id'],
            listener_crop=crop,orientations=orientations,listener_events=events,
            boundary_activity=boundary_activity(cut['start'],cut['end'],people[li]),
            source_manual_rejections=hits,unique_seconds=cut['duration_s'],fps=25,width=512,height=512,
            source_backed=True,training_admitted=False,audio_qc_status='not_recomputed',
            sources={role:sources[index]['source'] for role,index in [('speaker',si),('listener',li)]},
            end_reason=cut['semantic_end'],frame_adaptation_required=True)
        clips.append(row)
    assert all(a['end']<=b['start']+1e-6 for a,b in zip(clips,clips[1:]))
    return dict(pair=item['pair'],policy=POLICY,clips=clips,unresolved=unresolved,
        floors=floors,decisions=decisions,source_seconds=limit,
        composed_feedback=[s for p in people for s in p['segments'] if s['feedback']],
        source_audio_for_boundaries='original PCM, 10ms RMS',ok=True)
