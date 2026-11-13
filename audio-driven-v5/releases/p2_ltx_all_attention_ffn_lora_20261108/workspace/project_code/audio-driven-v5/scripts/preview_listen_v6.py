"""Bounded rule review only. Never selects 300h, creates caches, or starts training."""
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
from listen_policy_v6 import POLICY,natural_clips,listener_screen,require_upright,reviewed_noise_hits
from p3_listen.common import read,write,digest
from p3_listen import media

ROOT=base.ROOT
OLD=base.OUT
OUT=ROOT/'data_previews/seamless_listen_v6_review'
REPORTED='naturalistic_V03_S0985_I00000432'


def inspect_pair(item,defer_audio_uncertainty=False):
    name=item['pair']; rejected=[];clips=[];orientations=[]
    review_path=ROOT/'data_reviews/listener_noise_rejections.json'
    reviews=read(review_path)['rows'] if review_path.exists() else []
    try:
        loaded=[base.load_person(item[k]) for k in ('a','b')]
        records=[x[0] for x in loaded];geom=[x[1] for x in loaded];sources=[x[2] for x in loaded]
        people=[base.rules.prepare(r,i) for i,r in enumerate(records)]
        # Reject the whole pair if either raw participant video is rotated or uncertain.
        for g,s in zip(geom,sources):
            probe=json.loads(base.media.run(['ffprobe','-v','error','-select_streams','v:0',
                '-show_streams','-of','json',base.media.uri(s['source'])]))['streams'][0]
            rotations=[float(probe.get('tags',{}).get('rotate',0))]+[
                float(d.get('rotation',0)) for d in probe.get('side_data_list',[])]
            orientations.append(require_upright(g['arrays']['boxes_and_keypoints:keypoints'],
                g['width'],g['height'],next((r for r in rotations if r%360),0)))
        annotated=base.rules.annotate(dict(pair=name,subset=item['label'],sources=sources),dict(records=records),
                                     defer_untranscribed=defer_audio_uncertainty)
        for ci,c in enumerate(annotated['clips']):
            a,b=c['start'],c['end'];si=c['speaker_index'];li=c['listener_index']
            # Preserve the original parent crop: do not enlarge/recompute it per short child.
            ws=[w for w in people[si]['words'] if w['start']<b-.001 and w['end']>a+.001]
            if not ws:continue
            ca=max(a,min(w['start'] for w in ws));cb=min(b,max(w['end'] for w in ws))
            try:crop=base.rules.full_turn_crop(geom[li],ca,cb)
            except ValueError as e:
                rejected.append(dict(parent=ci,reason=str(e)));continue
            if crop['size']<512 or crop['wide_crop']:
                rejected.append(dict(parent=ci,reason='original_crop_below_512' if crop['size']<512 else 'wide_crop',crop=crop));continue
            parts,bad=natural_clips(a,b,people[si]['words'],people[si]['vad'],people[li]['vad'])
            rejected.extend(dict(parent=ci,**v) for v in bad)
            for j,w in enumerate(parts):
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
                row=dict(sample_id=f'{name}_v6_p{ci:03d}_{j:02d}',pair=name,split=item['split'],group_id=item['group_id'],
                    policy_revision=6,**w,fps=25,width=512,height=512,
                    speaker_id=c['speaker'],listener_id=c['listener'],speaker_index=si,listener_index=li,
                    parent_start=a,parent_end=b,listener_crop=crop,orientations=orientations,listener_screen=screen,
                    listener_events=[e for e in c['listener_events'] if e['end']>w['start'] and e['start']<w['end']],
                    sources={role:sources[c[role+'_index']]['source'] for role in ('speaker','listener')})
                clips.append(row)
        result=dict(pair=name,clips=clips,rejected=rejected,orientations=orientations,ok=True,policy=POLICY)
    except Exception as e:
        result=dict(pair=name,clips=[],rejected=[dict(reason=str(e))],ok=False,policy=POLICY)
    write(OUT/'audit'/(name+'.json'),result)
    print(json.dumps(dict(pair=name,accepted=len(result['clips']),rejected=len(result['rejected']),
                         error=result['rejected'] if not result['ok'] else None)),flush=True)
    return result


def publish(row):
    import soundfile as sf
    folder=OUT/'media'/row['sample_id'];folder.mkdir(parents=True,exist_ok=True)
    if (OUT/'samples.json').exists():
        for old in read(OUT/'samples.json')['samples']:
            previous={k:v for k,v in old.items() if k not in ('preview','media_verified')}
            if previous==row and old.get('media_verified'):
                print(json.dumps(dict(reused_verified_preview=row['sample_id'])),flush=True)
                return old
    print(json.dumps(dict(exporting=row['sample_id'],seconds=row['duration_s'])),flush=True)
    normalized=read(OLD/'normalized_pairs'/(row['pair']+'.json'))
    if not normalized['summary']['ok']:raise ValueError('Missing qualified normalization')
    for role in ('speaker','listener'):
        track=normalized['tracks'][row[role+'_index']]
        with sf.SoundFile(track['path']) as f:
            if f.samplerate!=48000 or f.channels!=1:raise ValueError('Audio contract')
            f.seek(round(row['start']*48000));wave=f.read(round(row['duration_s']*48000),dtype='int32')
        if len(wave)!=round(row['duration_s']*48000):raise ValueError('Truncated audio')
        sf.write(folder/(role+'.wav'),wave,48000,subtype='PCM_24')
    crop=row['listener_crop']
    vf=(f"setpts=PTS-STARTPTS,crop={crop['size']}:{crop['size']}:{crop['x']}:{crop['y']},"
        'scale=512:512,fps=25:eof_action=pass,setsar=1')
    ff=['ffmpeg','-v','error','-nostdin','-y','-threads','8']
    media.command(ff+['-noautorotate','-ss',str(row['start']),'-i',media.uri(row['sources']['listener'],'.mp4'),
        '-vf',vf,'-frames:v',str(row['frames']),'-an','-c:v','libx264','-threads','1','-preset','fast',
        '-crf','16','-pix_fmt','yuv420p','-metadata:s:v:0','rotate=0','-movflags','+faststart',str(folder/'silent.mp4')])
    media.command(ff+['-i',str(folder/'silent.mp4'),'-frames:v','1',str(folder/'reference.jpg')])
    # Original uncropped listener view, same time span, for orientation/crop inspection.
    media.command(ff+['-noautorotate','-ss',str(row['start']),'-i',media.uri(row['sources']['listener'],'.mp4'),
        '-vf',f"drawbox=x={crop['x']}:y={crop['y']}:w={crop['size']}:h={crop['size']}:color=lime:t=5,scale=640:-2",
        '-frames:v','1',str(folder/'source.jpg')])
    cmd=ff+['-i',str(folder/'silent.mp4'),'-i',str(folder/'speaker.wav'),'-i',str(folder/'listener.wav'),
        '-filter_complex','[1:a][2:a]amerge=inputs=2[m]']
    for mode,channel in [('both','[m]'),('speaker','1:a'),('listener','2:a')]:
        cmd+=['-map','0:v','-map',channel,'-c:v','copy','-c:a','aac','-b:a','192k','-t',str(row['duration_s']),
              '-movflags','+faststart',str(folder/(mode+'.mp4'))]
    media.command(cmd)
    # Full decode, duration and channel verification, not just ffprobe headers.
    import av
    for mode,channels in [('both',2),('speaker',1),('listener',1)]:
        with av.open(str(folder/(mode+'.mp4'))) as video:
            stream=video.streams.video[0];stream.thread_count=1
            frames=list(video.decode(video=0))
            if len(frames)!=row['frames'] or any((f.width,f.height)!=(512,512) for f in frames):
                raise ValueError('Preview video geometry mismatch')
        with av.open(str(folder/(mode+'.mp4'))) as video:
            if video.streams.audio[0].codec_context.channels!=channels:raise ValueError('Audio channel mismatch')
            if abs(video.streams.video[0].duration*float(video.streams.video[0].time_base)-row['duration_s'])>.002:
                raise ValueError('Preview duration mismatch')
            if not list(video.decode(audio=0)):raise ValueError('Audio decode failed')
    print(json.dumps(dict(verified=row['sample_id'])),flush=True)
    return dict(row,preview=f"media/{row['sample_id']}/",media_verified=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--samples',type=int,default=12)
    p.add_argument('--max-pairs',type=int,default=32);p.add_argument('--workers',type=int,default=4)
    a=p.parse_args()
    if not 1<=a.samples<=24 or not 1<=a.max_pairs<=64:raise ValueError('Small review only; no bulk processing')
    OUT.mkdir(parents=True,exist_ok=True)
    cat=read(OLD/'catalog.json');allpairs=cat['train']+cat['dev'];lookup={r['pair']:r for r in allpairs}
    candidates=[];groups=set()
    # Start from already normalized pairs; no new loudness pass or corpus expansion.
    for item in read(OLD/'selected.json')['pairs']:
        record=lookup[item['pair']]
        if record['group_id'] in groups:continue
        norm=OLD/'normalized_pairs'/(item['pair']+'.json')
        if not norm.exists() or not read(norm)['summary']['ok']:continue
        plan=read(item['path'])
        if not any(c['crop_boxes']['listener']['size']>=512 and not c['crop_boxes']['listener']['wide_crop'] for c in plan.get('clips',[])):continue
        candidates.append(record);groups.add(record['group_id'])
        if len(candidates)>=a.max_pairs:break
    report=inspect_pair(lookup[REPORTED])
    audits=[report];selected=[];used=set()
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        # Bounded batches, stop once enough distinct pairs pass.
        for offset in range(0,len(candidates),a.workers):
            batch=list(pool.map(inspect_pair,candidates[offset:offset+a.workers]));audits.extend(batch)
            for result in batch:
                if result['pair'] in used or not result['clips']:continue
                # Prefer an accepted vocal feedback example when available, then long complete speech.
                rows=sorted(result['clips'],key=lambda r:(r['end_reason']=='sentence_end',r['listener_screen']['listener_vad_seconds']>0,r['duration_s']),reverse=True)
                selected.append(rows[0]);used.add(result['pair'])
                if len(selected)>=a.samples:break
            if len(selected)>=a.samples:break
        if not selected:raise ValueError('No qualified examples in bounded candidates')
        samples=list(pool.map(publish,selected))
    output=dict(created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),policy=POLICY,
        status='small_sample_rule_review_only_not_training_data_seal',samples=samples,
        inspected_pairs=len(audits),requested_samples=a.samples,reported_sample=report,
        accepted_seconds=sum(r['unique_seconds'] for r in samples),
        note='Candidate pool prefiltered for existing normalization and crop>=512; not an unbiased corpus yield estimate.')
    write(OUT/'samples.json',output)
    shutil.copyfile(ROOT/'scripts/listen_v6_review.html',OUT/'index.html')
    link=ROOT/'data_previews/seamless_face_listening_v1/listen_v6'
    if not link.exists():link.symlink_to(OUT,target_is_directory=True)
    print(json.dumps(dict(url='http://127.0.0.1:8875/listen_v6/',samples=len(samples),pairs=len(audits))),flush=True)


if __name__=='__main__':main()
