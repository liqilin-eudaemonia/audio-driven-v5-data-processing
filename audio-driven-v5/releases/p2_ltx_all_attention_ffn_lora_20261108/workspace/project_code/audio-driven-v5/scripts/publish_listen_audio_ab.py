"""Publish A/B/C listener audio for ten reviewed videos and four sound diagnostics."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import sys
import time
import numpy as np
import soundfile as sf

ROOT=Path(__file__).resolve().parents[1]
os.environ['PATH']=str(ROOT/'scripts/media_bin')+os.pathsep+os.environ['PATH']
sys.path.insert(0,str(ROOT/'src'))
from p3_listen import media
from p3_listen.common import read,write,sha,digest
from listen_audio_ab import SR,PROFILES,analyze,reduce,anchor_rms,db,metrics

EXP=ROOT/'experiments/p3_listen_v6_audio_ab'
OUT=ROOT/'data_previews/seamless_listen_v6_audio_ab'
OLD=ROOT/'data_previews/seamless_listen_v6_review'
VERSION='audio-ab-v1'


def annotations(record):
    meta=record['metadata']
    vad=[[v['start'],v['end']] for v in meta.get('metadata:vad',[]) if v['end']>v['start']]
    words=[[w['start'],w['end']] for s in meta.get('metadata:transcript',[]) for w in s.get('words',[])
           if isinstance(w.get('start'),(int,float)) and isinstance(w.get('end'),(int,float)) and w['end']>w['start']]
    return vad,words


def process_source(record):
    ident=record['id'];folder=EXP/'tracks'/ident;folder.mkdir(parents=True,exist_ok=True)
    expected=dict(version=VERSION,source=record['source'],normalization=record['normalization'],profiles=PROFILES,
                  code_sha256=sha(ROOT/'scripts/listen_audio_ab.py'))
    if (folder/'complete.json').exists():
        old=read(folder/'complete.json')
        if old['contract']==expected and all(sha(folder/(m+'.wav'))==old['files'][m] for m in PROFILES):return old
    # Extract float source audio, before our loudness gain or any preview codec.
    raw_path=folder/'source.wav'
    media.command(['ffmpeg','-v','error','-nostdin','-y','-threads','1','-i',media.uri(record['source'],'.wav'),
                   '-ar',str(SR),'-ac','1','-c:a','pcm_f32le',str(raw_path)])
    raw,sr=sf.read(raw_path,dtype='float32');reference,ref_sr=sf.read(record['normalization']['path'],dtype='float32')
    if sr!=SR or ref_sr!=SR or len(reference)!=len(raw):raise ValueError('Source length/rate mismatch')
    vad,words=annotations(record);prepared=analyze(raw,vad,words)
    anchor_before=anchor_rms(reference,words);gain=10**(record['normalization']['gain_db']/20)
    reports={}
    for mode in PROFILES:
        cleaned=reduce(prepared,mode)*gain
        anchor_clean=anchor_rms(cleaned,words)
        correction=0. if anchor_before is None or not anchor_clean else float(np.clip(db(anchor_before/anchor_clean),-1.5,1.5))
        cleaned*=10**(correction/20)
        pending=folder/(mode+'.prelimit.wav');sf.write(pending,cleaned,SR,subtype='FLOAT')
        peak=10**(-1/20)
        filters=(f'aresample=192000,apad=pad_len=959,alimiter=limit={peak}:level=false:attack=5:release=50,'
                 'atrim=start_sample=959,asetpts=PTS-STARTPTS,aresample=48000')
        media.command(['ffmpeg','-v','error','-nostdin','-y','-threads','1','-i',str(pending),'-af',filters,
                       '-c:a','pcm_s24le',str(folder/(mode+'.wav'))])
        output,_=sf.read(folder/(mode+'.wav'),dtype='float32')
        if len(output)!=len(raw) or not np.isfinite(output).all():raise ValueError('Output temporal/finite contract')
        anchor_after=anchor_rms(output,words)
        reports[mode]=dict(anchor_gain_correction_db=correction,
            speech_anchor_change_db=None if not anchor_before or not anchor_after else db(anchor_after/anchor_before),
            sample_peak_dbfs=db(np.max(np.abs(output))),samples=len(output))
        pending.unlink()
    result=dict(contract=expected,reports=reports,noise_profile_available=prepared['noise_profile_available'],
                noise_reference_seconds=prepared['noise_reference_seconds'],
                files={mode:sha(folder/(mode+'.wav')) for mode in PROFILES})
    write(folder/'complete.json',result)
    print(json.dumps(dict(source=ident,reports=reports)),flush=True)
    return result


def publish_clip(row,inventory):
    ident=row['listener_id'];record=inventory[ident];folder=OUT/'media'/row['sample_id'];folder.mkdir(parents=True,exist_ok=True)
    report=read(EXP/'tracks'/ident/'complete.json');count=round(row['duration_s']*SR);start=round(row['start']*SR)
    waves={}
    for mode in ('baseline','mild','strong'):
        source=Path(record['normalization']['path']) if mode=='baseline' else EXP/'tracks'/ident/(mode+'.wav')
        with sf.SoundFile(source) as f:f.seek(start);wave=f.read(count,dtype='float32')
        if len(wave)!=count:raise ValueError('Preview audio truncated')
        sf.write(folder/(mode+'.wav'),wave,SR,subtype='PCM_24');waves[mode]=wave
    if row.get('video'):
        old=OLD/row['preview'];shutil.copyfile(old/'reference.jpg',folder/'reference.jpg')
        # Speaker audio and all video packets stay identical across A/B/C.
        # Only listener processing changes, making the listener-side effect audible.
        for mode in waves:
            media.command(['ffmpeg','-v','error','-nostdin','-y','-threads','1','-i',str(old/'silent.mp4'),
                '-i',str(old/'speaker.wav'),'-i',str(folder/(mode+'.wav')),
                '-filter_complex','[1:a][2:a]amerge=inputs=2[a]','-map','0:v','-map','[a]',
                '-c:v','copy','-c:a','aac','-b:a','192k','-t',str(row['duration_s']),
                '-movflags','+faststart',str(folder/(mode+'.mp4'))])
            media.command(['ffmpeg','-v','error','-nostdin','-y','-threads','1','-i',str(old/'silent.mp4'),
                '-i',str(folder/(mode+'.wav')),'-map','0:v','-map','1:a','-c:v','copy','-c:a','aac','-b:a','160k',
                '-t',str(row['duration_s']),'-movflags','+faststart',str(folder/(mode+'_listener.mp4'))])
    vad,words=annotations(record)
    events=[dict(start=max(a,row['start'])-row['start'],end=min(b,row['end'])-row['start'])
            for a,b in vad if b>row['start'] and a<row['end']]
    result=dict(row,path=f"media/{row['sample_id']}/",events=events,
        metrics={mode:metrics(waves['baseline'],waves[mode]) for mode in ('mild','strong')},
        source_reports=report['reports'],noise_profile_available=report['noise_profile_available'])
    event_metrics=[]
    for e in events:
        a,b=round(e['start']*SR),round(e['end']*SR)
        if b>a:
            event_metrics.append(dict(**e,changes={m:metrics(waves['baseline'][a:b],waves[m][a:b])['rms_change_db'] for m in ('mild','strong')}))
    result['event_metrics']=event_metrics
    # Byte-level video comparison plus full decode / exact WAV length verification.
    import av,hashlib
    video_digests=[]
    for mode in waves:
        info=sf.info(folder/(mode+'.wav'))
        if (info.frames,info.samplerate,info.channels)!=(count,SR,1):raise ValueError('WAV contract')
        if row.get('video'):
            with av.open(str(folder/(mode+'.mp4'))) as f:
                f.streams.video[0].thread_count=1
                frames=list(f.decode(video=0))
                if len(frames)!=row['frames']:raise ValueError('Video frames changed')
            with av.open(str(folder/(mode+'.mp4'))) as f:
                h=hashlib.sha256()
                for packet in f.demux(video=0):h.update(bytes(packet))
                video_digests.append(h.hexdigest())
            with av.open(str(folder/(mode+'.mp4'))) as f:
                if f.streams.audio[0].codec_context.channels!=2 or not list(f.decode(audio=0)):raise ValueError('Audio codec contract')
    if video_digests and len(set(video_digests))!=1:raise ValueError('A/B/C video packets differ')
    if row.get('video'):
        for mode in waves:
            with av.open(str(folder/(mode+'_listener.mp4'))) as f:
                f.streams.video[0].thread_count=1
                if len(list(f.decode(video=0)))!=row['frames']:raise ValueError('Listener video frames changed')
            with av.open(str(folder/(mode+'_listener.mp4'))) as f:
                if f.streams.audio[0].codec_context.channels!=1 or not list(f.decode(audio=0)):raise ValueError('Listener audio contract')
    result['verified']=True
    print(json.dumps(dict(preview=row['sample_id'],events=event_metrics)),flush=True)
    return result


def diagnostic_rows(rows,inventory):
    result=[]
    # These are explicit review-only sound probes, NOT newly qualified training clips.
    for row in rows:
        record=inventory[row['listener_id']];vad,words=annotations(record)
        candidates=[]
        for a,b in vad:
            if not .25<=b-a<=1.5 or a<1:continue
            if not (b<row['start']-1 or a>row['end']+1):continue
            if any(y>a-.2 and x<b+.2 for x,y in words):continue
            candidates.append((a,b))
        if not candidates:continue
        a,b=candidates[0];start=max(0.,a-.75);end=b+.9
        result.append(dict(sample_id=row['listener_id']+'_untranscribed_probe',listener_id=row['listener_id'],
            pair=row['pair'],start=start,end=end,duration_s=round(end-start,9),video=False,
            category='未知短声音诊断（是否笑声需听辨）',
            text='没有文字转写的短声音；可能为非语音反应，也可能为噪声。仅用于检查误伤风险，不视为训练合格样本。'))
        if len(result)==4:break
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=3);a=p.parse_args()
    inventory=read(EXP/'source_inventory.json');source=read(OLD/'samples.json')
    rows=[dict(r,video=True,category='已报告杂音' if 'S1324' in r['pair'] else
               '短附和 / 未知发声' if r['listener_screen']['listener_vad_seconds'] else '安静倾听') for r in source['samples']]
    rows.sort(key=lambda r:('S1324' not in r['pair'],r['category']=='安静倾听'))
    rows+=diagnostic_rows(rows,inventory)
    if len(rows)>14:raise ValueError('Only bounded small review samples are authorized')
    OUT.mkdir(parents=True,exist_ok=True)
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        list(pool.map(process_source,[inventory[i] for i in dict.fromkeys(r['listener_id'] for r in rows)]))
        samples=list(pool.map(lambda r:publish_clip(r,inventory),rows))
    write(OUT/'samples.json',dict(version=VERSION,created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        status='audio_quality_review_only',profiles=PROFILES,samples=samples,
        verified=True,known_laughter_labels=False,notes='No formal dataset, cache or training changes.'))
    shutil.copyfile(ROOT/'scripts/listen_audio_ab.html',OUT/'index.html')
    link=ROOT/'data_previews/seamless_face_listening_v1/listen_audio_ab'
    if not link.exists():link.symlink_to(OUT,target_is_directory=True)
    print(json.dumps(dict(url='http://127.0.0.1:8875/listen_audio_ab/',samples=len(samples))),flush=True)


if __name__=='__main__':main()
