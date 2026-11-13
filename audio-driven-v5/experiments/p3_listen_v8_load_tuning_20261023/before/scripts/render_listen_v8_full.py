"""Export V8 non-excluded candidates: listener video and separate normalized WAVs.

Review candidates remain quarantined. No model frame-count truncation or training.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

import soundfile as sf
import run_listen_v8_full as full
from p3_listen.common import read,write,sha,digest

OUT=full.OUT


def run(cmd):
    p=subprocess.run(cmd,capture_output=True,text=True,timeout=300)
    if p.returncode:raise RuntimeError(p.stderr[-4000:])
    return p.stdout


def export(row,verified_tracks):
    folder=OUT/'media'/row['split']/row['effective_decision']/row['pair']/row['sample_id']
    folder.mkdir(parents=True,exist_ok=True)
    marker=folder/'export.json'
    identity=digest({k:row[k] for k in ('sample_id','start','end','listener_crop','sources','normalized_audio')})
    if marker.exists():
        old=read(marker)
        if old['source_contract']==identity and all(sha(folder/k)==h for k,h in old['files_sha256'].items()):return old
    n=round(row['duration_s']*48000)
    for role in ('speaker','listener'):
        track=row['normalized_audio'][role]
        key=(track['path'],track['sha256'])
        if key not in verified_tracks:
            assert sha(track['path'])==track['sha256'],'Normalized source changed'
            verified_tracks.add(key)
        with sf.SoundFile(track['path']) as f:
            assert f.samplerate==48000 and f.channels==1
            f.seek(round(row['start']*48000));wave=f.read(n,dtype='int32')
        assert len(wave)==n
        pending=folder/(role+'.pending.wav');sf.write(pending,wave,48000,subtype='PCM_24')
        assert sf.info(pending).frames==n
        pending.replace(folder/(role+'.wav'))
    src=row['sources']['listener'];c=row['listener_crop']
    assert c['size']>=448 and not c['wide_crop']
    uri=f"subfile,,start,{src['offset']},end,{src['offset']+src['byte_size']},,:{src['archive']}"
    pending=folder/'listener.pending.mp4'
    vf=f"setpts=PTS-STARTPTS,crop={c['size']}:{c['size']}:{c['x']}:{c['y']},scale=512:512,fps=25,setsar=1"
    run(['ffmpeg','-v','error','-nostdin','-y','-threads','1','-filter_threads','1',
         '-noautorotate','-ss',str(row['start']),'-i',uri,'-i',str(folder/'listener.wav'),
         '-map','0:v:0','-map','1:a:0','-t',str(row['duration_s']),'-vf',vf,
         '-c:v','libx264','-threads','1','-preset','fast','-crf','17','-pix_fmt','yuv420p',
         '-c:a','aac','-b:a','192k','-metadata:s:v:0','rotate=0','-movflags','+faststart',str(pending)])
    probe=json.loads(run(['ffprobe','-v','error','-show_streams','-of','json',str(pending)]))
    v=next(x for x in probe['streams'] if x['codec_type']=='video')
    a=next(x for x in probe['streams'] if x['codec_type']=='audio')
    assert (v['width'],v['height'],a['channels'])==(512,512,1)
    assert v['codec_name']=='h264' and a['codec_name']=='aac' and v['pix_fmt']=='yuv420p'
    assert abs(float(v['duration'])-row['duration_s'])<=.081
    assert abs(float(a['duration'])-row['duration_s'])<=.025
    run(['ffmpeg','-v','error','-xerror','-threads','1','-i',str(pending),'-map','0:v','-map','0:a','-f','null','-'])
    pending.replace(folder/'listener.mp4')
    run(['ffmpeg','-v','error','-nostdin','-y','-threads','1','-filter_threads','1',
         '-i',str(folder/'listener.mp4'),'-frames:v','1','-threads','1',str(folder/'reference.jpg')])
    files=('listener.mp4','speaker.wav','listener.wav','reference.jpg')
    value=dict(sample_id=row['sample_id'],pair=row['pair'],split=row['split'],start=row['start'],end=row['end'],
        duration_s=row['duration_s'],folder=str(folder),source_contract=identity,files_sha256={f:sha(folder/f) for f in files},
        audio_samples=n,video_seconds=float(v['duration']),audio_seconds=float(a['duration']),decoded=True,
        audio_qc_status=row['audio_qc_status'],v8_combined_status=row['v8_combined_status'],
        boundary_review_flags=row['boundary_review_flags'],admission_pool=row['admission_pool'],
        training_admitted=False,frame_adaptation_required=True)
    write(marker,value);return value


def pair(path):
    result=read(path);outputs=[];verified=set()
    for row in result['clips']:
        if row['effective_decision']=='exclude':continue
        outputs.append(export(row,verified))
    value=dict(pair=result['pair'],complete=True,clips=outputs,utc=full.now())
    write(OUT/'render_results'/(result['pair']+'.json'),value)
    return value


def main(workers):
    full.verify()
    for d in ('render_results','render_errors'):(OUT/d).mkdir(exist_ok=True)
    with (OUT/'render.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        paths=iter(sorted((OUT/'results').glob('*.json')));counts=Counter();hours=Counter();started=time.monotonic()
        def status():
            write(OUT/'render_summary.json',dict(utc=full.now(),workers=workers,counts=dict(counts),
                exported_hours=dict(hours),wall_seconds=time.monotonic()-started,training_ready=False))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            pending={}
            def fill():
                while len(pending)<workers:
                    path=next(paths,None)
                    if path is None:break
                    pending[pool.submit(pair,path)]=path
            fill()
            while pending:
                done,_=wait(pending,return_when=FIRST_COMPLETED)
                for future in done:
                    path=pending.pop(future)
                    try:
                        value=future.result();counts['complete_pairs']+=1
                        for row in value['clips']:
                            key=row['split']+'/'+row['audio_qc_status'];counts[key]+=1;hours[key]+=row['duration_s']/3600
                    except Exception as e:
                        counts['error_pairs']+=1
                        write(OUT/'render_errors'/(path.stem+'.json'),dict(pair=path.stem,error=repr(e),traceback=traceback.format_exc()))
                fill();status()
        handles={}
        try:
            for path in sorted((OUT/'render_results').glob('*.json')):
                for row in read(path)['clips']:
                    key='_'.join((row['split'],row['audio_qc_status'],row['admission_pool']))
                    if key not in handles:handles[key]=(OUT/'manifests'/('rendered_'+key+'.jsonl.pending')).open('w')
                    handles[key].write(json.dumps(row,ensure_ascii=False)+'\n')
        finally:
            for h in handles.values():h.close()
        for k in handles:(OUT/'manifests'/('rendered_'+k+'.jsonl.pending')).replace(OUT/'manifests'/('rendered_'+k+'.jsonl'))
        status()
        if counts['error_pairs']:raise RuntimeError('Rendering had errors; see render_errors and resume')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=128);a=p.parse_args()
    if a.workers<1:p.error('workers must be positive')
    main(a.workers)
