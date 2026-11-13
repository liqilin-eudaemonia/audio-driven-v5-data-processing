"""Publish a stratified sample of completed formal training media on port 8875."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from p3_listen import media
from p3_listen.common import read,write,sha
OUT=ROOT/'data_previews/seamless_listen_300h_sample'
EXP=ROOT/'experiments/p3_listen_300h'


def main():
    import numpy as np
    import soundfile as sf
    from PIL import Image
    OUT.mkdir(parents=True,exist_ok=True)
    manifest=read(EXP/'windows.json');rows=manifest['windows'];done=set()
    for split in ('train','dev'):
        for entry in os.scandir(EXP/'media'/split):
            if (Path(entry.path)/'complete.json').exists():done.add(entry.name)
    candidates=[r for r in rows if r['sample_id'] in done]
    rng=random.Random(20261022);rng.shuffle(candidates);selected=[];groups=set()
    for split,per_bin in [('train',4),('dev',2)]:
        for lo,hi in [(0,130),(130,170),(170,201),(201,202)]:
            pool=[r for r in candidates if r['split']==split and lo<=r['frames']<hi]
            n=0
            for row in pool:
                if row['group_id'] in groups:continue
                selected.append(row);groups.add(row['group_id']);n+=1
                if n==per_bin:break
            if n!=per_bin:raise ValueError('Insufficient completed duration/group stratum')
    def publish(row):
        p,_=media.verify_media(EXP,row);folder=OUT/'media'/row['sample_id'];folder.mkdir(parents=True,exist_ok=True)
        for k,name in [('speaker_audio','speaker.wav'),('target_audio','listener.wav')]:shutil.copyfile(p[k],folder/name)
        Image.open(p['listener_image']).convert('RGB').save(folder/'reference.jpg',quality=88)
        cmd=[str(ROOT/'scripts/media_bin/ffmpeg'),'-v','error','-nostdin','-y','-threads','1',
             '-i',str(p['target_video']),'-i',str(p['speaker_audio']),'-i',str(p['target_audio']),
             '-filter_complex','[1:a][2:a]amerge=inputs=2[m]']
        for mode,audio in [('both','[m]'),('speaker','1:a'),('listener','2:a')]:
            cmd+=['-map','0:v','-map',audio,'-c:v','copy','-c:a','aac','-b:a','192k','-t',str(row['duration_s']),'-movflags','+faststart',str(folder/(mode+'.mp4'))]
        subprocess.run(cmd,check=True,stdout=subprocess.DEVNULL)
        plan=read(ROOT/'data_corpora/seamless_listen_300h/plans'/row['split']/(row['pair']+'.json'))
        clip=next(c for c in plan['clips'] if c['start']==row['clip_start'] and c['end']==row['clip_end'] and c['listener']==row['listener_id'])
        peaks={}
        for name in ('speaker','listener'):
            wave,_=sf.read(folder/(name+'.wav'));peak=float(np.abs(wave).max())
            peaks[name]=round(20*math.log10(max(peak,1e-12)),1)
        return dict(**row,preview=f"media/{row['sample_id']}/",parent_start=clip['parent_start'],parent_end=clip['parent_end'],
            kind=clip['kind'],listener_events=[e for e in clip['listener_events'] if e['end']>row['start'] and e['start']<row['end']],
            sample_peak_dbfs=peaks,source_media_sha256=sha(p['marker']))
    with ThreadPoolExecutor(max_workers=4) as pool:sample=list(pool.map(publish,selected))
    write(OUT/'samples.json',dict(seed=20261022,created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        manifest_sha256=sha(EXP/'windows.json'),available_completed=len(done),total_windows=len(rows),samples=sample,
        description='24 stratified examples from completed final media; 16 train + 8 dev, distinct participant/session groups.'))
    shutil.copyfile(ROOT/'scripts/listen_300h_preview.html',OUT/'index.html')
    link=ROOT/'data_previews/seamless_face_listening_v1/listen_300h'
    if link.is_symlink():link.unlink()
    link.symlink_to(OUT,target_is_directory=True)
    print(json.dumps(dict(url='http://127.0.0.1:8875/listen_300h/',samples=len(sample),available=len(done),directory=str(OUT)),ensure_ascii=False),flush=True)

if __name__=='__main__':main()
