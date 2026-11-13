"""Export the already-reviewed full-source split results as individual clips.

No replanning, routing or model inference. Video is cut from the continuous
review composite; audio is cut sample-exactly from the verified normalized WAVs.
Boxes are preview annotations only, not final crops or visual admission gates.
"""
import concurrent.futures
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import soundfile as sf
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from p3_listen.common import read, write, sha

ROOT=Path(__file__).resolve().parents[1]
EXP=ROOT/'experiments/p3_listen_full_source_recovery_20261023'
SOURCE=ROOT/'data_previews/seamless_listen_full_source_recovery'
OUT=ROOT/'data_previews/seamless_listen_full_source_clips'


def run(args):
    result=subprocess.run(args,capture_output=True,text=True)
    if result.returncode:
        raise RuntimeError(result.stderr[-4000:])
    return result.stdout


def export(task):
    pair, ordinal, row, view, panels=task
    folder=OUT/pair/row['sample_id']
    folder.mkdir(parents=True,exist_ok=True)
    duration=row['duration_s']
    for role in ('speaker','listener'):
        with sf.SoundFile(row['normalized_audio'][role]['path']) as f:
            assert f.samplerate==48000 and f.channels==1
            f.seek(round(row['start']*f.samplerate))
            wave=f.read(round(duration*f.samplerate),dtype='int32')
        assert len(wave)==round(duration*48000)
        sf.write(folder/(role+'.wav'),wave,48000,subtype='PCM_24')
    filters=['setpts=PTS-STARTPTS']
    for i,c in enumerate(view['crops']):
        if c['status']=='unmeasured':continue
        p=panels[i];sx=p['width']/p['source_width'];sy=p['height']/p['source_height']
        x=p['x']+c['x']*sx;y=p['y']+c['y']*sy
        filters.append(f'drawbox=x={x:.3f}:y={y:.3f}:w={c["size"]*sx:.3f}:h={c["size"]*sy:.3f}:color=0x62c9ff:t=2')
    for i in (0,1):
        role='SPEAKER' if row['speaker_index']==i else 'LISTENER'
        filters.append(f'drawbox=x={720*i}:y=0:w=720:h=32:color=black@0.75:t=fill')
        filters.append(f"drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf:text='{chr(65+i)} - {role}':x={720*i+12}:y=5:fontsize=21:fontcolor=white")
    ff=['ffmpeg','-v','error','-nostdin','-y','-threads','1','-filter_threads','1']
    run(ff+['-ss',str(row['start']),'-i',str(SOURCE/pair/'source.mp4'),'-t',str(duration),
        '-vf',','.join(filters),'-an','-c:v','libx264','-threads','1','-preset','fast','-crf','19',
        '-pix_fmt','yuv420p','-movflags','+faststart',str(folder/'silent.mp4')])
    command=ff+['-i',str(folder/'silent.mp4'),'-i',str(folder/'speaker.wav'),'-i',str(folder/'listener.wav'),
        '-filter_complex_threads','1','-filter_complex',
        '[1:a][2:a]join=inputs=2:channel_layout=stereo:map=0.0-FL|1.0-FR[m]']
    for mode,audio in [('both','[m]'),('speaker','1:a'),('listener','2:a')]:
        command+=['-map','0:v','-map',audio,'-c:v','copy','-c:a','aac','-b:a','192k',
                  '-t',str(duration),'-movflags','+faststart',str(folder/(mode+'.mp4'))]
    run(command)
    checks={}
    for mode,channels in [('both',2),('speaker',1),('listener',1)]:
        file=folder/(mode+'.mp4')
        probe=json.loads(run(['ffprobe','-v','error','-show_streams','-of','json',str(file)]))
        video=next(s for s in probe['streams'] if s['codec_type']=='video')
        audio=next(s for s in probe['streams'] if s['codec_type']=='audio')
        assert (video['width'],video['height'],audio['channels'])==(1440,480,channels)
        assert abs(float(video['duration'])-duration)<=.081
        assert abs(float(audio['duration'])-duration)<=.025
        run(['ffmpeg','-v','error','-xerror','-threads','1','-i',str(file),'-map','0:v','-map','0:a','-f','null','-'])
        checks[mode]=dict(video_seconds=float(video['duration']),audio_seconds=float(audio['duration']),
                          channels=channels,sha256=sha(file),decoded=True)
    (folder/'silent.mp4').unlink()
    exported=dict(view,pair=pair,ordinal=ordinal,preview=f'{pair}/{row["sample_id"]}/',media_checks=checks,
        audio_checks=dict(sample_rate=48000,samples=round(duration*48000),source_hashes_verified=True,
                          exact_interval=True,denoising_applied=False),
        source_kind='continuous review composite with exact normalized source WAV',
        crop_applied=False,crop_boxes='preview annotations only',training_admitted=False)
    write(folder/'clip.json',exported)
    print(json.dumps(dict(pair=pair,clip=ordinal,status=view['pool'],seconds=duration)),flush=True)
    return exported


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    tasks=[];groups=[];hashes={};verified_tracks=set()
    for group in read(SOURCE/'index.json')['pairs']:
        pair=group['pair'];path=EXP/'results'/(pair+'.json');result=read(path);review=read(SOURCE/pair/'review.json')
        assert sha(SOURCE/pair/'source.mp4')==review['media_verification']['sha256']
        hashes[str(path.relative_to(ROOT))]=sha(path)
        views={r['sample_id']:r for r in review['attempts']}
        rows=sorted(result['attempts'],key=lambda r:(r['start'],r['end'],r['sample_id']))
        assert {r['sample_id'] for r in rows}==set(views)
        for number,row in enumerate(rows,1):
            assert all(row[k]==views[row['sample_id']][k] for k in ('start','end','speaker_index','listener_index'))
            for track in row['normalized_audio'].values():
                if (track['path'],track['sha256']) not in verified_tracks:
                    assert sha(Path(track['path']))==track['sha256']
                    verified_tracks.add((track['path'],track['sha256']))
            tasks.append((pair,number,row,views[row['sample_id']],review['panels']))
        groups.append(dict(pair=pair,summary=group['summary'],total=len(rows),zip=pair+'.zip'))
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        clips=list(pool.map(export,tasks))
    write(OUT/'clips.json',dict(groups=groups,clips=clips,input_sha256=hashes,
        note='Existing 33 QC attempts exported without changing cuts or decisions; parent/child attempts may overlap.',
        segmentation_changed=False,audio_policy_changed=False,training_admitted=False))
    shutil.copyfile(ROOT/'scripts/listen_full_source_clips.html',OUT/'index.html')
    from fontTools import subset
    from fontTools.ttLib import TTFont
    font=TTFont('/tmp/v8_browser_libs/extracted/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',fontNumber=2)
    sub=subset.Subsetter();sub.populate(text=(OUT/'index.html').read_text()+(OUT/'clips.json').read_text());sub.subset(font)
    font.flavor='woff';font.save(OUT/'review.woff')
    for group in groups:
        with zipfile.ZipFile(OUT/group['zip'],'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
            for path in sorted((OUT/group['pair']).glob('*/*')):
                z.write(path,path.relative_to(OUT))
            z.writestr('manifest.json',json.dumps(dict(group=group,clips=[r for r in clips if r['pair']==group['pair']]),ensure_ascii=False,indent=2))
        group['zip_bytes']=(OUT/group['zip']).stat().st_size
    link=ROOT/'data_previews/seamless_face_listening_v1/listen_full_source_clips'
    if not link.exists():link.symlink_to(OUT,target_is_directory=True)
    write(OUT/'export_summary.json',dict(groups=groups,clips=len(clips),media_files=len(clips)*3,
        complete=True,input_sha256=hashes,code_sha256={p:sha(ROOT/p) for p in
            ['scripts/export_listen_full_source_clips.py','scripts/listen_full_source_clips.html']}))
    # The review page now defaults to context; publish only after that exists.
    import extend_listen_clip_context
    extend_listen_clip_context.main()
    print('http://127.0.0.1:8875/listen_full_source_clips/',flush=True)


if __name__=='__main__':main()
