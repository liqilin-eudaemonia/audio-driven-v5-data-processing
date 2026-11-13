"""Add +/-3 seconds of source context and burned-in cut markers to review clips.

Original slice files, numbering, routing, and scientific results stay intact.
"""
import concurrent.futures
import json
from pathlib import Path
import shutil
import zipfile

import soundfile as sf
import export_listen_full_source_clips as base
from p3_listen.common import read, write, sha

FONT='/tmp/v8_browser_libs/extracted/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'


def text_filter(text,x,y,size=24,color='white',enable=None):
    # Literal timestamps use periods, avoiding FFmpeg drawtext colon escaping.
    value=f"drawtext=fontfile={FONT}:text='{text}':x={x}:y={y}:fontsize={size}:fontcolor={color}"
    return value+(f":enable='{enable}'" if enable else '')


def export(task):
    view,row,source_duration,panels=task
    folder=base.OUT/view['preview']
    a=max(0.,row['start']-3)
    b=min(source_duration,row['end']+3)
    duration=b-a
    left,right=row['start']-a,row['end']-a
    context=dict(start=a,end=b,duration_s=duration,cut_start_offset=left,cut_end_offset=right,
        before_seconds=left,after_seconds=b-row['end'],requested_margin_seconds=3,
        clamped_at_source_start=a==0,clamped_at_source_end=b==source_duration,
        source_timebase=True,markers_burned_in=True)
    for role in ('speaker','listener'):
        with sf.SoundFile(row['normalized_audio'][role]['path']) as f:
            assert f.samplerate==48000 and f.channels==1
            f.seek(round(a*48000));wave=f.read(round(duration*48000),dtype='int32')
        assert len(wave)==round(duration*48000)
        sf.write(folder/('context_'+role+'.wav'),wave,48000,subtype='PCM_24')
    filters=['setpts=PTS-STARTPTS']
    for i,c in enumerate(view['crops']):
        if c['status']=='unmeasured':continue
        p=panels[i];sx=p['width']/p['source_width'];sy=p['height']/p['source_height']
        filters.append(f"drawbox=x={p['x']+c['x']*sx:.3f}:y={p['y']+c['y']*sy:.3f}:w={c['size']*sx:.3f}:h={c['size']*sy:.3f}:color=0x62c9ff:t=2:enable='gte(t,{left})*lt(t,{right})'")
    filters+=['drawbox=x=0:y=0:w=iw:h=35:color=black@0.8:t=fill']
    for i in (0,1):
        role='speaker' if row['speaker_index']==i else 'listener'
        filters.append(text_filter(f'片段角色 {chr(65+i)} - {role}',720*i+12,5,21))
    filters+=['drawbox=x=0:y=402:w=iw:h=78:color=black@0.85:t=fill',
        text_filter('切分前上下文',20,408,color='0xffc46b',enable=f'lt(t,{left})'),
        text_filter('切分范围内',20,408,color='0x59e0a5',enable=f'gte(t,{left})*lt(t,{right})'),
        text_filter('切分后上下文',20,408,color='0xffc46b',enable=f'gte(t,{right})'),
        text_filter(f'原片起点 {row["start"]:.3f} 秒  /  终点 {row["end"]:.3f} 秒',340,410,21),
        text_filter(f'前文 {left:.3f} 秒  |  后文 {b-row["end"]:.3f} 秒',1040,410,20),
        'drawbox=x=20:y=446:w=1400:h=14:color=0x596980:t=fill']
    x0=20+1400*left/duration;x1=20+1400*right/duration
    filters += [f'drawbox=x={x0}:y=446:w={x1-x0}:h=14:color=0x35ba84:t=fill',
        f'drawbox=x={x0}:y=439:w=3:h=26:color=0xffe08c:t=fill',
        f'drawbox=x={x1}:y=439:w=3:h=26:color=0xffe08c:t=fill',
        text_filter('I',f'18+1400*t/{duration}',439,27),
        text_filter('切分起点',640,56,36,'0xffe08c',f'gte(t,{left})*lt(t,{min(right,left+.6)})'),
        text_filter('切分终点',640,56,36,'0xffe08c',f'gte(t,{right})*lt(t,{right+.6})')]
    ff=['ffmpeg','-v','error','-nostdin','-y','-threads','1','-filter_threads','1']
    base.run(ff+['-ss',str(a),'-i',str(base.SOURCE/view['pair']/'source.mp4'),'-t',str(duration),
        '-vf',','.join(filters),'-an','-c:v','libx264','-threads','1','-preset','fast','-crf','19',
        '-pix_fmt','yuv420p','-movflags','+faststart',str(folder/'context_silent.mp4')])
    command=ff+['-i',str(folder/'context_silent.mp4'),'-i',str(folder/'context_speaker.wav'),
        '-i',str(folder/'context_listener.wav'),'-filter_complex_threads','1','-filter_complex',
        '[1:a][2:a]join=inputs=2:channel_layout=stereo:map=0.0-FL|1.0-FR[m]']
    for mode,audio in [('both','[m]'),('speaker','1:a'),('listener','2:a')]:
        command+=['-map','0:v','-map',audio,'-c:v','copy','-c:a','aac','-b:a','192k',
                  '-t',str(duration),'-movflags','+faststart',str(folder/('context_'+mode+'.mp4'))]
    base.run(command)
    checks={}
    for mode,channels in [('both',2),('speaker',1),('listener',1)]:
        file=folder/('context_'+mode+'.mp4')
        probe=json.loads(base.run(['ffprobe','-v','error','-show_streams','-of','json',str(file)]))
        video=next(s for s in probe['streams'] if s['codec_type']=='video')
        audio=next(s for s in probe['streams'] if s['codec_type']=='audio')
        assert (video['width'],video['height'],audio['channels'])==(1440,480,channels)
        assert abs(float(video['duration'])-duration)<=.081
        assert abs(float(audio['duration'])-duration)<=.025
        base.run(['ffmpeg','-v','error','-xerror','-threads','1','-i',str(file),'-map','0:v','-map','0:a','-f','null','-'])
        checks[mode]=dict(video_seconds=float(video['duration']),audio_seconds=float(audio['duration']),
            channels=channels,decoded=True,sha256=sha(file))
    (folder/'context_silent.mp4').unlink()
    context.update(media_checks=checks,audio_samples=round(duration*48000))
    result=dict(view,context=context)
    write(folder/'context.json',context)
    print(json.dumps(dict(pair=view['pair'],clip=view['ordinal'],before=left,after=b-row['end'])),flush=True)
    return result


def main():
    data=read(base.OUT/'clips.json')
    backup=base.OUT/'clips_without_context.json'
    if not backup.exists():shutil.copyfile(base.OUT/'clips.json',backup)
    lookup={};panels={};durations={};verified=set()
    for g in data['groups']:
        pair=g['pair'];result_path=base.EXP/'results'/(pair+'.json')
        assert sha(result_path)==data['input_sha256'][str(result_path.relative_to(base.ROOT))]
        result=read(result_path);review=read(base.SOURCE/pair/'review.json')
        assert sha(base.SOURCE/pair/'source.mp4')==review['media_verification']['sha256']
        lookup.update({r['sample_id']:r for r in result['attempts']})
        panels[pair]=review['panels'];durations[pair]=review['summary']['source_seconds']
    tasks=[]
    for view in data['clips']:
        row=lookup[view['sample_id']]
        assert all(view[k]==row[k] for k in ('start','end','speaker_index','listener_index'))
        for track in row['normalized_audio'].values():
            if track['path'] not in verified:
                assert sha(Path(track['path']))==track['sha256'];verified.add(track['path'])
        tasks.append((view,row,durations[view['pair']],panels[view['pair']]))
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        data['clips']=list(pool.map(export,tasks))
    for group in data['groups']:
        group['context_zip']=group['pair']+'_context.zip'
        with zipfile.ZipFile(base.OUT/group['context_zip'],'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
            for path in sorted((base.OUT/group['pair']).glob('*/context*')):
                z.write(path,path.relative_to(base.OUT))
            z.writestr('manifest.json',json.dumps(dict(group=group,clips=[r for r in data['clips'] if r['pair']==group['pair']]),ensure_ascii=False,indent=2))
    data['context_revision']='three-second-context-with-burned-cut-markers-1'
    write(base.OUT/'clips.json',data)
    shutil.copyfile(base.ROOT/'scripts/listen_full_source_clips.html',base.OUT/'index.html')
    from fontTools import subset
    from fontTools.ttLib import TTFont
    font=TTFont(FONT,fontNumber=2);sub=subset.Subsetter()
    sub.populate(text=(base.OUT/'index.html').read_text()+(base.OUT/'clips.json').read_text());sub.subset(font)
    font.flavor='woff';font.save(base.OUT/'review.woff')
    write(base.OUT/'context_export_summary.json',dict(clips=len(data['clips']),media_files=3*len(data['clips']),
        complete=True,original_exports_unchanged=True,segmentation_changed=False,audio_policy_changed=False,
        baseline_manifest_sha256=sha(backup),input_sha256=data['input_sha256'],
        code_sha256={p:sha(base.ROOT/p) for p in ['scripts/extend_listen_clip_context.py','scripts/listen_full_source_clips.html']}))


if __name__=='__main__':main()
