"""Export the user-selected S0557 surrounding minute for manual figure editing."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import numpy as np
import soundfile as sf

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from build_data_pipeline_selected_cases_20261112 import SVG,BLUE,ORANGE,INK,GRAY

BASE=ROOT/'writing/figures/data/selected_real_cases_20261112'
OUT=BASE/'case2_context60s'
START,END=206.525,266.525
DURATION=END-START
SELECTED=[233.290-START,239.760-START]
FFMPEG='/usr/bin/ffmpeg'
FFPROBE='/usr/bin/ffprobe'


def probe(path):
    r=subprocess.run([FFPROBE,'-v','error','-show_streams','-show_format','-of','json',str(path)],capture_output=True,text=True,check=True)
    return json.loads(r.stdout)


def uri(a):
    return f"subfile,,start,{a['offset']},end,{a['offset']+a['byte_size']},,:{a['archive']}"


def encode(name,source,audio):
    dest=OUT/f'{name}_original_view_60s.mp4'
    if dest.exists():
        try:
            p=probe(dest)
            if abs(float(p['format']['duration'])-DURATION)<.08:return p
        except (subprocess.SubprocessError,KeyError):pass
    tmp=dest.with_name(dest.stem+'.partial.mp4')
    command=[FFMPEG,'-y','-hide_banner','-loglevel','warning','-threads','3','-ss',str(START),'-i',uri(source),
             '-ss',str(START),'-i',audio,'-t',str(DURATION),'-map','0:v:0','-map','1:a:0',
             '-c:v','libx264','-preset','veryfast','-crf','18','-threads','6',
             '-pix_fmt','yuv420p','-c:a','aac','-b:a','192k','-movflags','+faststart',str(tmp)]
    with (OUT/f'{name}_encode.log').open('w') as log:
        subprocess.run(command,stdout=log,stderr=log,check=True)
    tmp.replace(dest)
    p=probe(dest)
    v=next(s for s in p['streams'] if s['codec_type']=='video')
    assert [v['width'],v['height']]==[2160,3840]
    assert abs(float(p['format']['duration'])-DURATION)<.08
    print(name,'video done',flush=True)
    return p


def waveform(waves,highlight):
    s=SVG(1800,520)
    s.text(35,40,'S0557 · surrounding 60 seconds · original participant tracks',27,INK,600)
    s.text(35,74,'Source video: 206.525–266.525 s  |  highlighted pair: 233.290–239.760 s',20,GRAY)
    x,w=65,1670
    if highlight:
        s.rect(x+SELECTED[0]/DURATION*w,106,(SELECTED[1]-SELECTED[0])/DURATION*w,306,'#edf4fd','none',0)
        for t in SELECTED:s.line(x+t/DURATION*w,100,x+t/DURATION*w,420,BLUE,1.5,dash='6 6')
    # A common scale preserves the relative amplitude of the existing tracks.
    scale=max(float(np.max(np.abs(v))) for v in waves.values())
    for i,(name,who) in enumerate([('speaker','A / P2517 · speaker in the selected pair'),('listener','B / P2254 · listener in the selected pair')]):
        y=135+i*155
        s.text(x,y-10,who,22,BLUE if i==0 else ORANGE,600)
        s.wave(waves[name],x,y,w,105,ORANGE,scale,bins=5200)
    s.axis(x,435,w,DURATION,19,ticks=list(range(0,61,10)))
    s.text(35,502,'Seconds from the start of this minute. Roles can change outside the selected pair. Same amplitude scale.',19,GRAY)
    s.save(OUT/('both_tracks_highlighted.svg' if highlight else 'both_tracks_plain.svg'),1.5)
    for name,wave in waves.items():
        if highlight:continue
        s=SVG(1800,180,white=False);s.wave(wave,20,10,1760,160,ORANGE,scale,bins=5200)
        s.save(OUT/f'{name}_waveform_transparent.svg',1.5)


def main():
    OUT.mkdir(exist_ok=True)
    row=json.loads((BASE/'case2/manifest_record.json').read_text());src=row['source']
    pair=json.loads(Path(src['source_result']).read_text());clip=pair['clips'][src['source_clip_index']]
    assert clip['sample_id']==row['sample_id']
    waves={};audio={}
    for name in ['speaker','listener']:
        a=clip['normalized_audio'][name];audio[name]=a['path']
        with sf.SoundFile(a['path']) as f:
            assert f.samplerate==48000 and f.channels==1
            first,last=round(START*f.samplerate),round(END*f.samplerate)
            assert last<=len(f);f.seek(first);wave=f.read(last-first,dtype='float32')
        assert len(wave)==2880000
        sf.write(OUT/f'{name}_60s.wav',wave,48000,subtype='FLOAT');waves[name]=wave
    waveform(waves,True);waveform(waves,False)
    with (OUT/'selected_pair_times.csv').open('w') as f:
        f.write('sample_id,source_start_s,source_end_s,context_start_s,context_end_s,relative_start_s,relative_end_s\n')
        f.write(f"{row['sample_id']},233.290,239.760,{START},{END},{SELECTED[0]:.3f},{SELECTED[1]:.3f}\n")
    print('waveforms and lossless WAVs ready; encoding two full-resolution videos',flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures={name:pool.submit(encode,name,clip['sources'][name],audio[name]) for name in audio}
        probes={name:f.result() for name,f in futures.items()}
    # Convenient side-by-side preview; full-resolution individual views remain separate.
    combo=OUT/'both_views_60s_preview.mp4'
    if not combo.exists():
        cmd=[FFMPEG,'-y','-hide_banner','-loglevel','warning','-threads','2','-i',str(OUT/'speaker_original_view_60s.mp4'),
             '-threads','2','-i',str(OUT/'listener_original_view_60s.mp4'),'-filter_complex_threads','2',
             '-filter_complex','[0:v]scale=540:960,setsar=1[a];[1:v]scale=540:960,setsar=1[b];[a][b]hstack=inputs=2[v];[0:a][1:a]amerge=inputs=2[au]',
             '-map','[v]','-map','[au]','-t','60','-c:v','libx264','-crf','20','-preset','veryfast','-threads','4',
             '-c:a','aac','-b:a','256k','-movflags','+faststart',str(combo)]
        with (OUT/'combined_encode.log').open('w') as f:subprocess.run(cmd,stdout=f,stderr=f,check=True)
    cp=probe(combo);assert abs(float(cp['format']['duration'])-60)<.08
    manifest=dict(sample_id=row['sample_id'],source_interval_s=[START,END],duration_s=DURATION,
                  selected_source_interval_s=[233.29,239.76],selected_relative_interval_s=SELECTED,
                  roles='A/P2517 and B/P2254 are speaker/listener in the selected 6.47s pair only; roles may switch elsewhere.',
                  video_sources=clip['sources'],audio_sources={k:clip['normalized_audio'][k] for k in audio},
                  audio='Existing independent participant tracks with whole-track normalization, BEFORE pair-level official-union silence gating. No mixture separation or context-wide inferred gating.',
                  video='Uncropped original views, native 2160x3840 and source frame rate; reencoded H264 CRF18 with corresponding independent participant AAC audio. Preview: A left, B right; stereo A left channel, B right channel.',
                  timing='Audio sampled exactly at requested bounds. Video uses original frame grid; boundary quantization is less than one source frame.',
                  probes=probes,combined_probe=cp)
    (OUT/'provenance.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    (OUT/'README.md').write_text('''# S0557 附近一分钟素材

源区间 **206.525–266.525 秒**，对应原视频03:26.525–04:26.525。原选pair在本分钟的 **26.765–33.235秒**。

- `both_tracks_highlighted.svg/pdf/png`：双方真实波形，同一幅度尺度；蓝色标出已选6.47秒片段。
- `both_tracks_plain.svg/pdf/png`：无蓝色高亮；另有每条音轨的透明背景SVG/PDF/PNG，方便自行剪图。
- `speaker_60s.wav`、`listener_60s.wav`：48kHz浮点无损独立音轨（沿用整轨归一化），**门控静音前**。一分钟内角色可能变化，未把原pair的listener静音规则延伸到整分钟。
- `speaker_original_view_60s.mp4`、`listener_original_view_60s.mp4`：双方分别的原始完整画面，2160×3840、原帧率，未套用6.47秒的固定人脸框。H264 CRF18重编码，保留各自独立声音。
- `both_views_60s_preview.mp4`：左右并排预览。画面左A/P2517、右B/P2254；声音左声道A、右声道B。个体原尺寸视频用于正式剪辑，并排版用于浏览。
- `selected_pair_times.csv`：源视频与这一分钟内的位置对应；`provenance.json`保存来源及媒体参数。

A是原选pair的speaker、B是原选pair的listener；一分钟内不要把这两个角色视为始终不变。原来6.47秒的净化视频、最终listener裁框、caption和Event素材仍在上一级case2，未作更改。
''')
    # Link from the existing figure page without rewriting prior assets or the old ZIP.
    web=ROOT/'data_previews/listen_benchmark_review/data_pipeline_selected2_20261112'
    page=web/'index.html';html=page.read_text()
    marker='<section id="context60s">'
    if marker not in html:
        section='''<section id="context60s"><h2>③ S0557 附近一分钟：供自行剪裁</h2><p>原视频 206.525–266.525 秒。已选pair位于本分钟的 <strong>26.765–33.235 秒</strong>。左侧A/P2517，右侧B/P2254；角色名称仅适用于已选pair。</p><p><a href="assets/case2_context60s/context60s_bundle.zip" download>下载一分钟全部素材ZIP</a> · <a href="assets/case2_context60s/both_tracks_highlighted.svg" download>波形SVG（标记pair）</a> · <a href="assets/case2_context60s/both_tracks_plain.svg" download>波形SVG（无高亮）</a> · <a href="assets/case2_context60s/both_tracks_highlighted.png" download>高清PNG</a></p><img src="assets/case2_context60s/both_tracks_highlighted.png"><video controls preload="metadata" style="width:700px" src="assets/case2_context60s/both_views_60s_preview.mp4"></video><p><a href="assets/case2_context60s/both_views_60s_preview.mp4" download>下载并排预览</a> · <a href="assets/case2_context60s/speaker_original_view_60s.mp4" download>下载A原尺寸视频</a> · <a href="assets/case2_context60s/listener_original_view_60s.mp4" download>下载B原尺寸视频</a> · <a href="assets/case2_context60s/speaker_60s.wav" download>A独立WAV</a> · <a href="assets/case2_context60s/listener_60s.wav" download>B独立WAV</a></p><p>此一分钟保留原有声音，未延伸pair的静音门控；与原来净化后的6.47秒训练素材分开提供。<a href="assets/case2_context60s/README.md">详细说明</a></p></section>'''
        html=html.replace('</html>',section+'</html>');page.write_text(html)
    shutil.copyfile(__file__,OUT/'export_script_snapshot.py')
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir() if p.is_file() and p.suffix not in ['.zip','.log'] and p.name!='FILES_SHA256.json'}
    (OUT/'FILES_SHA256.json').write_text(json.dumps(hashes,indent=2)+'\n')
    archive=OUT/'context60s_bundle.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for p in OUT.iterdir():
            if p.is_file() and p.suffix not in ['.zip','.log']:z.write(p,p.name)
    print(json.dumps(dict(output=str(OUT),zip_mb=archive.stat().st_size/1e6,url='http://localhost:8876/listen_benchmark_review/data_pipeline_selected2_20261112/#context60s'),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
