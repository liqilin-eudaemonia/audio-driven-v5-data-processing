"""Export actual paired tracks for a role-swapped turn and a later overlapping reply."""
import json
from pathlib import Path
import sys
import zipfile
import numpy as np
import soundfile as sf

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from build_data_pipeline_selected_cases_20261112 import SVG,ORANGE,BLUE,INK,GRAY
BASE=ROOT/'writing/figures/data/selected_real_cases_20261112'
OUT=BASE/'case2_role_swap'


def export(name,start,end,title,roles,transcript,sources):
    dest=OUT/name;dest.mkdir(parents=True,exist_ok=True)
    duration=end-start;waves={}
    for person in ['P2254','P2517']:
        with sf.SoundFile(sources[person]) as f:
            assert f.samplerate==48000 and f.channels==1
            first,last=round(start*f.samplerate),round(end*f.samplerate)
            f.seek(first);a=f.read(last-first,dtype='float32')
        assert len(a)==round(duration*48000)
        sf.write(dest/f'{person}.wav',a,48000,subtype='FLOAT');waves[person]=a
    scale=max(float(np.max(np.abs(a))) for a in waves.values())
    s=SVG(1500,540);s.text(35,40,title,27,INK,600)
    s.text(35,76,f'S0557 · source time {start:.3f}–{end:.3f} s · {duration:.3f} s',20,GRAY)
    for i,person in enumerate(['P2254','P2517']):
        y=128+i*164
        s.text(35,y-10,f'{person} · {roles[person]}',22,BLUE if i==0 else INK,600)
        s.wave(waves[person],35,y,1430,116,ORANGE,scale,bins=3200)
        # Independent, transparent strips for the user's existing figure.
        ss=SVG(1500,200,white=False);ss.wave(waves[person],25,15,1450,170,ORANGE,scale,bins=3200)
        ss.save(dest/f'{person}_transparent.svg',2)
        ss=SVG(1500,280);ss.text(35,38,f'{person} · {roles[person]}',26,INK,600)
        ss.wave(waves[person],35,65,1430,135,ORANGE,scale,bins=3200)
        ss.axis(35,220,1430,duration,17)
        ss.text(1465,269,f'Source {start:.3f}–{end:.3f} s · shared scale',17,GRAY,anchor='end')
        ss.save(dest/f'{person}_waveform.svg',2)
    s.axis(35,451,1430,duration,17)
    s.text(35,510,'Seconds from segment start · original independent tracks before pair-level gating · shared scale',18,GRAY)
    s.save(dest/'both_tracks.svg',2)
    record=dict(source_interval_s=[start,end],duration_s=duration,roles=roles,transcript_reference=transcript,
                audio_sources=sources,normalization='Existing whole-track normalization; no extra gain or pair-level silence gating.',
                waveform='Signed min/max envelope from WAV samples, common amplitude scale.',
                chronology='Earlier than selected 233.290–239.760 s pair' if end<233.29 else 'Later than selected 233.290–239.760 s pair; both participants speak, not a clean single-speaker interval.')
    (dest/'provenance.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')


def main():
    OUT.mkdir(exist_ok=True)
    row=json.loads((BASE/'case2/manifest_record.json').read_text());src=row['source']
    clip=json.loads(Path(src['source_result']).read_text())['clips'][src['source_clip_index']]
    sources={'P2254':clip['normalized_audio']['listener']['path'],'P2517':clip['normalized_audio']['speaker']['path']}
    found=[c for c in json.loads(Path(src['source_result']).read_text())['clips'] if abs(c['start']-176.37)<1e-7 and abs(c['end']-185.297)<1e-7]
    assert len(found)==1 and found[0]['speaker_id'].endswith('_P2254') and found[0]['listener_id'].endswith('_P2517')
    export('role_swapped_176370_185297',176.370,185.297,'Role-swapped turn: P2254 speaks, P2517 listens',
           {'P2254':'speaker','P2517':'listener'},found[0]['text'],sources)
    export('later_reply_269053_273876',269.053,273.876,'Later reply: P2254 speaks with overlapping P2517 speech',
           {'P2254':'spoken response','P2517':'overlapping speech'},
           'So Australians who came to vacation in the U.S. were going back home, I guess.',sources)
    (OUT/'README.md').write_text('''# S0557：P2254主讲的波形素材

## 清楚的身份互换片段

`role_swapped_176370_185297/`：源视频176.370–185.297秒，共8.927秒。已有J切分记录明确为P2254 speaker、P2517 listener。它在原选233.290–239.760秒案例**之前**，不是之后。

包含双方对齐波形`both_tracks.svg/pdf/png`，各自独立波形和透明背景版本，以及两条无损48kHz浮点WAV。均取真实官方独立音轨的已有整轨归一化版本，**未新增静音门控**。图中共用幅度尺度，不做独立拉高。

## 原案例之后的接话

`later_reply_269053_273876/`：源视频269.053–273.876秒，共4.823秒。官方转写中P2254说“So Australians who came to vacation in the U.S. were going back home, I guess.”；P2517同时仍在说话，因此标作重叠接话，不冒充纯speaker/listener互换片段。

每个目录提供PNG、可编辑SVG、PDF、透明波形和独立WAV；`provenance.json`记录来源及时间。
''')
    page=ROOT/'data_previews/listen_benchmark_review/data_pipeline_selected2_20261112/index.html'
    html=page.read_text()
    if 'id="role-swap"' not in html:
        section='''<section id="role-swap"><h2>⑤ P2254变为speaker：角色互换波形</h2><p><a href="assets/case2_role_swap/role_swap_waveforms.zip" download>下载全部波形素材ZIP</a></p><p><strong>清楚的身份互换：原视频176.370–185.297秒（8.927秒）。</strong>P2254主讲、P2517倾听。注意它在原选6.47秒案例之前。</p>'''
        for dirname in ['role_swapped_176370_185297','later_reply_269053_273876']:
            if dirname.startswith('later'):section+='<h3>原案例之后的接话（双方声音重叠）</h3><p>原视频269.053–273.876秒，P2254接话时P2517仍在讲话；不作为清楚的单人主讲互换案例。</p>'
            b='assets/case2_role_swap/'+dirname+'/'
            section+=f'<img src="{b}both_tracks.png"><p><a href="{b}both_tracks.png" download>双方波形PNG</a> · <a href="{b}both_tracks.svg" download>可编辑SVG</a></p>'
            for person in ['P2254','P2517']:
                section+=f'<p>{person}：<a href="{b}{person}_waveform.png" download>独立波形PNG</a> · <a href="{b}{person}_transparent.png" download>透明PNG</a> · <a href="{b}{person}_transparent.svg" download>透明SVG</a> · <a href="{b}{person}.wav" download>WAV</a></p>'
        page.write_text(html.replace('</html>',section+'</section></html>'))
    with zipfile.ZipFile(OUT/'role_swap_waveforms.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in OUT.rglob('*'):
            if p.is_file() and p.suffix!='.zip':z.write(p,p.relative_to(OUT))
    print(str(OUT))


if __name__=='__main__':main()
