"""Standalone, equal-scale before/after waveforms for the selected S0557 pair."""
import json
from pathlib import Path
import sys
import zipfile
import numpy as np
import soundfile as sf

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from build_data_pipeline_selected_cases_20261112 import SVG,ORANGE,INK,GRAY

OUT=ROOT/'writing/figures/data/selected_real_cases_20261112/case2'


def main():
    waves={}
    for name in ['before','clean']:
        wave,sr=sf.read(OUT/f'listener_{name}.wav',dtype='float32')
        assert sr==48000 and len(wave)==310560 and wave.ndim==1
        waves[name]=wave
    scale=float(np.max(np.abs(waves['before'])))
    assert scale>0 and np.max(np.abs(waves['clean']))<=scale
    files=[]
    for name,title in [('before','Listener audio · Before'),('clean','Listener audio · After')]:
        for transparent in [False,True]:
            stem=f'listener_{name}_waveform'+('_transparent' if transparent else '')
            s=SVG(1400,190 if transparent else 300,white=not transparent)
            if not transparent:s.text(35,38,title,28,INK,600)
            s.wave(waves[name],35,15 if transparent else 70,1330,160,ORANGE,scale,bins=3000)
            if not transparent:
                s.axis(35,242,1330,6.47,18)
                s.text(1365,289,'6.470 s · relative to pair start · shared amplitude scale',17,GRAY,anchor='end')
            path=OUT/(stem+'.svg');s.save(path,2)
            files.extend([path,path.with_suffix('.png'),path.with_suffix('.pdf')])
    s=SVG(1400,510)
    s.text(35,38,'S0557 · listener audio before / after',28,INK,600)
    for i,(name,title) in enumerate([('before','Before'),('clean','After')]):
        y=70+i*180;s.text(35,y+60,title,23,INK,600);s.wave(waves[name],150,y,1215,135,ORANGE,scale,bins=3000)
    s.axis(150,435,1215,6.47,18)
    s.text(1365,490,'Same time and amplitude scales · no separate normalization',18,GRAY,anchor='end')
    path=OUT/'listener_waveform_comparison.svg';s.save(path,2)
    files.extend([path,path.with_suffix('.png'),path.with_suffix('.pdf')])
    note=dict(sample_id='naturalistic_V03_S0557_I00000077_local_c4e0cb6f423c',duration_s=6.47,sample_rate=sr,
              waveform='Signed min/max envelope of actual float32 WAV samples',shared_absolute_peak=scale,
              separate_normalization=False,shared_time_axis=True,
              inputs=['listener_before.wav','listener_clean.wav'],
              transparent_versions='Same waveform geometry and amplitude scale, no titles or axes.')
    p=OUT/'waveform_export_notes.json';p.write_text(json.dumps(note,indent=2)+'\n');files.append(p)
    with zipfile.ZipFile(OUT/'listener_before_after_waveforms.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in files:z.write(p,p.name)
    web=ROOT/'data_previews/listen_benchmark_review/data_pipeline_selected2_20261112/index.html'
    html=web.read_text()
    if 'id="separate-waveforms"' not in html:
        section='''<section id="separate-waveforms"><h2>④ S0557 6.47秒：净化前／后独立波形图</h2><p>两张图保持完全相同的时间轴和幅度刻度。提供高清PNG、可编辑SVG、PDF，以及无坐标无文字透明版本。</p><p><a href="assets/case2/listener_before_after_waveforms.zip" download>全部波形素材ZIP</a> · <a href="assets/case2/listener_waveform_comparison.png" download>两行对照PNG</a></p>'''
        for key,label in [('before','净化前'),('clean','净化后')]:
            stem=f'listener_{key}_waveform'
            section+=f'<h3>{label}</h3><img src="assets/case2/{stem}.png"><p><a href="assets/case2/{stem}.png" download>高清PNG</a> · <a href="assets/case2/{stem}.svg" download>可编辑SVG</a> · <a href="assets/case2/{stem}.pdf" download>PDF</a> · <a href="assets/case2/{stem}_transparent.png" download>透明PNG</a> · <a href="assets/case2/{stem}_transparent.svg" download>透明SVG</a></p>'
        web.write_text(html.replace('</html>',section+'</section></html>'))
    print(json.dumps(dict(output=str(OUT),files=len(files),shared_peak=scale),ensure_ascii=False))


if __name__=='__main__':main()
