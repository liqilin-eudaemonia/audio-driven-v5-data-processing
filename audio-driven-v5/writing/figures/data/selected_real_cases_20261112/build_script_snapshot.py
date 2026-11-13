"""Export two user-selected real cases as editable, data-faithful paper assets.

CPU only. Read the sealed 20-case exports and two indexed original video members;
do not change training data, annotations, crop policy, or detector results.
"""
import base64
import csv
import hashlib
import json
import shutil
import sys
import textwrap
from pathlib import Path
from xml.sax.saxutils import escape

import av
import cairosvg
import numpy as np
import soundfile as sf
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from listen_benchmark.frame_crop import crop_original_frame_reused
from listen_benchmark.official_audio_gate import make_gate
from listen_visual_quality.measurement import SourceBytes

OLD = ROOT / 'data_previews/listen_benchmark_review/data_pipeline_cases20_20261112/media'
OUT = ROOT / 'writing/figures/data/selected_real_cases_20261112'
WEB = ROOT / 'data_previews/listen_benchmark_review/data_pipeline_selected2_20261112'
IDS = ['naturalistic_V03_S1371_I00000205_local_08ad393f3e19',
       'naturalistic_V03_S0557_I00000077_local_c4e0cb6f423c']
BLUE, ORANGE, PURPLE = '#2784bd', '#dc9200', '#8060bd'
INK, GRAY, LINE = '#3d4351', '#697386', '#d1d8e3'


def read(p):
    return json.loads(Path(p).read_text())


def write(p, value):
    Path(p).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


class SVG:
    def __init__(self, width, height, white=True):
        self.w, self.h = width, height
        self.parts = [f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
                      '<defs><marker id="arrow" markerWidth="9" markerHeight="9" refX="8" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8" fill="#697386"/></marker></defs>']
        if white:
            self.rect(0, 0, width, height, '#fff', 'none', 0)

    def rect(self, x, y, w, h, fill='#fff', stroke=LINE, radius=8, **attrs):
        extra = ' '.join(f'{k.replace("_", "-")}="{v}"' for k, v in attrs.items())
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{fill}" stroke="{stroke}" {extra}/>')

    def text(self, x, y, value, size=19, color=INK, weight=400, anchor='start'):
        self.parts.append(f'<text x="{x}" y="{y}" font-family="DejaVu Sans,Arial,sans-serif" font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" fill="{color}">{escape(str(value))}</text>')

    def paragraph(self, x, y, value, width, size=19, color=INK, step=None, weight=400):
        step = step or size * 1.42
        # Conservative wrapping: leave margin for longer/wider glyphs.
        lines = textwrap.wrap(value, max(10, int(width / (size * .56))))
        for i, line in enumerate(lines):
            self.text(x, y + i * step, line, size, color, weight)
        return y + len(lines) * step

    def line(self, x1, y1, x2, y2, color=LINE, width=1.2, arrow=False, dash=None):
        tail = ' marker-end="url(#arrow)"' if arrow else ''
        if dash:
            tail += f' stroke-dasharray="{dash}"'
        self.parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{width}"{tail}/>')

    def image(self, path, x, y, w, h):
        mime = 'image/jpeg' if Path(path).suffix == '.jpg' else 'image/png'
        encoded = base64.b64encode(Path(path).read_bytes()).decode()
        self.parts.append(f'<image x="{x}" y="{y}" width="{w}" height="{h}" preserveAspectRatio="xMidYMid meet" xlink:href="data:{mime};base64,{encoded}"/>')

    def wave(self, wave, x, y, w, h, color=ORANGE, scale=None, bins=None):
        """Signed min/max envelope; every polygon ordinate comes from real PCM."""
        a = np.asarray(wave).reshape(-1)
        scale = max(float(np.max(np.abs(a))), 1e-8) if scale is None else max(scale, 1e-8)
        count = min(len(a), bins or max(150, round(w * 2)))
        edges = np.linspace(0, len(a), count + 1).astype(int)
        center = y + h / 2
        self.line(x, center, x + w, center, color, .65)
        low = [float(a[edges[i]:edges[i+1]].min()) for i in range(count)]
        high = [float(a[edges[i]:edges[i+1]].max()) for i in range(count)]
        xx = np.linspace(x, x + w, count)
        points = [(u, center - v / scale * h * .46) for u, v in zip(xx, high)]
        points += [(u, center - v / scale * h * .46) for u, v in zip(xx[::-1], low[::-1])]
        self.parts.append('<polygon fill="' + color + '" points="' + ' '.join(f'{u:.2f},{v:.2f}' for u, v in points) + '"/>')

    def intervals(self, intervals, duration, x, y, w, h, color=BLUE, fill='#e6f2fb'):
        for a, b in intervals:
            assert 0 <= a < b <= duration + 1e-7
            self.rect(x + a / duration * w, y, (b-a) / duration * w, h, fill, color, 0)

    def axis(self, x, y, w, duration, size=14, start=0, ticks=None):
        self.line(x, y, x + w, y, GRAY)
        values = ticks if ticks is not None else list(range(int(duration)+1))
        for t in values:
            xx = x + t / duration * w
            self.line(xx, y, xx, y + 4, GRAY)
            self.text(xx, y + size + 7, f'{start+t:g}', size, GRAY, anchor='middle')

    def save(self, path, scale=1.5):
        path = Path(path)
        path.write_text('\n'.join(self.parts) + '\n</svg>\n')
        cairosvg.svg2png(url=str(path), write_to=str(path.with_suffix('.png')), scale=scale)
        cairosvg.svg2pdf(url=str(path), write_to=str(path.with_suffix('.pdf')))


def native_frames(row, dest, times):
    src = row['source']
    start = src['source_start_s']
    results, i, previous, previous_t = [], 0, None, None
    with SourceBytes(src['source']) as member, av.open(member) as container:
        stream = container.streams.video[0]
        stream.thread_type = 'SLICE'
        stream.codec_context.thread_count = 1
        container.seek(int(max(0, start-1) / float(stream.time_base)), stream=stream, backward=True, any_frame=False)
        for frame in container.decode(video=0):
            if frame.pts is None:
                continue
            t = float(frame.pts * frame.time_base)
            while i < len(times) and t >= start + times[i]:
                target = start + times[i]
                chosen = previous if previous is not None and abs(previous_t-target) <= abs(t-target) else frame
                pts = float(chosen.pts * chosen.time_base)
                name = f'keyframe_{i+1:02d}.png'
                crop_original_frame_reused(chosen, src['video_transform']).save(dest/name)
                if i == 0:
                    chosen.to_image().save(dest/'original_frame.png')
                results.append(dict(file=name, requested_pair_time_s=times[i], source_pts_s=pts, pair_time_s=pts-start))
                i += 1
            if i == len(times):
                break
            previous, previous_t = frame, t
    assert len(results) == len(times)
    # It must be the same native RGB reference frame used by the review exporter.
    assert np.array_equal(np.array(Image.open(dest/'keyframe_01.png')), np.array(Image.open(dest/'listener_first_frame.png')))
    return results


def standalone_wave(dest, name, wave, duration):
    for transparent in [False, True]:
        s = SVG(1200, 220 if not transparent else 150, white=not transparent)
        if not transparent:
            s.text(35, 32, 'Speaker audio', 24, INK, 600)
        s.wave(wave, 35, 48 if not transparent else 10, 1130, 115)
        if not transparent:
            s.axis(35, 171, 1130, duration, 17)
            s.text(1165, 210, f'{duration:.3f} s  ·  relative to pair start', 15, GRAY, anchor='end')
        s.save(dest/(name + ('_transparent' if transparent else '') + '.svg'))


def export_case(n, sid):
    src = OLD/sid
    dest = OUT/f'case{n}'
    dest.mkdir(parents=True, exist_ok=True)
    item = read(src/'sample.json')
    row = read(src/'manifest_record.json')
    assert item['id'] == sid and row['sample_id'] == sid
    assert item['in_final_ltx500h'] and item['in_qwen_sft_train']
    assert item['final_crop'] == row['source']['video_transform']['crop']
    for old, new in [('reference.png', 'listener_first_frame.png'), ('speaker.wav','speaker.wav'),
                     ('listener.wav','listener_clean.wav'), ('original_listener.wav','listener_before.wav'),
                     ('listener.mp4','listener_clean_preview.mp4'), ('caption_original.json','caption_original.json'),
                     ('event_targets.json','event_targets.json'), ('audio_recipe.json','audio_recipe.json'),
                     ('crop_provenance.json','crop_provenance.json'), ('manifest_record.json','manifest_record.json')]:
        shutil.copyfile(src/old, dest/new)
    waves = {}
    for key, name in [('speaker','speaker'), ('clean','listener_clean'), ('before','listener_before')]:
        wave, sr = sf.read(dest/(name+'.wav'), dtype='float32')
        assert sr == 48000 and wave.ndim == 1 and len(wave) == round(item['duration'] * sr)
        waves[key] = wave
    gate, info = make_gate(row['audio_recipe']['official_listener'], len(waves['before']), sr)
    assert np.array_equal(waves['clean'], waves['before']*gate)
    write(dest/'gate_exact.json', info)
    (dest/'caption_verbatim.txt').write_text(item['caption']+'\n')
    standalone_wave(dest, 'speaker_waveform', waves['speaker'], item['duration'])
    if n == 2:
        item['figure_frames'] = native_frames(row, dest, [0, 3.55, 3.85, 5.35, 6.20])
        im = Image.open(dest/'original_frame.png').convert('RGB')
        c = item['final_crop']; painter = ImageDraw.Draw(im)
        painter.rectangle([c['x'],c['y'],c['x']+c['width'],c['y']+c['height']], outline=BLUE, width=12)
        im.save(dest/'original_frame_with_final_crop.png')
        # Small embedded preview is an exact aspect-preserving resize; full source PNG is separate.
        im.thumbnail((675,1200), Image.Resampling.LANCZOS)
        im.save(dest/'original_frame_with_final_crop_preview.png')
        result = read(row['source']['source_result'])
        clip = result['clips'][row['source']['source_clip_index']]
        assert clip['sample_id'] == sid
        cs, ce = item['source_start_s'] - 2, 242.18
        assert ce > item['source_end_s'] and 0 <= cs
        context = {}
        for key, path in [('speaker',clip['normalized_audio']['speaker']['path']), ('listener',row['audio_recipe']['source']['path'])]:
            with sf.SoundFile(path) as f:
                assert f.samplerate == sr
                f.seek(round(cs*sr))
                context[key] = f.read(round(ce*sr)-round(cs*sr),dtype='float32')
        item['figure_context'] = dict(source_start_s=cs,source_end_s=ce,
            selected_relative=[item['source_start_s']-cs,item['source_end_s']-cs],
            source_files=dict(speaker=clip['normalized_audio']['speaker']['path'],listener=row['audio_recipe']['source']['path']))
        item['_context'] = context
    item['_waves'] = waves
    item['_gate'] = info
    return item


def before_after(s, item, x, y, w, h, labels=True, labelw=68):
    waves, duration = item['_waves'], item['duration']
    scale = float(np.max(np.abs(waves['before'])))
    labelw = labelw if labels else 0
    xx, ww = x + labelw, w - labelw
    keep = [[a/48000,b/48000] for a,b in item['_gate']['keep_intervals_samples']]
    for k, key in enumerate(['before','clean']):
        yy = y + k * h * .48
        s.intervals(keep,duration,xx,yy,ww,h*.35,ORANGE,'#fff3d5')
        if labels:
            s.text(x,yy+h*.22,'Before' if k == 0 else 'After',16)
        s.wave(waves[key],xx,yy,ww,h*.35,ORANGE,scale)


def cue_plot(s,item,x,y,w,rowh=34,labelw=140,axis=True):
    duration=item['duration'];xx=x+labelw;ww=w-labelw
    lanes=[('Raised brows',[[q['start'],q['end']] for q in item['reaction_cues']],BLUE,'#e4f0fb'),
           ('Nod candidates',item['nod']['candidate_intervals'],BLUE,'#e4f0fb'),
           ('Word: “Right?”',[[q['start'],q['end']] for q in item['vocal_cues']],ORANGE,'#fff2cc')]
    for k,(name,ranges,col,fill) in enumerate(lanes):
        yy=y+k*rowh;s.text(x,yy+16,name,16,col);s.line(xx,yy+8,xx+ww,yy+8)
        s.intervals(ranges,duration,xx,yy,ww,19,col,fill)
    if axis:s.axis(xx,y+rowh*3,ww,duration,14)


def real_event_rows(item):
    lookup={r['category']:r for r in item['event_targets']}
    keys=[('head nodding','Nod'),('smiling','Smile'),('eyebrow raising','Raised brows'),
          ('head tilting','Head tilt'),('speech','Speech'),('frowning','Frown'),('gaze shifting','Gaze shift')]
    return [(label,lookup[('audio:' if key=='speech' else 'visual:')+key]) for key,label in keys]


def event_table(s,item,x,y,w,rowh=37,font=17):
    widths=[w*.47,w*.15,w*.15,w*.23];headers=['Behavior','y','m','weight']
    for k,(name,row) in enumerate([(None,None)]+real_event_rows(item)):
        xx=x
        values=headers if k==0 else [name,str(int(row['y'])) if row['known'] else '—',str(int(row['known'])),f'{row["effective_weight"]:g}']
        fill='#f3f0f9' if k==0 else '#e8f4ee' if row['y']==1 else '#f5f6f9'
        for j,(text,width) in enumerate(zip(values,widths)):
            s.rect(xx,y+k*rowh,width,rowh,fill,LINE,0)
            s.text(xx+10,y+k*rowh+rowh*.68,text,font,PURPLE if k==0 else INK,600 if k==0 else 400)
            xx+=width


def component_assets(item):
    dest=OUT/'case2';d=item['duration']
    s=SVG(1100,480);s.text(28,36,'Listener audio: official words ∪ sentences ∪ VAD',25,INK,600)
    before_after(s,item,28,65,1040,190,labelw=142)
    xx,w=170,898
    for k,key in enumerate(['words','sentences','vad']):
        yy=280+k*40;s.text(28,yy+19,key.capitalize(),18)
        rr=[[max(0,r['start']),min(d,r['end'])] for r in item['official_listener'][key] if r['start']<d and r['end']>0]
        s.line(xx,yy+9,xx+w,yy+9)
        s.intervals(rr,d,xx,yy,w,20,ORANGE,'#fff2cc')
    s.axis(xx,408,w,d,16)
    s.text(28,462,'Same amplitude scale. ±100 ms margins; 10 ms fades; clip duration preserved.',18,GRAY)
    s.save(dest/'listener_before_after.svg')
    s=SVG(1100,250);s.text(28,36,'Auxiliary cues · seconds relative to this pair',25,INK,600)
    cue_plot(s,item,28,70,1040,rowh=39,labelw=180)
    s.text(28,237,'Candidate intervals are hints, not ground-truth temporal labels or exact repetition counts.',16,GRAY)
    s.save(dest/'cue_timeline.svg')
    s=SVG(1500,375)
    for i,f in enumerate(item['figure_frames']):
        x=25+i*298;s.image(dest/f['file'],x,25,270,270)
        s.text(x+135,330,f'{max(0,f["pair_time_s"]):.3f} s',23,BLUE,anchor='middle')
    s.text(25,366,'Actual nearest source frames · final listener crop · 512 × 512',20,GRAY)
    s.save(dest/'keyframe_strip.svg')
    s=SVG(690,400);s.text(20,33,'Actual clip-level Event targets (selected classes)',23,INK,600)
    event_table(s,item,20,56,650,36,18)
    s.text(20,379,'m=0 → ignored; weak negative weight=0.1; repeats share one label.',16,GRAY)
    s.save(dest/'event_labels.svg')
    with (dest/'event_targets_all19.csv').open('w') as f:
        fields=['category','y','known','weight','class_enabled','effective_weight'];writer=csv.DictWriter(f,fields);writer.writeheader();writer.writerows(item['event_targets'])
    cues=dict(sample_id=item['id'],time_origin='pair start',duration_s=d,nod_candidates=item['nod']['candidate_intervals'],
              reaction_cues=item['reaction_cues'],vocal_cues=item['vocal_cues'],official_listener=item['official_listener'],
              note='All cues copied, not inferred from these still frames. Candidate interval count is not exact action count.')
    write(dest/'cues_exact.json',cues)


def pipeline(item):
    """Layout follows the user sketch; every visual observation is a real asset."""
    dest=OUT/'case2';d=item['duration'];s=SVG(1800,1175)
    s.rect(15,15,1770,430,'#fff',LINE,12)
    s.text(36,53,'(a) Role-aware pair construction',31,INK,700)
    # Actual source-track context; no fictional role reversal.
    s.rect(35,75,455,345,'#fafbfd',LINE)
    s.text(52,105,'Paired participant tracks',23,INK,600)
    c=item['figure_context'];duration=c['source_end_s']-c['source_start_s'];sel=c['selected_relative']
    xx,ww=65,405
    s.rect(xx+sel[0]/duration*ww,136,(sel[1]-sel[0])/duration*ww,199,'#eaf4fc','none',0)
    for i,key in enumerate(['speaker','listener']):
        y=142+i*100;s.text(54,y-8, 'A · speaker' if i==0 else 'B · listener (before gating)',17,BLUE if i==0 else ORANGE)
        s.wave(item['_context'][key],xx,y,ww,66,ORANGE)
    for t in sel:
        at=xx+t/duration*ww;s.line(at,125,at,343,BLUE,1.1,dash='5 5')
    s.axis(xx,345,ww,duration,13,start=c['source_start_s'],ticks=[0,2,5,8,duration])
    s.text(260,397,f'Selected pair: {d:.2f} s · sentence-aware',17,GRAY,anchor='middle')
    s.line(495,247,523,247,GRAY,2,True)
    # Crop and official gate, both real.
    s.rect(535,75,668,158,'#fff',BLUE)
    s.text(550,101,'Quality screening + final listener crop',22,BLUE,600)
    s.image(dest/'original_frame_with_final_crop_preview.png',562,112,64,113)
    s.text(644,150,'Original',15,GRAY);s.text(644,174,'2160 × 3840',15,GRAY)
    s.line(772,169,819,169,GRAY,2,True)
    s.image(dest/'listener_first_frame.png',835,111,114,114)
    s.text(969,153,'512 × 512',19,BLUE,600)
    s.text(969,180,f'Face p10: {item["face_p10"]:.0f} px',16,GRAY)
    s.rect(535,245,668,175,'#fff',ORANGE)
    s.text(550,272,'Retain official vocal regions; zero elsewhere',21,INK,600)
    before_after(s,item,550,286,630,95)
    s.text(550,405,'Words ∪ sentences ∪ VAD · ±100 ms · 10 ms fades',16,GRAY)
    s.line(1209,247,1238,247,GRAY,2,True)
    # Current paired training data.
    s.rect(1250,75,510,345,'#fff',LINE)
    s.text(1505,106,'Paired training sample',24,INK,600,anchor='middle')
    s.text(1268,136,'Input: speaker audio + listener reference',19,INK,600)
    s.wave(item['_waves']['speaker'],1275,150,300,68)
    s.text(1600,190,'+',24,INK,600)
    s.image(dest/'listener_first_frame.png',1650,149,66,66)
    s.text(1268,249,'Target: listener video + purified audio',19,INK,600)
    for i,f in enumerate(item['figure_frames']):s.image(dest/f['file'],1274+i*91,264,82,82)
    s.wave(item['_waves']['clean'],1274,353,451,30)
    s.text(1268,407,'Source IDs + original interval + crop + audio recipe',15,PURPLE)
    # Lower panels.
    s.line(1506,445,1506,467,GRAY,2)
    s.line(1506,467,55,467,GRAY,2)
    s.line(55,467,55,495,GRAY,2,True)
    s.rect(15,495,1042,636,'#fff',LINE,12)
    s.rect(1073,495,712,636,'#fff',LINE,12)
    s.text(36,534,'(b) Cue-assisted reaction description',29,INK,700)
    s.text(1094,534,'(c) Clip-level behavior labels',29,INK,700)
    # Panel B: 5 true frames + clean audio -> Omni -> original caption excerpt.
    s.rect(35,551,555,269,'#fafbfd',LINE)
    s.text(50,579,'Listener video + purified listener audio',20,INK,600)
    for i,f in enumerate(item['figure_frames']):
        x=49+i*106;s.image(dest/f['file'],x,597,97,97)
        s.text(x+48.5,718,f'{max(0,f["pair_time_s"]):.2f} s',15,BLUE,anchor='middle')
    s.wave(item['_waves']['clean'],54,738,516,47)
    s.text(50,805,'Frames shown for illustration; model receives video.',15,GRAY)
    s.rect(35,834,555,265,'#fff',LINE)
    s.text(50,862,'Detected cues: categories + intervals',21,INK,600)
    cue_plot(s,item,50,886,515,rowh=39,labelw=147)
    s.paragraph(50,1054,'Auxiliary hints; no exact repetition count or temporal ground truth implied.',507,16,GRAY,22)
    s.line(596,684,601,684,GRAY,2)
    s.line(596,961,601,961,GRAY,2)
    s.line(601,684,601,961,GRAY,2)
    s.line(601,824,610,824,GRAY,2,True)
    s.rect(613,765,86,116,'#f1ecfb',PURPLE)
    s.text(656,809,'Qwen3-',18,PURPLE,600,anchor='middle')
    s.text(656,839,'Omni',18,PURPLE,600,anchor='middle')
    s.line(702,824,715,824,GRAY,2,True)
    s.rect(718,551,318,548,'#f7f4fb',PURPLE)
    s.text(734,579,'Original caption · verbatim',19,PURPLE,600)
    end=s.paragraph(734,615,item['caption'],282,17,INK,25)
    s.paragraph(734,end+30,'Order + sustained states + simultaneous actions',282,16,PURPLE,23)
    s.line(1040,684,1089,684,GRAY,2,True)
    # Panel C includes all five actual positives, one actual weak negative and unknown.
    s.rect(1095,552,665,112,'#faf9fd',LINE)
    s.text(1110,580,'Description → Qwen3 text extractor',22,PURPLE,600)
    s.paragraph(1110,611,'Open behavior names + original-text evidence; validity and cue checks.',625,18,INK,25)
    s.line(1428,669,1428,693,GRAY,2,True)
    s.text(1097,720,'Evidence from the actual paragraph',21,INK,600)
    evidence=[('Smile','maintains a sustained smile'),('Raised brows','he raises his brows'),('Nod','nods his head slightly'),('Head tilt','tilting his head down and to his left'),('Speech','He then says, “Right?”')]
    for i,(name,quote) in enumerate(evidence):
        yy=751+i*27;s.text(1102,yy,name,17,ORANGE if name=='Speech' else BLUE,600)
        s.text(1245,yy,quote,16,INK)
    event_table(s,item,1095,898,665,23,15)
    s.text(1100,1111,'Presence only · positive 1 · weak negative 0.1 · masked 0',16,GRAY)
    s.text(35,1159,'Real sample: V03_S0557_I00000077 · 233.290–239.760 s · all cues and Event values copied from final training records.',17,GRAY)
    s.save(OUT/'data_pipeline_real_case.svg',1.75)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    items=[export_case(i+1,sid) for i,sid in enumerate(IDS)]
    component_assets(items[1]);pipeline(items[1])
    audit=[]
    for n,item in enumerate(items,1):
        record={k:v for k,v in item.items() if not k.startswith('_')}
        record['gate_exact']=item['_gate']
        write(OUT/f'case{n}'/'provenance.json',record)
        audit.append(dict(sample_id=item['id'],final_crop=item['final_crop'],duration_s=item['duration'],
                          listener_first_frame_sha256=sha(OUT/f'case{n}'/'listener_first_frame.png'),
                          speaker_wav_sha256=sha(OUT/f'case{n}'/'speaker.wav'),
                          before_after_sample_exact=True,caption_preserved=True,
                          original_source_export=bool(n==2),in_final_ltx500h=True,in_qwen_sft_train=True))
    write(OUT/'AUDIT.json',dict(cases=audit,training_data_changed=False,gpu_used=False,
          build_script=str(Path(__file__).resolve()),build_script_sha256=sha(__file__),
          figure_caption_excerpt='Full original caption, no omitted or rewritten sentences.',
          waveform='Signed min/max envelope from WAV PCM, no smoothing or synthetic samples. Before/after share scale.',
          event_semantics='Actual frozen y/known/effective_weight. No fabricated temporal or count labels.'))
    readme='''# 两个真实 case 的论文配图素材

## 第一个片段：输入条件

`naturalistic_V03_S1371_I00000205_local_08ad393f3e19`，6.306 秒。

- `case1/speaker_waveform.{svg,pdf,png}`：真实配对 speaker 音轨，有时间轴。
- `case1/speaker_waveform_transparent.{svg,pdf,png}`：透明背景、无文字版本，方便拼图。
- `case1/listener_first_frame.png`：最终修正版裁框后的 listener 首帧，512×512，和已有训练审核导出的首帧一致。
- `case1/speaker.wav`：对应 speaker 独立音轨；不是 listener 或混音。

## 第二个片段：完整数据处理流程示例

`naturalistic_V03_S0557_I00000077_local_c4e0cb6f423c`，6.470 秒；源区间 233.290–239.760 秒；原始人脸短边 p10=414.04 像素。

- `data_pipeline_real_case.{svg,pdf,png}`：沿用用户草图三部分布局的真实素材流程图。SVG文字、箭头、波形、表格可编辑；PDF为矢量排版，PNG为预览。
- `case2/original_frame.png`：原始完整单人视角帧，2160×3840。原始数据是双方各自的视频，未拼造双人镜头。
- `case2/original_frame_with_final_crop.png`：原始画面及实际最终裁框；x=0,y=134,width=height=1181；不改变原始图像比例。
- `case2/listener_first_frame.png` 与 `keyframe_01.png`–`keyframe_05.png`：最终裁框后的真实原生帧，512×512；具体源PTS与相对时间在 provenance.json。不是从压缩预览重新截取。
- `case2/keyframe_strip.{svg,pdf,png}`：连续关键帧素材。
- `case2/listener_before_after.{svg,pdf,png}`：原 listener／净化 listener 同尺度波形，附官方词、句、VAD区间。
- `case2/cue_timeline.{svg,pdf,png}`、`cues_exact.json`：两段点头候选、抬眉、Right?词线索，全部沿用现有结果。
- `case2/caption_verbatim.txt`、`caption_original.json`：完整原文；流程图保留完整原始段落，没有重写或删减动作。
- `case2/event_labels.{svg,pdf,png}`：实际训练Event标签节选；`event_targets_all19.csv`和`event_targets.json`为完整19维记录。
- `case2/listener_before.wav`、`listener_clean.wav`、`speaker.wav`：无损浮点PCM音轨；`listener_clean_preview.mp4`为有声预览，非训练缓存。

## 图中的口径

1. 两条样本均来自最后500h，且同时在Qwen SFT train中；来源由原有冻结20条清单复用，没有重新挑选或打标。
2. 首帧指训练输入的listener首帧。speaker条件是独立speaker音轨；没有把listener误当speaker。
3. 官方词与句均为3.710–3.970秒，VAD另保留4.296–4.900秒。并集后扩边得到3.610–4.070和4.196–5.000秒；10ms淡入淡出，区间外置零，时长不变。gate_exact.json记录精确样本索引。
4. 波形直接由真实采样的有符号最小/最大包络绘制。Before/After共享幅度刻度，未分别放大。第二条门控外原始能量约2.596%，去除部分并不等于已证实的纯杂音，不夸大为强降噪案例。
5. 点头候选为3.440–3.760和4.960–5.720秒；不能据此声称恰好点头两次。抬眉3.480–3.960秒；词线索Right?为3.710–3.970秒。检测线索不是人工时间真值；帧下标签只写实际时间，不为静态图补造动作判断。
6. Event是整段presence监督：本例5个正类为抬眉、点头、头倾斜、微笑和speech；frown为0.1权重弱负例；gaze shift为mask=0，不是确定负例。所有值来自已存训练缓存。mouth closing全局禁用，仅在完整19维表中保留。
7. 一次/反复动作共享一个整段Event标签。图中类别与证据节选保留原文含义，完整caption和完整标签可追溯。
8. 原始图、音频、caption、缓存及用户草图未改；只新增配图素材。来源和SHA256见AUDIT.json与各case/provenance.json。

复现：`/private/software/conda/bin/python scripts/build_data_pipeline_selected_cases_20261112.py`
'''
    (OUT/'README.md').write_text(readme)
    # Serve the finished files through the already established Cursor port.
    WEB.mkdir(parents=True,exist_ok=True)
    link=WEB/'assets'
    if not link.exists():link.symlink_to(OUT,target_is_directory=True)
    html='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>两个真实case · 配图素材</title><style>body{margin:32px auto;max-width:1500px;padding:0 24px;font:17px/1.65 system-ui;color:#30394b;background:#f6f8fc}section{background:white;border:1px solid #d1d8e3;border-radius:14px;padding:24px;margin:24px 0}img{max-width:100%;height:auto}a{color:#2169b0}.row{display:flex;gap:28px;align-items:start;flex-wrap:wrap}.row>div{flex:1;min-width:280px}video{width:400px;max-width:100%}audio{max-width:100%}code{overflow-wrap:anywhere}h1,h2{line-height:1.3}</style><h1>两个真实case · 数据处理配图素材</h1><p><a href="assets/selected_real_cases.zip" download>下载全部素材 ZIP</a> · <a href="assets/README.md">完整说明与口径</a></p><section><h2>① S1371：speaker音轨＋listener首帧</h2><div class="row"><div><img src="assets/case1/speaker_waveform.png"><p>Speaker独立音轨：</p><audio controls src="assets/case1/speaker.wav"></audio><p><a href="assets/case1/speaker_waveform.svg" download>波形SVG</a> · <a href="assets/case1/speaker_waveform_transparent.png" download>透明PNG</a> · <a href="assets/case1/listener_first_frame.png" download>首帧原图512×512</a></p></div><div style="flex:0 0 300px"><img src="assets/case1/listener_first_frame.png"></div></div></section><section><h2>② S0557：真实数据处理流程图</h2><p><a href="assets/data_pipeline_real_case.svg" download>可编辑SVG</a> · <a href="assets/data_pipeline_real_case.pdf" download>PDF</a> · <a href="assets/data_pipeline_real_case.png" download>高清PNG</a></p><a href="assets/data_pipeline_real_case.png"><img src="assets/data_pipeline_real_case.png"></a><div class="row"><div><video controls preload="metadata" src="assets/case2/listener_clean_preview.mp4"></video></div><div><p>原listener：</p><audio controls src="assets/case2/listener_before.wav"></audio><p>净化listener：</p><audio controls src="assets/case2/listener_clean.wav"></audio><p>Speaker：</p><audio controls src="assets/case2/speaker.wav"></audio></div></div></section><section><h2>第二条的独立素材</h2><p><a href="assets/case2/original_frame.png" download>原始完整帧</a> · <a href="assets/case2/original_frame_with_final_crop.png" download>原画＋最终裁框</a> · <a href="assets/case2/listener_first_frame.png" download>首帧</a> · <a href="assets/case2/caption_verbatim.txt">完整原caption</a> · <a href="assets/case2/event_targets_all19.csv" download>完整Event表</a> · <a href="assets/case2/cues_exact.json">检测线索与官方区间</a></p>'''
    for name in ['keyframe_strip','listener_before_after','cue_timeline','event_labels']:
        html+=f'<p><a href="assets/case2/{name}.svg" download>{name} · SVG</a> · <a href="assets/case2/{name}.png" download>PNG</a></p><img src="assets/case2/{name}.png">'
    html+='</section><p>净化前后波形使用相同幅度刻度。辅助候选不等于精确次数或时间真值；Event为整段presence标签。</p></html>'
    (WEB/'index.html').write_text(html)
    # ZIP outside root first, avoiding self-inclusion on reruns.
    archive=OUT.parent/'selected_real_cases'
    oldzip=OUT/'selected_real_cases.zip'
    oldzip.unlink(missing_ok=True)
    shutil.make_archive(str(archive),'zip',OUT)
    shutil.move(str(archive)+'.zip',oldzip)
    print(json.dumps(dict(output=str(OUT),url='http://localhost:8876/listen_benchmark_review/data_pipeline_selected2_20261112/',cases=audit),ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
