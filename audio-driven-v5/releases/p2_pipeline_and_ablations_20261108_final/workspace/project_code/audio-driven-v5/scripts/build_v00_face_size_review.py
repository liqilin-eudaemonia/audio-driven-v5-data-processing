"""Bounded five-sample review: source pixels, face box and recorded crop. CPU only."""
import html
import json
import math
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data_previews/listen_benchmark_review/v00_face_size_20261101'


def run(args):
    return subprocess.run(args, check=True, capture_output=True).stdout


def ff(src, timestamp, target, vf=None, duration=None):
    args = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-threads', '1',
            '-ss', str(timestamp), '-i', str(src), '-an', '-filter_threads', '1']
    if vf:
        # Normalize decoded pixels before odd-width crops; avoids the system
        # ffmpeg scaler's corrupted 10-bit planar crop output on these sources.
        args += ['-vf', 'format=rgb24,' + vf]
    if duration is None:
        args += ['-frames:v', '1']
    else:
        args += ['-t', str(duration), '-c:v', 'libx264', '-preset', 'fast',
                 '-crf', '18', '-pix_fmt', 'yuv420p', '-movflags', '+faststart']
    run(args + [str(target)])


def main():
    records = json.loads((OUT / 'sources.json').read_text())
    results = []
    sections = []
    for index, rec in enumerate(records, 1):
        dest = OUT / str(index)
        dest.mkdir(exist_ok=True)
        source = rec['source']; archive = source['source']
        quality = json.loads(Path(rec['quality']['measurement_file']).read_text())
        valid = [x for x in quality['focus_samples'] if x.get('bbox_native') and x.get('raw_detected')]
        chosen = min(valid, key=lambda x: abs(x['t'] - rec['duration_s'] / 2))
        box = chosen['bbox_native']
        crop = source['video_transform']['crop']
        x, y, w, h = [crop[k] for k in ('x', 'y', 'width', 'height')]
        with tempfile.TemporaryDirectory(prefix='v00_face_review_') as td:
            video = Path(td) / 'source.mp4'
            # Read one indexed MP4 member; no tar scan or extraction of other members.
            with open(archive['archive'], 'rb') as inp, video.open('wb') as out:
                inp.seek(archive['offset']); remaining = archive['byte_size']
                while remaining:
                    chunk = inp.read(min(1048576, remaining))
                    if not chunk:
                        raise EOFError(archive['archive'])
                    out.write(chunk); remaining -= len(chunk)
            probe = json.loads(run(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                                  '-show_entries', 'stream=width,height', '-of', 'json', str(video)]))['streams'][0]
            width, height = probe['width'], probe['height']
            assert 0 <= x < x + w <= width and 0 <= y < y + h <= height
            assert 0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height
            t = chosen['source_time']
            ff(video, t, dest/'source.png')
            face = [math.floor(box[0]), math.floor(box[1]), math.ceil(box[2]), math.ceil(box[3])]
            fw, fh = face[2]-face[0], face[3]-face[1]
            ff(video, t, dest/'face.png', f'crop={fw}:{fh}:{face[0]}:{face[1]}:exact=1')
            filt = f'crop={w}:{h}:{x}:{y}:exact=1,scale=512:512:flags=bicubic'
            ff(video, t, dest/'crop512.png', filt)
            ff(video, source['source_start_s'], dest/'crop512.mp4', filt, rec['duration_s'])
        item = dict(sample_id=rec['sample_id'], p10=rec['face_short_side_p10'],
                    p50=rec['face_short_side_p50'], source_width=width, source_height=height,
                    preview_source_time=t, preview_relative_time=t-source['source_start_s'],
                    face_box=box, face_pixels=[fw,fh], crop=crop,
                    crop_revision=rec['crop_revision'], media_dir=str(index))
        results.append(item)
        esc = html.escape
        sections.append(f'''<section><h2>{index}. 人脸短边 p10：{item['p10']:.1f} px</h2>
<p class="id">{esc(rec['sample_id'])}</p><p>p50：{item['p50']:.1f} px · 本帧人脸：{fw}×{fh} px · 原画：{width}×{height} · 记录裁框：{w}×{h} · 片长：{rec['duration_s']:.2f}s</p>
<div class="row"><div><h3>原始整帧（缩小展示）</h3><div class="source" style="aspect-ratio:{width}/{height}"><img src="{index}/source.png"><svg viewBox="0 0 {width} {height}"><rect x="{x}" y="{y}" width="{w}" height="{h}" stroke="#ffd54a"/><rect x="{box[0]}" y="{box[1]}" width="{box[2]-box[0]}" height="{box[3]-box[1]}" stroke="#46e49c"/></svg></div><p>绿框：本帧人脸；黄框：本条记录裁框</p><a href="{index}/source.png" target="_blank">打开完整原图</a></div>
<div><h3>人脸局部，保持原生像素</h3><div class="native"><img src="{index}/face.png" width="{fw}" height="{fh}"></div><p>1 图像像素对应 1 CSS 像素；浏览器请用100%缩放。这里没有放大。</p><a href="{index}/face.png" download>下载原生人脸局部</a></div>
<div><h3>记录裁框 → 512×512</h3><img class="crop" src="{index}/crop512.png"><p>双三次缩放，无增强、无超分。放大不会增加原始细节。</p><details><summary>播放整段裁剪视频（仅画面，无音轨）</summary><video controls preload="none" playsinline poster="{index}/crop512.png" src="{index}/crop512.mp4"></video></details></div></div></section>''')
        print(index, rec['sample_id'], fw, fh, flush=True)
    (OUT/'data.json').write_text(json.dumps(results, ensure_ascii=False, indent=2)+'\n')
    page = '''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>V00 原始人脸尺寸实物对照</title>
<style>body{margin:24px auto;padding:0 20px;max-width:1600px;background:#f3f5f8;color:#172333;font:16px/1.65 sans-serif}h1{font-size:26px}h2{font-size:21px}.row{display:grid;grid-template-columns:minmax(300px,1fr) 260px 512px;gap:24px}section{background:white;padding:24px;border-radius:12px;margin:24px 0}h3{font-size:16px}.id{overflow-wrap:anywhere;color:#496078}p{margin:8px 0}.source{position:relative;width:100%}.source img{width:100%;display:block}.source svg{position:absolute;inset:0;width:100%;height:100%;fill:none;stroke-width:4}.native{height:300px;display:flex;align-items:center;justify-content:center;background:#edf0f4}.native img{max-width:none}.crop,video{width:512px;height:512px;max-width:100%;object-fit:contain}a{color:#185baf}@media(max-width:1150px){.row{grid-template-columns:1fr}.source{max-width:700px}.native{width:300px}}</style>
<h1>V00 原始人脸尺寸：直接看实际画面</h1><p>从最终保留且来源含情绪字段的 V00 中，按人脸 p10 尺寸档选了5个不同来源的示例。前4条覆盖主要尺寸范围；第5条是少数≥224 px的对照。这是分档示例，不是随机质量评估。</p>
<p><b>p10 是整段采样人脸短边的第10百分位，下面单帧尺寸可能不同。</b>依次看原画占比、原生人脸局部和512裁剪后的细节。裁剪使用各条现有记录中的实际坐标，未替换裁框；没有美化、锐化或超分。</p>'''
    (OUT/'index.html').write_text(page+'\n'.join(sections)+'</html>')


if __name__ == '__main__':
    main()
