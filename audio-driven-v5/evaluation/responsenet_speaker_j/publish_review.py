"""把已切好的 ResponseNet 样本放进完整原视频审核页。

先运行 segment.py 写出 results/<id>.json。本脚本不重新切分，也不跑音质模型。
它读取官方 speaker / listener 的视频、wav 和 ASR，生成：

- 左右并排的原时间线视频
- 进度条用的 review.json
- 每条 pair 分开的画面和音轨
- 页面脚本 responsenet_review.js
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT / 'scripts')]
import segment

PAGE = ROOT / 'data_previews/seamless_face_listening_v1/listen_full_source'
MEDIA = Path('/public/0_DATA/2_Avatar/ResponseNet/test')
TEXT = MEDIA / 'text'
DEFAULT_SAMPLES = ('6V1HXfo', '9lhidn7')
ROLES = (
    ('speaker', segment.SPEAKER_ROLE),
    ('listener', segment.LISTENER_ROLE),
)


def run(cmd):
    subprocess.run(cmd, check=True)


def panel(index, width, height):
    scale = min(720 / width, 480 / height)
    out_w, out_h = int(width * scale) // 2 * 2, int(height * scale) // 2 * 2
    return dict(x=index * 720 + (720 - out_w) / 2, y=(480 - out_h) / 2, width=out_w, height=out_h,
                source_width=width, source_height=height, rotation_degrees=0)


def probe(path):
    raw = subprocess.check_output([
        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height', '-of', 'json', str(path)])
    stream = json.loads(raw)['streams'][0]
    return int(stream['width']), int(stream['height'])


def render_source(sample, limit, folder):
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / 'source.mp4'
    videos = [MEDIA / 'video/speaker' / f'{sample}.mp4', MEDIA / 'video/listener' / f'{sample}.mp4']
    audios = [MEDIA / 'audio/speaker' / f'{sample}.wav', MEDIA / 'audio/listener' / f'{sample}.wav']
    panels = [panel(i, *probe(path)) for i, path in enumerate(videos)]
    if target.exists():
        return panels
    filters = []
    for i, item in enumerate(panels):
        filters.append(
            f'[{i}:v]setpts=PTS-STARTPTS,scale={item["width"]}:{item["height"]},setsar=1,'
            f'pad=720:480:(ow-iw)/2:(oh-ih)/2:black,fps=25[v{i}]')
    filters += [
        '[v0][v1]hstack=inputs=2[v]',
        '[2:a]aformat=sample_rates=48000:channel_layouts=mono,asetpts=PTS-STARTPTS[a0]',
        '[3:a]aformat=sample_rates=48000:channel_layouts=mono,asetpts=PTS-STARTPTS[a1]',
        '[a0][a1]join=inputs=2:channel_layout=stereo:map=0.0-FL|1.0-FR[a]',
    ]
    cmd = ['ffmpeg', '-y', '-v', 'error', '-nostdin']
    for path in videos + audios:
        cmd += ['-i', str(path)]
    cmd += ['-filter_complex', ';'.join(filters), '-map', '[v]', '-map', '[a]',
            '-t', f'{limit:.3f}', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '23',
            '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', '160k', '-movflags', '+faststart', str(target)]
    run(cmd)
    return panels


def cut_pair(sample, index, pair):
    start, end = pair['start'], pair['end']
    folder = PAGE / sample / 'clips'
    folder.mkdir(parents=True, exist_ok=True)
    files = {}
    for role, _title in ROLES:
        stem = f'pair{index:02d}_{role}'
        mp4 = folder / f'{stem}.mp4'
        wav = folder / f'{stem}.wav'
        if not mp4.exists():
            run(['ffmpeg', '-y', '-loglevel', 'error',
                 '-ss', f'{start:.3f}', '-to', f'{end:.3f}', '-i', str(MEDIA / 'video' / role / f'{sample}.mp4'),
                 '-ss', f'{start:.3f}', '-to', f'{end:.3f}', '-i', str(MEDIA / 'audio' / role / f'{sample}.wav'),
                 '-map', '0:v:0', '-map', '1:a:0',
                 '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '23',
                 '-vf', 'scale=640:640', '-c:a', 'aac', '-b:a', '160k',
                 '-movflags', '+faststart', str(mp4)])
        if not wav.exists():
            run(['ffmpeg', '-y', '-loglevel', 'error',
                 '-ss', f'{start:.3f}', '-to', f'{end:.3f}', '-i', str(MEDIA / 'audio' / role / f'{sample}.wav'),
                 str(wav)])
        files[role] = dict(video=f'{sample}/clips/{stem}.mp4', audio=f'{sample}/clips/{stem}.wav')
    return dict(index=index, start=start, end=end, duration_s=pair['duration_s'], text=pair['text'], files=files)


def asr_block(sample, result, role, title):
    used = ((result.get('asr_documents') or {}).get(role) or {})
    source = used.get('source') or 'official'
    path = Path(used['path']) if used.get('path') else TEXT / role / f'{sample}.json'
    if source == 'official':
        doc = json.loads(path.read_text())
    else:
        doc, _meta = segment.document_for(TEXT / role / f'{sample}.json', sample, role)
    if source == 'official':
        segments = []
        for row in (doc.get('timestamps') or {}).get('segments') or []:
            text = str(row.get('text') or '').strip()
            if text:
                segments.append(dict(start=float(row['start']), end=float(row['end']), text=text))
        label = '官方 ASR'
    else:
        segments = []
        for word in (doc.get('timestamps') or {}).get('words') or []:
            text = str(word.get('text') or '').strip()
            try:
                start, end = float(word['start_time']), float(word['end_time'])
            except (KeyError, TypeError, ValueError):
                continue
            if text and end > start:
                segments.append(dict(start=start, end=end, text=text))
        label = '本地 Whisper 词时间，打在这条官方音轨上'
    return dict(role=role, title=title, text=str(doc.get('text') or '').strip(), segments=segments,
                source=str(path), word_time_source=source, label=label)


def official_asr(sample, result=None):
    result = result or {}
    return [asr_block(sample, result, role, title) for role, title in ROLES]


def partition(limit, pairs, regions, gaps, unresolved):
    edges = {0.0, float(limit)}
    for rows in (pairs, regions, gaps, unresolved):
        for row in rows:
            edges.add(max(0.0, min(limit, float(row['start']))))
            edges.add(max(0.0, min(limit, float(row['end']))))
    ordered = sorted(edges)
    spans = []
    for start, end in zip(ordered, ordered[1:]):
        if end - start < 1e-4:
            continue
        point = (start + end) / 2
        pair = next((row for row in pairs if row['start'] <= point < row['end']), None)
        region = next((row for row in regions if row['start'] <= point < row['end']), None)
        gap = next((row for row in gaps if row['start'] <= point < row['end']), None)
        failed = [row for row in unresolved if row['start'] <= point < row['end']]
        text = ''
        speaker_index = None
        if pair:
            status, reasons, text = 'segmented', ['观察对象 pair，未跑音质模型'], pair['text']
            speaker_index = 0
        elif failed:
            status = 'blocked'
            reasons = sorted({row.get('reason') or 'safe_region_below_2s' for row in failed})
            speaker_index = failed[0].get('speaker_index')
        elif region and region['speaker_index'] == 1:
            status, reasons, speaker_index = 'uncovered', ['Agent 自己在说，不反向切'], 1
        elif gap:
            status, reasons = 'uncovered', [gap.get('reason') or 'no_primary_speech']
        else:
            status, reasons = 'uncovered', ['insufficient_safe_context_or_padding']
        value = dict(start=start, end=end, status=status, reasons=reasons, text=text)
        if speaker_index is not None:
            value['speaker_index'] = speaker_index
        if spans and all(spans[-1].get(key) == value.get(key) for key in ('status', 'reasons', 'text', 'speaker_index')):
            spans[-1]['end'] = end
        else:
            spans.append(value)
    return spans


def publish(sample_id):
    result_path = HERE / 'results' / f'{sample_id}.json'
    if not result_path.exists():
        raise SystemExit(f'还没有切分结果 {result_path}，先运行 segment.py {sample_id}')
    manifest = {row['id']: row for row in segment.load_manifest()}
    if sample_id not in manifest:
        raise SystemExit(f'官方清单里没有 {sample_id}')
    result = json.loads(result_path.read_text())
    ctx, built = segment.build_context(manifest[sample_id])
    regions, gaps = segment.j.role_regions(ctx.people, ctx.limit, [])
    unresolved = result['diagnostic'].get('unresolved') or []
    pairs = result['pairs']
    covered = sum(row['duration_s'] for row in pairs)
    attempts = [dict(sample_id=row['pair_id'], start=row['start'], end=row['end'], duration_s=row['duration_s'],
                     speaker_index=0, listener_index=1, text=row['text'], pool='本次切分（未跑音质）',
                     reasons=['观察对象 pair，未跑音质模型'],
                     recovery_complete_sentence=row.get('semantic_end') == 'sentence_terminal',
                     crops=[dict(status='unmeasured'), dict(status='unmeasured')]) for row in pairs]
    local_asr = any((result.get('asr_documents') or {}).get(role, {}).get('source') not in (None, 'official')
                    for role, _title in ROLES)
    summary = dict(pair=sample_id, source_seconds=ctx.limit, old_kept_clips=0, old_kept_seconds=0.0,
                   initial_candidates=len(pairs), qc_attempts=0, final_kept_clips=len(pairs),
                   new_kept_seconds=covered, added_seconds=covered, removed_seconds=0.0,
                   training_admitted=False, visual_filter_applied=False,
                   dropped_listener_regions=built['regions_dropped_listener_as_speaker'],
                   note='ResponseNet 官方原视频。说话人固定为 A：Speaker / Human · 观察对象。音质模型没有跑。')
    page_copy = None
    if local_asr:
        page_copy = dict(
            method='A 是观察对象，B 是 Agent，左右都是官方原画面和原音轨。词时间来自打在这两条音轨上的 Whisper。长于 1.5 秒的词已按该音轨的实际声音收短，切分规则没有改。橙色是这次切出的观察对象 pair。',
            asr_title='这次切分用的词时间，speaker 和 Agent 分开',
            asr_note='官方 test/text 里观察对象没有可用词。这里是收短之后的 Whisper 词时间。点一个词会跳到原视频的对应时间。',
        )
        if not pairs:
            page_copy['method'] += ' 这条仍然没有 pair：剩下只属于观察对象的缝不足 2 秒，或者两边仍然重叠。'
            page_copy['cut_title'] = '这条没有裁出 pair'
            page_copy['cut_note'] = '收短词时间之后，仍然没有大约 2 到 12 秒、且不与 Agent 实质发言重叠的观察对象片段。'
    folder = PAGE / sample_id
    panels = render_source(sample_id, ctx.limit, folder)
    review = dict(pair=sample_id, review_kind='responsenet_fixed_speaker', summary=summary,
                  media=f'{sample_id}/source.mp4', panels=panels, old=[], attempts=attempts, kept_ids=[],
                  timeline=partition(ctx.limit, pairs, regions, gaps, unresolved), shots=[[], []],
                  regions=[dict(start=row['start'], end=row['end'], speaker_index=row['speaker_index']) for row in regions],
                  source_gaps=gaps, diagnostic=dict(unresolved=unresolved), overview_crops=[], issues=[],
                  transcripts=[[dict(start=row['start'], end=row['end'], text=row['text']) for row in person['segments']]
                               for person in ctx.people],
                  transcript_label='本地词时间' if local_asr else '官方 ASR',
                  page_copy=page_copy,
                  official_asr=official_asr(sample_id, result),
                  cut_media=[cut_pair(sample_id, index, pair) for index, pair in enumerate(pairs, 1)])
    (folder / 'review.json').write_text(json.dumps(review, ensure_ascii=False, indent=2) + '\n')
    return summary


def install_page():
    target = PAGE / 'responsenet_review.js'
    shutil.copyfile(HERE / 'review_page.js', target)
    html = PAGE / 'index.html'
    text = html.read_text()
    if 'src="responsenet_review.js"' not in text:
        text = text.replace('<script>', '<script src="responsenet_review.js"></script><script>', 1)
    # Helpers stay inside the page script. Export them once for the external file.
    export = "window.reviewPage={query:$,esc,fmt,seek};"
    broken = "window.reviewPage={query:$,esc,fmt,seek};const fmt="
    if broken in text:
        text = text.replace(export, '', 1)
    if export not in text:
        anchor = "function seek(t,play=false){"
        if anchor not in text:
            raise SystemExit('审核页里找不到跳转函数')
        text = text.replace(anchor, export + anchor, 1)
    text = replace_role_label(text)
    text = replace_review_hook(text)
    html.write_text(text)


def replace_role_label(text):
    old = "${(data.review_kind==='responsenet_fixed_speaker'?(i?'Listener / Agent · 真实参考':'Speaker / Human · 观察对象'):(i?'B':'A'))} 官方 ASR："
    new = "${(window.responseNetRoleName?window.responseNetRoleName(i):(i?'B':'A'))} ${data.review_kind==='responsenet_fixed_speaker'?'官方 ASR':'ASR'}："
    if old in text:
        return text.replace(old, new, 1)
    if "window.responseNetRoleName?window.responseNetRoleName(i)" in text:
        return text
    raise SystemExit('审核页里找不到 ASR 标签')


def replace_review_hook(text):
    start = text.find("if(data.review_kind==='responsenet_fixed_speaker'){")
    if start < 0:
        if 'window.renderResponseNetReview?.()' in text:
            return text
        raise SystemExit('审核页里找不到 ResponseNet 展示分支')
    depth = 0
    for index in range(start, len(text)):
        char = text[index]
        if char == '{':
            depth += 1
        elif char == '}':
            depth -= 1
            if depth == 0:
                end = index + 1
                break
    else:
        raise SystemExit('ResponseNet 展示分支括号不配对')
    return text[:start] + 'window.renderResponseNetReview?.();' + text[end:]


def main(argv):
    samples = tuple(argv) if argv else DEFAULT_SAMPLES
    install_page()
    index_path = PAGE / 'index.json'
    index = json.loads(index_path.read_text())
    index['pairs'] = [row for row in index['pairs'] if row['pair'] not in samples]
    for sample in samples:
        summary = publish(sample)
        index['pairs'].append(dict(pair=sample, summary=summary, responsenet=True))
        print(sample, 'pairs', summary['final_kept_clips'], 'seconds', round(summary['source_seconds'], 2), flush=True)
    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
