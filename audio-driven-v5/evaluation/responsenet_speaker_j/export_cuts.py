"""把 140 条切分结果导出成可查阅的文件，并写位置清单。

先运行 segment.py --all。本脚本不重新切分，也不跑音质模型。
每条 pair 各写一份观察对象和 Agent 的画面、音轨、ASR。
官方整段文件的位置一并记在 cuts/index.json 和 CUTS.md。
"""
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE)]
import segment

OFFICIAL = Path('/public/0_DATA/2_Avatar/ResponseNet/test')
CUTS = HERE / 'cuts'
ROLES = ('speaker', 'listener')


def run(cmd):
    subprocess.run(cmd, check=True)


def slice_asr(doc, source, start, end):
    words = []
    for word in (doc.get('timestamps') or {}).get('words') or []:
        try:
            word_start, word_end = float(word['start_time']), float(word['end_time'])
        except (KeyError, TypeError, ValueError):
            continue
        if word_end > start and word_start < end and str(word.get('text') or '').strip():
            words.append(dict(text=str(word.get('text') or ''), start=word_start, end=word_end))
    segments = []
    for row in (doc.get('timestamps') or {}).get('segments') or []:
        text = str(row.get('text') or '').strip()
        if not text:
            continue
        try:
            row_start, row_end = float(row['start']), float(row['end'])
        except (KeyError, TypeError, ValueError):
            continue
        if row_end > start and row_start < end:
            segments.append(dict(text=text, start=row_start, end=row_end))
    text = ' '.join(word['text'].strip() for word in words).strip()
    return dict(
        source=str(source),
        word_time_source=doc.get('word_time_source') or 'official',
        pair_start=start, pair_end=end, text=text, words=words, segments=segments,
    )


def cut_media(sample_id, index, pair, role, text_path):
    start, end = pair['start'], pair['end']
    folder = CUTS / sample_id / f'pair{index:02d}'
    folder.mkdir(parents=True, exist_ok=True)
    video = folder / f'{role}.mp4'
    audio = folder / f'{role}.wav'
    asr_path = folder / f'{role}_asr.json'
    official_video = OFFICIAL / 'video' / role / f'{sample_id}.mp4'
    official_audio = OFFICIAL / 'audio' / role / f'{sample_id}.wav'
    if not video.exists():
        run(['ffmpeg', '-y', '-loglevel', 'error',
             '-ss', f'{start:.3f}', '-to', f'{end:.3f}', '-i', str(official_video),
             '-ss', f'{start:.3f}', '-to', f'{end:.3f}', '-i', str(official_audio),
             '-map', '0:v:0', '-map', '1:a:0',
             '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '23',
             '-vf', 'scale=640:640', '-c:a', 'aac', '-b:a', '160k',
             '-movflags', '+faststart', str(video)])
    if not audio.exists():
        run(['ffmpeg', '-y', '-loglevel', 'error',
             '-ss', f'{start:.3f}', '-to', f'{end:.3f}', '-i', str(official_audio), str(audio)])
    if not asr_path.exists():
        official_text = OFFICIAL / 'text' / role / f'{sample_id}.json'
        doc, meta = segment.document_for(official_text, sample_id, role)
        asr_path.write_text(json.dumps(slice_asr(doc, meta['path'], start, end), ensure_ascii=False, indent=2) + '\n')
    return dict(video=str(video), audio=str(audio), asr=str(asr_path))


def official_paths(sample_id):
    record = {}
    for role in ROLES:
        record[role] = dict(
            video=str(OFFICIAL / 'video' / role / f'{sample_id}.mp4'),
            audio=str(OFFICIAL / 'audio' / role / f'{sample_id}.wav'),
            asr=str(OFFICIAL / 'text' / role / f'{sample_id}.json'),
        )
    return record


def asr_used(result, sample_id, role):
    item = ((result.get('asr_documents') or {}).get(role) or {})
    path = item.get('path')
    if path and Path(path).exists():
        return dict(source=item.get('source') or 'official', path=str(path))
    return dict(source='official', path=str(OFFICIAL / 'text' / role / f'{sample_id}.json'))


def export_sample(sample_id):
    result = json.loads((HERE / 'results' / f'{sample_id}.json').read_text())
    official = official_paths(sample_id)
    used = {role: asr_used(result, sample_id, role) for role in ROLES}
    pairs = []
    for index, pair in enumerate(result['pairs'], 1):
        pairs.append(dict(
            index=index, pair_id=pair['pair_id'], start=pair['start'], end=pair['end'],
            duration_s=pair['duration_s'], text=pair['text'],
            speaker=cut_media(sample_id, index, pair, 'speaker', used['speaker']['path']),
            listener=cut_media(sample_id, index, pair, 'listener', used['listener']['path']),
        ))
    return dict(
        id=sample_id,
        pair_count=len(pairs),
        limit_seconds=result['limit_seconds'],
        speaker_role=segment.SPEAKER_ROLE,
        listener_role=segment.LISTENER_ROLE,
        paper_comparable=result.get('paper_comparable', True),
        asr_documents=used,
        official=official,
        pairs=pairs,
    )


def write_markdown(catalog):
    lines = [
        '# ResponseNet 切分结果位置',
        '',
        '说话人固定为 Speaker / Human · 观察对象。Agent 自己说话的区间没有反向切。音质模型没有跑。',
        '',
        '每条样本先给出官方整段文件，再给出按 pair 切开的画面、音轨和 ASR。',
        '',
        f"样本 {catalog['sample_count']} 条，pair {catalog['pair_count']} 条，合计 {catalog['pair_seconds']:.2f} 秒。",
        f"有 pair 的样本 {catalog['samples_with_pairs']} 条。没有 pair 的 {len(catalog['samples_without_pairs'])} 条："
        + '、'.join(f'`{item}`' for item in catalog['samples_without_pairs']) + '。',
        '',
        '位置：',
        '',
        '- 切分判断：`results/<样本id>.json`',
        '- 切开的画面、音轨、ASR：`cuts/<样本id>/pairNN/`',
        '- 机器可读清单：`cuts/index.json`',
        '- 本地词时间：`local_asr/<样本id>/{speaker,listener}.json`',
        f"- 官方整段：`{catalog['official_root']}`",
        '',
        '官方 ASR 没有可用词的样本，词时间来自对官方分轨打的 Whisper large-v3，写在 `local_asr/<样本id>/`。官方 `test/text` 没有改。这些样本的 `paper_comparable` 为 false。',
        '',
    ]
    for sample in catalog['samples']:
        lines.append(f"## {sample['id']}")
        lines.append('')
        lines.append(f"时间轴 {sample['limit_seconds']:.2f} 秒，观察对象 pair {sample['pair_count']} 条。")
        lines.append('')
        lines.append('| 角色 | 画面 | 音轨 | ASR |')
        lines.append('|---|---|---|---|')
        for role, title in (('speaker', '观察对象'), ('listener', 'Agent')):
            item = sample['official'][role]
            lines.append(f"| 官方整段 · {title} | `{item['video']}` | `{item['audio']}` | `{item['asr']}` |")
            used = (sample.get('asr_documents') or {}).get(role) or {}
            if used.get('source') not in (None, 'official'):
                lines.append(f"| 本次切分用的词时间 · {title} |  |  | `{used['path']}` |")
        lines.append('')
        if not sample['pairs']:
            lines.append('这条没有切出观察对象 pair。')
            lines.append('')
            continue
        for pair in sample['pairs']:
            lines.append(f"### pair {pair['index']:02d} · {pair['start']:.3f}–{pair['end']:.3f} 秒")
            lines.append('')
            lines.append(pair['text'])
            lines.append('')
            lines.append('| 角色 | 画面 | 音轨 | ASR |')
            lines.append('|---|---|---|---|')
            for role, title in (('speaker', '观察对象'), ('listener', 'Agent')):
                item = pair[role]
                lines.append(f"| {title} | `{item['video']}` | `{item['audio']}` | `{item['asr']}` |")
            lines.append('')
    path = HERE / 'CUTS.md'
    path.write_text('\n'.join(lines))
    return path


def main():
    results = sorted((HERE / 'results').glob('*.json'))
    if not results:
        raise SystemExit('还没有切分结果，先运行 segment.py --all')
    workers = max(1, int(os.environ.get('RESPONSENET_CUT_WORKERS', '8')))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        samples = list(pool.map(lambda path: export_sample(path.stem), results))
    samples.sort(key=lambda item: item['id'])
    missing = [item['id'] for item in samples if item['pair_count'] == 0]
    catalog = dict(
        policy='listen-pair-segmentation-j-v1',
        adaptation=segment.ADAPTATION,
        audio_qc_status='not_recomputed',
        speaker_fixed_to_official_speaker_track=True,
        official_root=str(OFFICIAL),
        sample_count=len(samples),
        samples_with_pairs=sum(1 for item in samples if item['pair_count']),
        samples_without_pairs=missing,
        pair_count=sum(item['pair_count'] for item in samples),
        pair_seconds=round(sum(pair['duration_s'] for item in samples for pair in item['pairs']), 2),
        samples=samples,
    )
    CUTS.mkdir(parents=True, exist_ok=True)
    index_path = CUTS / 'index.json'
    index_path.write_text(json.dumps(catalog, ensure_ascii=False, indent=2) + '\n')
    markdown = write_markdown(catalog)
    print('index', index_path, 'pairs', catalog['pair_count'], flush=True)
    print('doc', markdown, flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
