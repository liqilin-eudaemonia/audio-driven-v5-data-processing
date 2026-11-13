"""ResponseNet 切分：固定 J v1，说话人只保留观察对象。

不修改 releases/listen_pair_segmentation_j_v1_20261029 冻结的规则文件。
官方 speaker 轨道固定为 people[0]（页面上的 Speaker / Human · 观察对象）。
官方 listener 轨道只作为听的一方，参与重叠阻断和短反馈，不再反向出 pair。
一条官方样本里观察对象的多段发言，仍按 J 拆成多条 pair。

本入口只做 J 的初始切分（propose）。不加载音质模型，不改 Seamless 全量结果。
"""
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'src')]
import listen_local_recovery_semantic_trial as j  # noqa: E402
import listen_turns_v8 as turns  # noqa: E402

MANIFEST = Path('/private/liqi_workspace/project_code/audio-driven-v4/evaluation/benchmark_alignment/assets/responsenet_inputs/manifest.json')
RELEASE = ROOT / 'releases/listen_pair_segmentation_j_v1_20261029/manifest.json'
HERE = Path(__file__).resolve().parent
OUT = HERE / 'results'
LOCAL_ASR = HERE / 'local_asr'
OBSERVATION_SUBJECT = 0
SPEAKER_ROLE = 'Speaker / Human · 观察对象'
LISTENER_ROLE = 'Listener / Agent · 真实参考'
ADAPTATION = 'responsenet-fixed-observation-subject-j-v1'


def rule_sha256():
    return hashlib.sha256(Path(j.__file__).read_bytes()).hexdigest()


def assert_frozen_rule():
    pinned = json.loads(RELEASE.read_text())['module_sha256']
    actual = rule_sha256()
    if actual != pinned:
        raise SystemExit(f'J v1 规则文件已偏离冻结指纹：{actual} != {pinned}')
    return actual


def load_manifest():
    data = json.loads(MANIFEST.read_text())
    return data['samples']


def official_words(doc):
    words = []
    for word in (doc.get('timestamps') or {}).get('words') or []:
        try:
            start, end = float(word['start_time']), float(word['end_time'])
        except (KeyError, TypeError, ValueError):
            continue
        text = str(word.get('text') or '')
        if end > start and text.strip():
            words.append(dict(start=start, end=end, text=text))
    words.sort(key=lambda w: (w['start'], w['end']))
    return words


def _aligned(words):
    ok = bool(words) and all(0 < w['end'] - w['start'] <= 1.5 for w in words)
    return ok and all(b['start'] >= a['end'] - .001 for a, b in zip(words, words[1:]))


def _classify(text, start, end):
    feedback = turns.feedback(text, end - start)
    tokens = turns.tokens(text)
    uncertain = (not feedback and 3 < end - start <= 6 and 0 < len(tokens) <= 2
                 and all(token in turns.FEEDBACK_WORDS for token in tokens))
    return feedback or uncertain, uncertain


def _segment(index, text, words):
    aligned = _aligned(words)
    start = min(w['start'] for w in words) if aligned else words[0]['start']
    end = max(w['end'] for w in words) if aligned else words[-1]['end']
    is_feedback, uncertain = _classify(text, start, end)
    stored = []
    for word in words:
        item = dict(word, feedback=is_feedback, segment=f'seg_{index:03d}')
        stored.append(item)
    return dict(id=f'seg_{index:03d}', start=start, end=end, text=text, words=stored,
                feedback=is_feedback, word_count=len(stored), word_aligned=aligned,
                formal_takeover=False, long_feedback_timing_uncertain=uncertain)


def person_from_official(doc):
    """把官方词时间收成 J 使用的 person。没有词的空时段不占话轮。"""
    words = official_words(doc)
    claimed = [False] * len(words)
    segments = []
    for segment in (doc.get('timestamps') or {}).get('segments') or []:
        try:
            start, end = float(segment['start']), float(segment['end'])
        except (KeyError, TypeError, ValueError):
            continue
        if not (math.isfinite(start) and math.isfinite(end)) or end <= start:
            continue
        chosen = []
        for index, word in enumerate(words):
            if claimed[index]:
                continue
            overlap = min(word['end'], end) - max(word['start'], start)
            if overlap > 0 and overlap >= 0.5 * (word['end'] - word['start']):
                chosen.append(index)
        if not chosen:
            continue
        for index in chosen:
            claimed[index] = True
        text = str(segment.get('text') or '').strip() or ' '.join(words[i]['text'] for i in chosen)
        segments.append(_segment(len(segments), text, [words[i] for i in chosen]))
    pending = [words[i] for i, taken in enumerate(claimed) if not taken]
    group = []
    for word in pending:
        if group and word['start'] - group[-1]['end'] > j.MAX_WORD_GAP:
            text = ' '.join(item['text'] for item in group)
            segments.append(_segment(len(segments), text, group))
            group = []
        group.append(word)
    if group:
        text = ' '.join(item['text'] for item in group)
        segments.append(_segment(len(segments), text, group))
    flat = sorted((word for segment in segments for word in segment['words']), key=lambda w: (w['start'], w['end']))
    return dict(segments=segments, words=flat, vad=[])


def keep_observation_subject(regions):
    return [region for region in regions if region['speaker_index'] == OBSERVATION_SUBJECT]


def _track_threshold(mono, sample_rate):
    hop = max(1, int(round(sample_rate * 0.02)))
    count = len(mono) // hop
    if count == 0:
        return 0.008
    rms = np.sqrt(np.mean(mono[:count * hop].reshape(count, hop) ** 2, axis=1))
    return max(0.008, float(np.percentile(rms, 95)) * 0.25)


def _active_span(mono, sample_rate, start, end, threshold):
    """在词时间里面只保留这条音轨真正有声音的最长一段。"""
    hop = max(1, int(round(sample_rate * 0.02)))
    left = max(0, int(start * sample_rate))
    right = min(len(mono), int(end * sample_rate))
    chunk = mono[left:right]
    count = len(chunk) // hop
    if count == 0:
        return None
    rms = np.sqrt(np.mean(chunk[:count * hop].reshape(count, hop) ** 2, axis=1))
    active = np.flatnonzero(rms >= threshold)
    if len(active) == 0:
        return None
    best_start = best_end = run = previous = int(active[0])
    for index in active[1:]:
        index = int(index)
        if index == previous + 1:
            previous = index
            continue
        if previous - run > best_end - best_start:
            best_start, best_end = run, previous
        run = previous = index
    if previous - run > best_end - best_start:
        best_start, best_end = run, previous
    onset = start + best_start * hop / sample_rate
    offset = min(end, start + (best_end + 1) * hop / sample_rate)
    if offset - onset <= 0.02:
        return None
    return onset, offset


def repair_long_words(doc, mono, sample_rate):
    """把长于 1.5 秒的词收到实际有声音的区间。静音上的长词删掉。

    官方文本不走这里。J 会把超过 1.5 秒、又对不齐的词扩成整段阻断，
    所以先把这种词收短，规则本身不改。
    """
    threshold = _track_threshold(mono, sample_rate)
    repaired = []
    for word in (doc.get('timestamps') or {}).get('words') or []:
        text = str(word.get('text') or '').strip()
        try:
            start, end = float(word['start_time']), float(word['end_time'])
        except (KeyError, TypeError, ValueError):
            continue
        if not text or end <= start:
            continue
        if end - start <= 1.5:
            repaired.append(dict(text=text, start_time=start, end_time=end))
            continue
        span = _active_span(mono, sample_rate, start, end, threshold)
        if span is None:
            continue
        onset, offset = span
        piece = onset
        while offset - piece > 1.5:
            repaired.append(dict(text=text, start_time=round(piece, 3), end_time=round(piece + 1.4, 3)))
            piece += 1.4
        repaired.append(dict(text=text, start_time=round(piece, 3), end_time=round(offset, 3)))
    updated = dict(doc)
    updated['timestamps'] = dict(words=repaired, segments=[])
    updated['text'] = ' '.join(word['text'] for word in repaired)
    updated['word_time_repair'] = 'energy_tighten_words_over_1_5s'
    return updated


def document_for(official_path, sample_id, role, local_root=None):
    """有本地词时间就用本地词时间。官方文件始终不改。"""
    official_path = Path(official_path)
    root = Path(local_root) if local_root is not None else LOCAL_ASR
    local_path = root / sample_id / f'{role}.json'
    if local_path.exists():
        doc = json.loads(local_path.read_text())
        source = str(doc.get('word_time_source') or 'local')
        audio = doc.get('audio')
        if audio and Path(audio).exists():
            wave, sample_rate = sf.read(audio, dtype='float32', always_2d=True)
            doc = repair_long_words(doc, wave.mean(axis=1), sample_rate)
        return doc, dict(source=source, path=str(local_path))
    return json.loads(official_path.read_text()), dict(source='official', path=str(official_path))


def rms_10ms(path):
    """与 J 相同的 10 毫秒原始 RMS，直接读官方 wav。

    个别官方 wav 是 11025 Hz，不能整除出 10 毫秒。只在算 RMS 时重采样到 16 kHz，
    导出的音轨仍是官方原文件。
    """
    with sf.SoundFile(path) as handle:
        if handle.samplerate % 100:
            wave = handle.read(dtype='float32', always_2d=True).mean(axis=1)
            duration = len(wave) / handle.samplerate
            target = 16000
            count = max(1, int(round(duration * target)))
            wave = np.interp(np.linspace(0, len(wave) - 1, count), np.arange(len(wave)), wave)
            hop = target // 100
            usable = len(wave) // hop * hop
            if not usable:
                return np.empty(0), duration
            block = wave[:usable].reshape(-1, hop).astype(np.float64)
            return np.sqrt(np.mean(block ** 2, axis=1)), duration
        hop = handle.samplerate // 100
        duration = handle.frames / handle.samplerate
        parts = []
        while True:
            wave = handle.read(hop * 10000, dtype='float32', always_2d=True)
            if not len(wave):
                break
            mono = wave.mean(axis=1)
            count = len(mono) // hop
            if count:
                block = mono[:count * hop].reshape(count, hop).astype(np.float64)
                parts.append(np.sqrt(np.mean(block ** 2, axis=1)))
    values = np.concatenate(parts) if parts else np.empty(0)
    return values, duration


def probe_video(path):
    raw = subprocess.check_output([
        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height,avg_frame_rate', '-of', 'json', str(path),
    ])
    stream = json.loads(raw)['streams'][0]
    num, den = stream['avg_frame_rate'].split('/')
    fps = float(num) / float(den)
    return dict(width=int(stream['width']), height=int(stream['height']), fps=fps,
                source=str(path), rotation_degrees=0, arrays={})


def build_context(sample):
    sample_id = sample['id']
    listener_text = Path(sample['target']['text'])
    speaker_text = listener_text.parents[1] / 'speaker' / f'{sample_id}.json'
    speaker_doc, speaker_asr = document_for(speaker_text, sample_id, 'speaker')
    listener_doc, listener_asr = document_for(listener_text, sample_id, 'listener')
    people = j.prepare_people([
        person_from_official(speaker_doc),
        person_from_official(listener_doc),
    ])
    speaker_wav = sample['input']['speaker_audio']
    listener_wav = sample['target']['audio']
    rms = []
    durations = []
    for path in (speaker_wav, listener_wav):
        values, duration = rms_10ms(path)
        rms.append(values)
        durations.append(duration)
    limit = min(durations)
    regions, gaps = j.role_regions(people, limit, [])
    kept = keep_observation_subject(regions)
    videos = [probe_video(sample['input']['speaker_video']), probe_video(sample['target']['video'])]
    ctx = j.SourceContext.__new__(j.SourceContext)
    ctx.item = dict(pair=sample_id, split='responsenet_test', group_id=sample_id)
    ctx.people = people
    ctx.rms = rms
    ctx.limit = limit
    ctx.records = [dict(id=f'{sample_id}:speaker'), dict(id=f'{sample_id}:listener')]
    ctx.sources = videos
    ctx.geometry = videos
    ctx.shots = []
    ctx.shots_by_track = [[], []]
    ctx.regions = kept
    ctx.source_gaps = gaps
    ctx.reviews = []
    return ctx, dict(
        sample_id=sample_id,
        speaker_text=str(speaker_text),
        listener_text=str(listener_text),
        asr_documents=dict(speaker=speaker_asr, listener=listener_asr),
        paper_comparable=speaker_asr['source'] == 'official' and listener_asr['source'] == 'official',
        speaker_words=len(people[0]['words']),
        listener_words=len(people[1]['words']),
        regions_both_directions=len(regions),
        regions_observation_subject=len(kept),
        regions_dropped_listener_as_speaker=len(regions) - len(kept),
        limit_seconds=limit,
    )


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def segment_sample(sample, digest):
    ctx, summary = build_context(sample)
    rows, diagnostic = ctx.propose([])
    if any(row['speaker_index'] != OBSERVATION_SUBJECT for row in rows):
        raise RuntimeError(f"{sample['id']} 出现了非观察对象 pair")
    pairs = [dict(pair_id=row['sample_id'], start=row['start'], end=row['end'],
                  duration_s=row['duration_s'], text=row['text'],
                  speaker_index=row['speaker_index'], speaker_role=SPEAKER_ROLE,
                  listener_role=LISTENER_ROLE, semantic_start=row.get('semantic_start'),
                  semantic_end=row.get('semantic_end')) for row in rows]
    summary.update(
        adaptation=ADAPTATION,
        policy='listen-pair-segmentation-j-v1',
        policy_revision=j.REVISION,
        rule_module='scripts/listen_local_recovery_semantic_trial.py',
        rule_sha256=digest,
        speaker_role=SPEAKER_ROLE,
        listener_role=LISTENER_ROLE,
        speaker_fixed_to_official_speaker_track=True,
        stage='initial_segmentation',
        audio_qc_status='not_recomputed',
        pair_count=len(pairs),
        pairs=pairs,
        unresolved=len(diagnostic.get('unresolved') or []),
        clips=rows,
        diagnostic=diagnostic,
    )
    return summary


def write_result(summary):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{summary['sample_id']}.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_jsonable) + '\n')
    return path


def main(argv):
    digest = assert_frozen_rule()
    samples = load_manifest()
    if not argv or argv[0] in ('-h', '--help'):
        print('用法：python evaluation/responsenet_speaker_j/segment.py <样本id>')
        print('      python evaluation/responsenet_speaker_j/segment.py --all')
        print('      python evaluation/responsenet_speaker_j/segment.py --local-only')
        print(f'规则指纹 {digest}')
        return 2
    if argv[0] == '--local-only':
        chosen = [s for s in samples if (LOCAL_ASR / s['id'] / 'speaker.json').exists()]
        if not chosen:
            raise SystemExit(f'没有本地词时间：{LOCAL_ASR}')
    elif argv[0] == '--all':
        chosen = samples
    else:
        chosen = [s for s in samples if s['id'] == argv[0]]
    if not chosen:
        raise SystemExit(f'清单里没有样本 {argv[0]}')
    for sample in chosen:
        summary = segment_sample(sample, digest)
        path = write_result(summary)
        print(summary['sample_id'], 'pairs', summary['pair_count'],
              'dropped_listener_regions', summary['regions_dropped_listener_as_speaker'],
              'speaker_words', summary['speaker_words'], path, flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
