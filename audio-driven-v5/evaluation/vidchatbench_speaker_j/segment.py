"""VidChatBench 切分：固定 J v1，说话人只保留观察对象。

不修改冻结的 J 规则文件，也不跑音质模型。
human 固定为 people[0]（Speaker / Human · 观察对象）。
agent 只作为听的一方，参与重叠阻断和短反馈，不再反向出 pair。

画面和声音从 /public/0_DATA/2_Avatar/Speakervid/videos_fps24 的 24fps 单人片段读取。
官方 ASR 只有句级起止，没有词时间，不能拿来下刀。
词时间是对这两条 24fps 音轨用本地 Whisper large-v3 重新转写的。
"""
import hashlib
import importlib.util
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


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


response_segment = _load('responsenet_speaker_segment', ROOT / 'evaluation/responsenet_speaker_j/segment.py')

REPAIR = Path('/private/liqi_workspace/project_code/audio-driven-v4/evaluation/benchmark_alignment/assets/speakervid_repair_20261021')
MANIFEST = REPAIR / 'release/manifest.json'
READINESS = Path('/private/liqi_workspace/project_code/audio-driven-v4/evaluation/benchmark_alignment/assets/speakervid_official94_20261021/readiness.json')
FPS24 = Path('/public/0_DATA/2_Avatar/Speakervid/videos_fps24')
WHISPER = Path('/private/codes/exp/boxian_workspace/models/benchmark/huggingface/hub/models--openai--whisper-large-v3/snapshots/06f233fe06e710322aca913c1bc4249a0d71fce1')
RELEASE = ROOT / 'releases/listen_pair_segmentation_j_v1_20261029/manifest.json'
OUT = Path(__file__).resolve().parent / 'results'
MEDIA = Path(__file__).resolve().parent / 'fps24'
ADAPTATION = 'vidchatbench-fps24-fixed-observation-subject-j-v1'
WORD_TIME_SOURCE = 'whisper_large_v3_on_videos_fps24'


def assert_frozen_rule():
    pinned = json.loads(RELEASE.read_text())['module_sha256']
    actual = hashlib.sha256(Path(j.__file__).read_bytes()).hexdigest()
    if actual != pinned:
        raise SystemExit(f'J v1 规则文件已偏离冻结指纹：{actual} != {pinned}')
    return actual


def load_manifest():
    return json.loads(MANIFEST.read_text())['samples']


def wav_seconds(path):
    info = sf.info(path)
    return info.frames / info.samplerate


def local_word_document(path, media_seconds):
    """把本地词级转写收成 J 入口能读的词表。零时长和超出本条 wav 的词丢掉。"""
    doc = json.loads(Path(path).read_text())
    words = []
    dropped = 0
    for word in doc.get('words') or []:
        stamp = word.get('timestamp') or [None, None]
        if stamp[0] is None or stamp[1] is None:
            dropped += 1
            continue
        try:
            start, end = float(stamp[0]), float(stamp[1])
        except (TypeError, ValueError):
            dropped += 1
            continue
        if not (math.isfinite(start) and math.isfinite(end)) or start >= media_seconds:
            dropped += 1
            continue
        end = min(end, media_seconds)
        text = str(word.get('text') or '').strip()
        if end <= start or not text:
            dropped += 1
            continue
        words.append(dict(text=text, start_time=start, end_time=end))
    return dict(text=str(doc.get('text') or ''), timestamps=dict(words=words, segments=[])), dropped


def official_blocks(path, media_seconds):
    doc = json.loads(Path(path).read_text())
    segments = []
    for row in doc.get('segments') or []:
        text = str(row.get('text') or '').strip()
        if not text:
            continue
        start, end = float(row['start']), float(row['end'])
        segments.append(dict(start=start, end=end, text=text,
                             past_media=end > media_seconds + 0.05 or start >= media_seconds))
    return dict(text=str(doc.get('text') or '').strip(), segments=segments, source=str(path))


def readiness_row(sample_id):
    catalog = getattr(readiness_row, 'rows', None)
    if catalog is None:
        catalog = {row['id']: row for row in json.loads(READINESS.read_text())['samples']}
        readiness_row.rows = catalog
    if sample_id not in catalog:
        raise SystemExit(f'24fps 清单里没有样本 {sample_id}')
    return catalog[sample_id]


def fps24_sources(sample_id):
    """观察对象和 Agent 都取 videos_fps24 里的 24fps 单人片段。"""
    row = readiness_row(sample_id)
    sources = {}
    for role, key in (('speaker', 'human'), ('listener', 'agent')):
        video = Path(row['roles'][key]['local_proxy'])
        if FPS24 not in video.parents or not video.is_file():
            raise SystemExit(f'{sample_id} 的 {role} 不在 {FPS24}：{video}')
        sources[role] = dict(video=video, official_asr=Path(row['roles'][key]['official_asr']))
    return sources


def extract_wav(video, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists() or dest.stat().st_size == 0:
        subprocess.check_call([
            'ffmpeg', '-y', '-v', 'error', '-i', str(video),
            '-vn', '-ac', '1', '-ar', '16000', '-c:a', 'pcm_s16le', str(dest),
        ])
    return dest


def whisper_words(wav, dest):
    """官方 ASR 没有词时间。词时间只从这条 24fps 音轨现算，并缓存。"""
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    import transformers.pipelines.automatic_speech_recognition as asr_pipe
    asr_pipe.is_torchcodec_available = lambda: False
    from transformers import pipeline
    pipe = getattr(whisper_words, 'pipe', None)
    if pipe is None:
        pipe = pipeline('automatic-speech-recognition', model=str(WHISPER), device='cpu')
        whisper_words.pipe = pipe
    wave, rate = sf.read(wav, dtype='float32', always_2d=False)
    if getattr(wave, 'ndim', 1) > 1:
        wave = wave.mean(axis=1)
    result = pipe({'raw': wave, 'sampling_rate': rate}, return_timestamps='word')
    words = []
    for chunk in result.get('chunks') or []:
        stamp = chunk.get('timestamp') or (None, None)
        words.append(dict(text=str(chunk.get('text') or ''), timestamp=[stamp[0], stamp[1]]))
    dest.write_text(json.dumps(dict(text=result.get('text') or '', words=words), ensure_ascii=False, indent=2) + '\n')
    return dest


def build_context(sample):
    sample_id = sample['id']
    sources = fps24_sources(sample_id)
    folder = MEDIA / sample_id
    speaker_wav = str(extract_wav(sources['speaker']['video'], folder / 'speaker.wav'))
    listener_wav = str(extract_wav(sources['listener']['video'], folder / 'listener.wav'))
    speaker_seconds = wav_seconds(speaker_wav)
    listener_seconds = wav_seconds(listener_wav)
    speaker_doc, speaker_dropped = local_word_document(
        whisper_words(speaker_wav, folder / 'speaker_words.json'), speaker_seconds)
    listener_doc, listener_dropped = local_word_document(
        whisper_words(listener_wav, folder / 'listener_words.json'), listener_seconds)
    people = j.prepare_people([
        response_segment.person_from_official(speaker_doc),
        response_segment.person_from_official(listener_doc),
    ])
    rms = []
    for path in (speaker_wav, listener_wav):
        values, _duration = response_segment.rms_10ms(path)
        rms.append(values)
    limit = max(speaker_seconds, listener_seconds)
    width = int(math.ceil(limit * 100))
    rms = [np.pad(values, (0, max(0, width - len(values))))[:width] for values in rms]
    regions, gaps = j.role_regions(people, limit, [])
    kept = response_segment.keep_observation_subject(regions)
    videos = [response_segment.probe_video(sources['speaker']['video']),
              response_segment.probe_video(sources['listener']['video'])]
    ctx = j.SourceContext.__new__(j.SourceContext)
    ctx.item = dict(pair=sample_id, split='vidchatbench_local8', group_id=sample_id)
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
    summary = dict(
        sample_id=sample_id,
        media_root=str(FPS24),
        speaker_wav=speaker_wav,
        listener_wav=listener_wav,
        speaker_video=str(sources['speaker']['video']),
        listener_video=str(sources['listener']['video']),
        speaker_seconds=speaker_seconds,
        listener_seconds=listener_seconds,
        speaker_words=len(people[0]['words']),
        listener_words=len(people[1]['words']),
        speaker_words_dropped=speaker_dropped,
        listener_words_dropped=listener_dropped,
        regions_both_directions=len(regions),
        regions_observation_subject=len(kept),
        regions_dropped_listener_as_speaker=len(regions) - len(kept),
        limit_seconds=limit,
        official_asr=[
            dict(role='speaker', title=response_segment.SPEAKER_ROLE,
                 **official_blocks(sources['speaker']['official_asr'], speaker_seconds)),
            dict(role='listener', title=response_segment.LISTENER_ROLE,
                 **official_blocks(sources['listener']['official_asr'], listener_seconds)),
        ],
    )
    return ctx, summary


def segment_sample(sample, digest):
    ctx, summary = build_context(sample)
    rows, diagnostic = ctx.propose([])
    if any(row['speaker_index'] != response_segment.OBSERVATION_SUBJECT for row in rows):
        raise RuntimeError(f"{sample['id']} 出现了非观察对象 pair")
    pairs = [dict(pair_id=row['sample_id'], start=row['start'], end=row['end'],
                  duration_s=row['duration_s'], text=row['text'],
                  speaker_index=row['speaker_index'], speaker_role=response_segment.SPEAKER_ROLE,
                  listener_role=response_segment.LISTENER_ROLE,
                  semantic_start=row.get('semantic_start'), semantic_end=row.get('semantic_end'))
             for row in rows]
    summary.update(
        adaptation=ADAPTATION,
        policy='listen-pair-segmentation-j-v1',
        policy_revision=j.REVISION,
        rule_module='scripts/listen_local_recovery_semantic_trial.py',
        rule_sha256=digest,
        speaker_role=response_segment.SPEAKER_ROLE,
        listener_role=response_segment.LISTENER_ROLE,
        speaker_fixed_to_human_track=True,
        word_time_source=WORD_TIME_SOURCE,
        official_asr_has_word_times=False,
        paper_comparable=False,
        media_release='vidchatbench_content_verified_local_v2',
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
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=response_segment._jsonable) + '\n')
    return path


def main(argv):
    digest = assert_frozen_rule()
    samples = load_manifest()
    if not argv or argv[0] in ('-h', '--help'):
        print('用法：python evaluation/vidchatbench_speaker_j/segment.py <样本id>')
        print('      python evaluation/vidchatbench_speaker_j/segment.py --all')
        print(f'规则指纹 {digest}')
        return 2
    chosen = samples if argv[0] == '--all' else [row for row in samples if row['id'] == argv[0]]
    if not chosen:
        raise SystemExit(f'本地 8 条清单里没有样本 {argv[0]}')
    for sample in chosen:
        summary = segment_sample(sample, digest)
        path = write_result(summary)
        print(summary['sample_id'], 'pairs', summary['pair_count'],
              'dropped_listener_regions', summary['regions_dropped_listener_as_speaker'],
              'limit', round(summary['limit_seconds'], 2), path, flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
