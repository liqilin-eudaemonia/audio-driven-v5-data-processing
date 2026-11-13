"""把已切好的 VidChatBench 本地样本放进完整时间线审核页。

先运行 segment.py。本脚本不重新切分，也不跑音质模型。
画面和音轨来自本地内容验收过的 human / agent 分轨，不是官方未修复原片。
"""
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT / 'scripts')]
import segment


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_response_publish():
    """ResponseNet 的发布脚本也叫 segment。先让开这个名字，再把本文件的模块放回去。"""
    saved = sys.modules.pop('segment', None)
    sys.path.insert(0, str(ROOT / 'evaluation/responsenet_speaker_j'))
    try:
        return _load('responsenet_publish_review', ROOT / 'evaluation/responsenet_speaker_j/publish_review.py')
    finally:
        if saved is not None:
            sys.modules['segment'] = saved


response_publish = _load_response_publish()

PAGE = ROOT / 'data_previews/seamless_face_listening_v1/listen_full_source'
REVIEW_JS = ROOT / 'evaluation/responsenet_speaker_j/review_page.js'
READINESS = Path('/private/liqi_workspace/project_code/audio-driven-v4/evaluation/benchmark_alignment/assets/speakervid_official94_20261021/readiness.json')
# 8 条里只有这两条切出了观察对象 pair。
DEFAULT_SAMPLES = (
    'P8ZY3udZ7no_1920x1080_full_video_118_A_00',
    'ep3yBD3MjM8_1920x1080_full_video_057_A_01',
)


def duration_of(path):
    raw = subprocess.check_output([
        'ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'json', str(path)])
    return float(json.loads(raw)['format']['duration'])


def render_source(sample, limit, folder):
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / 'source.mp4'
    videos = [sample['input']['speaker_video'], sample['target']['video']]
    audios = [sample['input']['speaker_audio'], sample['target']['audio']]
    panels = [response_publish.panel(i, *response_publish.probe(path)) for i, path in enumerate(videos)]
    if target.exists():
        return panels
    filters = []
    for index, item in enumerate(panels):
        pad = max(0.0, limit - duration_of(videos[index]))
        filters.append(
            f'[{index}:v]setpts=PTS-STARTPTS,scale={item["width"]}:{item["height"]},setsar=1,'
            f'pad=720:480:(ow-iw)/2:(oh-ih)/2:black,fps=24,'
            f'tpad=stop_mode=clone:stop_duration={pad:.3f}[v{index}]')
    filters += [
        '[v0][v1]hstack=inputs=2[v]',
        f'[2:a]aformat=sample_rates=48000:channel_layouts=mono,atrim=0:{limit:.3f},asetpts=PTS-STARTPTS,apad=whole_dur={limit:.3f}[a0]',
        f'[3:a]aformat=sample_rates=48000:channel_layouts=mono,atrim=0:{limit:.3f},asetpts=PTS-STARTPTS,apad=whole_dur={limit:.3f}[a1]',
        '[a0][a1]join=inputs=2:channel_layout=stereo:map=0.0-FL|1.0-FR[a]',
    ]
    cmd = ['ffmpeg', '-y', '-v', 'error', '-nostdin']
    for path in videos + audios:
        cmd += ['-i', str(path)]
    cmd += ['-filter_complex', ';'.join(filters), '-map', '[v]', '-map', '[a]',
            '-t', f'{limit:.3f}', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '23',
            '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', '160k', '-movflags', '+faststart', str(target)]
    response_publish.run(cmd)
    return panels


def cut_role(video, audio, start, end, mp4, wav):
    span = max(0.05, end - start)
    video_left = max(0.0, duration_of(video) - start)
    audio_left = max(0.0, duration_of(audio) - start)
    cmd = ['ffmpeg', '-y', '-loglevel', 'error']
    if video_left > 0.02:
        cmd += ['-ss', f'{start:.3f}', '-t', f'{span:.3f}', '-i', str(video)]
        video_filter = (f'[0:v]scale=640:-2,setsar=1,fps=25,'
                        f'tpad=stop_mode=clone:stop_duration={max(0.0, span - video_left):.3f}[v]')
    else:
        cmd += ['-f', 'lavfi', '-t', f'{span:.3f}', '-i', 'color=c=black:s=640x360:r=25']
        video_filter = '[0:v]scale=640:-2,setsar=1[v]'
    if audio_left > 0.02:
        cmd += ['-ss', f'{start:.3f}', '-t', f'{span:.3f}', '-i', str(audio)]
        audio_filter = (f'[1:a]aformat=sample_rates=16000:channel_layouts=mono,'
                        f'atrim=0:{span:.3f},asetpts=PTS-STARTPTS,apad=whole_dur={span:.3f}[a]')
    else:
        cmd += ['-f', 'lavfi', '-t', f'{span:.3f}', '-i', 'anullsrc=r=16000:cl=mono']
        audio_filter = '[1:a]aformat=sample_rates=16000:channel_layouts=mono[a]'
    if not mp4.exists():
        response_publish.run(cmd + ['-filter_complex', f'{video_filter};{audio_filter}',
                                    '-map', '[v]', '-map', '[a]', '-t', f'{span:.3f}',
                                    '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '23',
                                    '-c:a', 'aac', '-b:a', '160k', '-movflags', '+faststart', str(mp4)])
    if not wav.exists():
        audio_cmd = ['ffmpeg', '-y', '-loglevel', 'error']
        if audio_left > 0.02:
            audio_cmd += ['-ss', f'{start:.3f}', '-t', f'{span:.3f}', '-i', str(audio),
                          '-af', f'atrim=0:{span:.3f},apad=whole_dur={span:.3f}', str(wav)]
        else:
            audio_cmd += ['-f', 'lavfi', '-t', f'{span:.3f}', '-i', 'anullsrc=r=16000:cl=mono', str(wav)]
        response_publish.run(audio_cmd)


def cut_pair(sample, index, pair):
    start, end = pair['start'], pair['end']
    folder = PAGE / sample['id'] / 'clips'
    folder.mkdir(parents=True, exist_ok=True)
    files = {}
    media = {
        'speaker': (sample['input']['speaker_video'], sample['input']['speaker_audio']),
        'listener': (sample['target']['video'], sample['target']['audio']),
    }
    for role, (video, audio) in media.items():
        stem = f'pair{index:02d}_{role}'
        mp4 = folder / f'{stem}.mp4'
        wav = folder / f'{stem}.wav'
        cut_role(video, audio, start, end, mp4, wav)
        files[role] = dict(video=f'{sample["id"]}/clips/{stem}.mp4', audio=f'{sample["id"]}/clips/{stem}.wav')
    return dict(index=index, start=start, end=end, duration_s=pair['duration_s'], text=pair['text'], files=files)


def clear_previous_render(folder):
    """旧页面用的是 25fps 验收文件。改从 24fps 片段切之后，这些成品要重做。"""
    if not folder.exists():
        return
    for name in ('source.mp4', 'original_speaker.mp4', 'original_listener.mp4'):
        path = folder / name
        if path.is_symlink() or path.exists():
            path.unlink()
    clips = folder / 'clips'
    if clips.is_dir():
        for path in clips.iterdir():
            path.unlink()


def link_originals(sample_id, folder):
    """本机只有 24fps 单人片段。整段 1920×1080 原片不在磁盘上。"""
    catalog = json.loads(READINESS.read_text())
    row = next(item for item in catalog['samples'] if item['id'] == sample_id)
    linked = []
    for role, key, title in (
        ('speaker', 'human', 'Speaker / Human · 观察对象'),
        ('listener', 'agent', 'Listener / Agent · 真实参考'),
    ):
        source = Path(row['roles'][key]['local_proxy'])
        if not source.is_file():
            raise SystemExit(f'找不到 24fps 片段 {source}')
        target = folder / f'original_{role}.mp4'
        if target.is_symlink() or target.exists():
            target.unlink()
        target.symlink_to(source)
        linked.append(dict(
            role=role, title=title, file=f'{sample_id}/original_{role}.mp4',
            seconds=duration_of(source), fps=24,
            note=f'24fps 单人片段 {source.name}。对应的整段 1920×1080 原片不在本机。',
        ))
    return linked


def cut_notes(sample, pair):
    notes = {}
    listener_seconds = duration_of(sample['target']['video'])
    if pair['start'] >= listener_seconds - 0.02:
        notes['listener'] = (f'这条 pair 从 {pair["start"]:.2f} 秒开始，agent 本地视频只有 {listener_seconds:.2f} 秒，'
                             '起点已经超出文件，所以这里是黑屏和静音。')
    elif pair['end'] > listener_seconds + 0.05:
        notes['listener'] = (f'agent 本地视频在 {listener_seconds:.2f} 秒结束，这条 pair 到 {pair["end"]:.2f} 秒，'
                             '超出的部分是黑屏和静音。')
    return notes


def page_copy(summary):
    return dict(
        method=('A 是 Speaker / Human · 观察对象，B 是 Listener / Agent。'
                '两条画面和声音都来自 /public/0_DATA/2_Avatar/Speakervid/videos_fps24 里的 24fps 单人片段。'
                '橙色是这次从这两条片段切出的观察对象 pair。音质模型没有跑。'
                '官方 ASR 只有句级时间，下刀用的词时间是对这两条音轨重新转写的。'),
        legend=('<span><i class="dot" style="background:#ffb65d"></i>本次切出的观察对象 pair</span>'
                '<span><i class="dot" style="background:#485975"></i>未成 pair（含 Agent 自己说话）</span>'
                '<span><i class="dot" style="background:#de6e7a"></i>不足 2 秒等未成段</span>'
                '<span>细线 = 每个 pair 的边界</span>'),
        cards=[
            ['24fps 片段', f'{summary["source_seconds"]:.1f} 秒', 'videos_fps24 里的 human 在左，agent 在右'],
            ['观察对象 pair', f'{summary["final_kept_clips"]} 段 · {summary["new_kept_seconds"]:.1f} 秒', '未跑音质模型'],
            ['Agent 说话区间', f'{summary["dropped_listener_regions"]} 个未反向切', '不把听众当成说话人'],
            ['时间轴', '与上方视频同一条', '两条 24fps 片段里较长的那条'],
        ],
        shot='没有做镜头检测。播放器里是这两条 24fps 片段排在一起的完整时间线，切分不会把没选中的时间从视频里删掉。',
        asr_title='官方 ASR，human 和 agent 分开',
        asr_note='原文来自这两条 24fps 片段对应的官方 ASR，没有改字。官方文件只有句级起止，没有词时间。标了“超出本条音视频”的时间不能当成这支文件里的位置。',
        cut_title='裁出来的 pair：画面和音轨分开',
        cut_note='按进度条上的起止，从 videos_fps24 的这两条 24fps 文件切开。较短的一条在自己的结尾之后是黑屏和静音。',
    )


def publish(sample_id):
    result_path = HERE / 'results' / f'{sample_id}.json'
    if not result_path.exists():
        raise SystemExit(f'还没有切分结果 {result_path}，先运行 segment.py {sample_id}')
    manifest = {row['id']: row for row in segment.load_manifest()}
    if sample_id not in manifest:
        raise SystemExit(f'本地 8 条清单里没有 {sample_id}')
    sample = manifest[sample_id]
    result = json.loads(result_path.read_text())
    ctx, built = segment.build_context(sample)
    regions, gaps = segment.j.role_regions(ctx.people, ctx.limit, [])
    unresolved = result['diagnostic'].get('unresolved') or []
    pairs = result['pairs']
    covered = sum(row['duration_s'] for row in pairs)
    attempts = [dict(sample_id=row['pair_id'], start=row['start'], end=row['end'], duration_s=row['duration_s'],
                     speaker_index=0, listener_index=1, text=row['text'], pool='本次切分（未跑音质）',
                     reasons=['观察对象 pair，未跑音质模型'],
                     recovery_complete_sentence=row.get('semantic_end') == 'sentence_terminal',
                     crops=[dict(status='unmeasured'), dict(status='unmeasured')]) for row in pairs]
    summary = dict(pair=sample_id, label=sample_id.split('_')[0], source_seconds=ctx.limit,
                   old_kept_clips=0, old_kept_seconds=0.0, initial_candidates=len(pairs), qc_attempts=0,
                   final_kept_clips=len(pairs), new_kept_seconds=covered, added_seconds=covered,
                   removed_seconds=0.0, training_admitted=False, visual_filter_applied=False,
                   dropped_listener_regions=built['regions_dropped_listener_as_speaker'],
                   note='画面和声音来自 videos_fps24 的 24fps 片段。说话人固定为 A。音质模型没有跑。')
    folder = PAGE / sample_id
    media = dict(id=sample_id,
                 input=dict(speaker_video=result['speaker_video'], speaker_audio=result['speaker_wav']),
                 target=dict(video=result['listener_video'], audio=result['listener_wav']))
    clear_previous_render(folder)
    panels = render_source(media, ctx.limit, folder)
    review = dict(pair=sample_id, review_kind='vidchatbench_fixed_speaker', summary=summary,
                  page_copy=page_copy(summary), transcript_label='本地词时间',
                  media=f'{sample_id}/source.mp4', panels=panels, old=[], attempts=attempts, kept_ids=[],
                  timeline=response_publish.partition(ctx.limit, pairs, regions, gaps, unresolved),
                  shots=[[], []],
                  regions=[dict(start=row['start'], end=row['end'], speaker_index=row['speaker_index']) for row in regions],
                  source_gaps=gaps, diagnostic=dict(unresolved=unresolved), overview_crops=[], issues=[],
                  transcripts=[[dict(start=row['start'], end=row['end'], text=row['text']) for row in person['segments']]
                               for person in ctx.people],
                  official_asr=result['official_asr'],
                  cut_media=[dict(cut_pair(media, index, pair), notes=cut_notes(media, pair))
                             for index, pair in enumerate(pairs, 1)])
    (folder / 'review.json').write_text(json.dumps(review, ensure_ascii=False, indent=2) + '\n')
    return summary


def install_page():
    shutil.copyfile(REVIEW_JS, PAGE / 'responsenet_review.js')
    html_path = PAGE / 'index.html'
    text = html_path.read_text()
    old_label = "${data.review_kind==='responsenet_fixed_speaker'?'官方 ASR':'ASR'}"
    new_label = "${data.transcript_label||(data.review_kind==='responsenet_fixed_speaker'?'官方 ASR':'ASR')}"
    if old_label in text:
        text = text.replace(old_label, new_label, 1)
    elif 'data.transcript_label||' not in text:
        raise SystemExit('审核页里找不到转写标签')
    nav = ("responsenet.map(r=>`<a href=\"?pair=${encodeURIComponent(r.pair)}\">${esc(r.pair)} · ${r.summary.source_seconds.toFixed(0)} 秒</a>`).join(' · ')};"
           "const holdouts=")
    vidchat_nav = ("responsenet.map(r=>`<a href=\"?pair=${encodeURIComponent(r.pair)}\">${esc(r.pair)} · ${r.summary.source_seconds.toFixed(0)} 秒</a>`).join(' · ')};"
                   "const vidchatbench=index.pairs.filter(r=>r.vidchatbench);if(vidchatbench.length){$('#issueLinks').hidden=false;"
                   "$('#issueLinks').innerHTML+='<br><strong>VidChatBench 观察对象切分：</strong> '+"
                   "vidchatbench.map(r=>`<a href=\"?pair=${encodeURIComponent(r.pair)}\">${esc(r.summary.label||r.pair)} · ${r.summary.source_seconds.toFixed(0)} 秒</a>`).join(' · ')};"
                   "const holdouts=")
    if nav in text:
        text = text.replace(nav, vidchat_nav, 1)
    elif 'r.vidchatbench' not in text:
        raise SystemExit('审核页里找不到 ResponseNet 导航')
    html_path.write_text(text)


def main(argv):
    samples = tuple(argv) if argv else DEFAULT_SAMPLES
    install_page()
    index_path = PAGE / 'index.json'
    index = json.loads(index_path.read_text())
    index['pairs'] = [row for row in index['pairs'] if row['pair'] not in samples]
    for sample in samples:
        summary = publish(sample)
        index['pairs'].append(dict(pair=sample, summary=summary, vidchatbench=True))
        print(sample, 'pairs', summary['final_kept_clips'], 'seconds', round(summary['source_seconds'], 2), flush=True)
    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
