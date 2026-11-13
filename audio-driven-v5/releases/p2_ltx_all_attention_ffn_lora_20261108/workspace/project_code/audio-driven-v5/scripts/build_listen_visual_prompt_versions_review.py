#!/usr/bin/env python3
"""Publish the frozen October 29 visual prompt experiments; never edit labels."""
import hashlib
import base64
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
import soundfile as sf
from listen_audio_postprocess.recipes import masks, read_listener_audio
BASE = ROOT / 'data_previews/listen_benchmark_review'
OUT = BASE / 'prompt_versions_20261029'
RUNS = {
    'r1': 'listen_visual_observable50_20261029',
    'r2': 'listen_visual_observation50_20261029',
    'r3': 'listen_visual_actions50_20261029',
}
SERIES = [
    ('r1:baseline', 'A · 原始简单 prompt'),
    ('r1:observable', 'B · 追加禁止推演'),
    ('r2:baseline', 'A · 原始简单 prompt（第二轮）'),
    ('r2:observation_v1', 'C · 新观察式 v1'),
    ('r3:observation_v1', 'C · 新观察式 v1（第三轮）'),
    ('r3:observed_actions_v2', 'D · 允许动作命名＋动作清单'),
]
FOCUS = {
    'S1103': ('动作是否完整', '先看持续微笑，再看右手遮嘴、睁眼与抬眉。检查新版是否漏动作，以及衣服、耳饰描述是否多余；末尾 speaking / reacting 是否超出视觉依据。'),
    'S0404': ('是否过度合并', 'D 的整段输出只有一个长区间，文字内提到两次闭眼时间。检查这些动作是否应独立计时；切换短窗确认是否另有候选，不能只凭整段输出断言全片漏检。'),
    'S1032': ('是否切得太碎', '第三轮 C、D 的整段各有 52 条。对照慢放，判断相邻重复句是在描述真实多次动作，还是把连续状态切成许多短条。'),
    'S2147': ('心理推演残留', '留意思考、反思等解释性措辞。画面是否只支持眼睛、嘴或头部变化？区分可见描述与对内心状态的猜测。'),
    'S1049': ('否定点头不是检出', 'D 含 nod 词的描述可能是在说没有点头。阅读完整句子，并观察实际运动；不能把关键词出现次数当作动作检出次数。'),
    'S0343': ('动作名称与漏检', '旧版写过点头，但它不是人工真值。检查实际是否存在完整上下运动及返回，还是转头、低头或姿态变化；同时检查新版是否遗漏真实动作。'),
    'S1052': ('持续表情是否保留', '比较微笑、嘴部与眼部描述。检查减少重复后，持续微笑是否仍被覆盖；不能仅凭条数减少判断质量提高。'),
    'S1279': ('重复与真实多次动作', '检查反复出现的目光、嘴部描述及相邻短区间。同一句描述在不同时间可能对应真实多次动作，不应一律删除。'),
    'S0263': ('否定动作措辞', '留意 no / not 等否定词：未见点头与实际检出点头是两回事。对照视频判断动作与时间，而非统计关键词。'),
    'S2076': ('否定动作措辞', '留意未见明显点头等措辞，检查其是否混进了本应只列实际发生事件的输出。'),
    'S1033': ('高帧率是否更好', '在新增 C 输入方式对照中，整段多图29条、视频8fps为15条、视频16fps为71条。检查16fps是否把动作切成重复短区间，而不是只比较帧数。'),
}


def read(p):
    return json.loads(p.read_text())


def compact_mask(mask):
    """Store only unity intervals and exact float32 fades; omitted samples are zero."""
    def spans(active):
        edges = np.diff(np.pad(active.astype(np.int8), (1, 1)))
        return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))
    return {'ones': spans(mask == 1), 'fades': [
        [a, base64.b64encode(mask[a:b].astype('<f4').tobytes()).decode()]
        for a, b in spans((mask != 0) & (mask != 1))]}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    samples = read(ROOT / 'experiments' / RUNS['r1'] / 'samples.json')
    ids = [s['sample_id'] for s in samples]
    assert len(ids) == len(set(ids)) == 50
    targets = {t['sample_id']: t for t in read(ROOT / 'experiments/p2_train50h_postprocessed_reselected_20261027/manifests/targets.json')}
    analyses, results, hashes = {}, {}, {}
    for rid, run in RUNS.items():
        folder = ROOT / 'experiments' / run
        other = read(folder / 'samples.json')
        assert {s['sample_id']: s['source'] for s in other} == {s['sample_id']: s['source'] for s in samples}
        analyses[rid] = read(folder / 'analysis.json')
        results[rid] = {}
        for sid in ids:
            path = folder / 'results' / (sid + '.json')
            results[rid][sid] = read(path)
            hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    payload = {'schema': 'listen-visual-prompt-review-v1', 'human_ground_truth': False,
               'training_admitted': False, 'runs': RUNS, 'series': [], 'samples': []}
    for key, label in SERIES:
        rid, arm = key.split(':')
        payload['series'].append({'key': key, 'label': label, 'run': RUNS[rid],
                                  'stats': analyses[rid]['stats'][arm]})
    for sample in samples:
        sid, src = sample['sample_id'], sample['source']
        target = Path(targets[sid]['target_video'])
        meta = read(target.with_name('media.json'))
        assert target.is_file()
        assert meta['video_transform'] == src['video_transform'], sid
        assert abs(meta['source_start_s'] - src['source_start_s']) < 1e-6, sid
        assert abs(meta['duration_s'] - src['duration_s']) < 1e-6, sid
        media = OUT / 'media' / sid
        media.mkdir(parents=True, exist_ok=True)
        link = media / 'video.mp4'
        if link.is_symlink():
            assert link.resolve() == target.resolve()
        elif not link.exists():
            link.symlink_to(target)
        else:
            raise RuntimeError(f'Refuse to overwrite {link}')
        code = sid.split('_')[2]
        focus = FOCUS.get(code)
        item = {'id': sid, 'short': code, 'duration': src['duration_s'],
                'video': f'media/{sid}/video.mp4', 'focus': focus,
                'source_start_s': src['source_start_s'], 'crop': src['video_transform'],
                'series': {}, 'render_duration': meta['render_duration_s']}
        recipe_path = ROOT / 'experiments/p2_train50h_postprocessed_reselected_20261027/audio_postprocess/recipes' / (sid + '.json')
        recipe = read(recipe_path)
        assert recipe['sample_id'] == sid
        assert Path(recipe['source']['path']).resolve() == Path(targets[sid]['target_audio']).resolve()
        assert recipe['source']['sha256'] == targets[sid]['media_hashes']['target_audio']
        wave, rate, audio_meta = read_listener_audio(recipe)
        gate, extra, gate_info = masks(recipe)
        for role, path in [('listener_original', Path(recipe['source']['path'])), ('speaker_reference', target.with_name('speaker.wav'))]:
            assert path.is_file(), path
            info = sf.info(path)
            assert info.samplerate == rate and abs(info.duration - src['duration_s']) < 1/rate + 1e-6
            if role == 'listener_original':
                assert info.subtype == 'PCM_16' and info.channels == 1
            link = media / (role + '.wav')
            if link.is_symlink():
                assert link.resolve() == path.resolve()
            elif not link.exists():
                link.symlink_to(path)
            else:
                raise RuntimeError(f'Refuse to overwrite {link}')
        # Publish masks and provenance only. Processed audio is created in browser memory.
        item['audio'] = {
            'original': f'media/{sid}/listener_original.wav',
            'speaker': f'media/{sid}/speaker_reference.wav',
            'recipe_sha256': audio_meta['recipe_sha256'],
            'source_sha256': recipe['source']['sha256'],
            'source_samples': recipe['source']['samples'], 'sample_rate': rate, 'channels': 1,
            'gate': compact_mask(gate), 'extra': compact_mask(extra),
            'processed_pcm_sha256': hashlib.sha256(wave.astype('<f4').tobytes()).hexdigest(),
            'processed_full_silence': not bool(np.any(wave)),
            'waveform_changed': audio_meta['waveform_changed'],
            'original_rms': float(np.sqrt(np.mean(sf.read(recipe['source']['path'], dtype='float32')[0].astype('float64')**2))),
            'processed_rms': float(np.sqrt(np.mean(wave.astype('float64')**2))),
            'gate_positive_intervals_samples': gate_info['positive_intervals_samples'],
            'speaker_is_context_only': True,
        }
        reference_frames = {}
        for key, _ in SERIES:
            rid, arm = key.split(':')
            records = [r for r in results[rid][sid]['records'] if r['arm'] == arm]
            assert records and len({r['view'] for r in records}) == len(records)
            item['series'][key] = []
            for r in records:
                frames = r['media']['video']['frames']
                signature = [(f['time_s'], f['source_pts'], f['jpeg_sha256']) for f in frames]
                ref = reference_frames.setdefault(r['view'], (r['start'], r['end'], signature))
                assert ref == (r['start'], r['end'], signature), (sid, key, r['view'])
                assert all(f['size'] == [512, 512] for f in frames)
                rec = {k: r[k] for k in ('view', 'start', 'end', 'status', 'prompt', 'raw', 'finish_reason')}
                rec.update({'parsed': r.get('parsed'), 'error': r.get('error'), 'frame_count': len(frames)})
                # Preserve window-local times and descriptions exactly. The UI offsets times only for playback.
                item['series'][key].append(rec)
        payload['samples'].append(item)
    payload['samples'].sort(key=lambda s: (list(FOCUS).index(s['short']) if s['short'] in FOCUS else 99, s['id']))
    input_run = ROOT / 'experiments/listen_visual_input50_20261029'
    if (input_run / 'COMPLETE.json').exists():
        input_analysis = read(input_run / 'analysis.json')
        assert input_analysis['all_samples_complete']
        input_names = [('image8', 'C · 多图＋时间戳 8 fps（本轮）'),
                       ('video8', 'C · 原生视频 8 fps'), ('video16', 'C · 原生视频 16 fps')]
        payload['runs']['r4'] = input_run.name
        for arm, label in input_names:
            payload['series'].append({'key': 'r4:'+arm, 'label': label, 'run': input_run.name,
                                      'stats': input_analysis['stats'][arm]})
        for item in payload['samples']:
            file = input_run / 'results' / (item['id'] + '.json')
            d = read(file)
            hashes[str(file.relative_to(ROOT))] = hashlib.sha256(file.read_bytes()).hexdigest()
            for arm, _ in input_names:
                selected = [r for r in d['records'] if r['arm'] == arm]
                assert len(selected) == len(item['series']['r1:baseline'])
                item['series']['r4:'+arm] = []
                for r in selected:
                    rec = {k:r[k] for k in ('view','start','end','status','prompt','raw','finish_reason')}
                    rec.update(parsed=r.get('parsed'), error=r.get('error'),
                               frame_count=len(r['media']['video']['frames']),
                               native=r['native'], prompt_tokens=r['prompt_tokens'])
                    item['series']['r4:'+arm].append(rec)
        payload['input_comparison'] = dict(completed_samples=50, requests=879,
            native_timestamp_max_error_s=input_analysis['max_native_timestamp_rounding_error_s'],
            human_ground_truth=False)
    (OUT / 'data.json').write_text(json.dumps(payload, ensure_ascii=False, separators=(',', ':')) + '\n')
    shutil.copyfile(Path(__file__).with_name('listen_visual_prompt_versions_review.html'), OUT / 'index.html')
    shutil.copyfile(Path(__file__).with_name('listen_visual_prompt_review_audio.js'), OUT / 'audio.js')
    (OUT / 'build_audit.json').write_text(json.dumps({'samples': 50, 'series': len(payload['series']),
        'requests': sum(len(rs) for s in payload['samples'] for rs in s['series'].values()),
        'same_source_crop_start_duration': True, 'identical_input_frames_across_first_six_series': True,
        'descriptions_unmodified': True, 'source_sha256': hashes}, ensure_ascii=False, indent=2) + '\n')
    # Add a discoverable entry without replacing any historical review or embedded data.
    marker = 'id="visual-prompt-versions-20261029"'
    banner = ('<div ' + marker + ' style="padding:14px 20px;background:#e8f1ff;color:#163b65;font:16px system-ui">'
              '<a href="/listen_benchmark_review/prompt_versions_20261029/">打开：四版视觉 prompt · 同样本 50 条对照与人工复核提示（10月29日）</a>'
              ' · 本轮样本与历史原文整合的50条不同。</div>')
    for path in [BASE / 'index.html', BASE / 'existing_fusion_verbatim/index.html', BASE / 'semantic_prompt_pilot_20261027/index.html']:
        if not path.exists():
            continue
        text = path.read_text()
        if marker not in text:
            backup = path.with_name('index.before_visual_prompt_versions_20261029.html')
            if not backup.exists():
                shutil.copyfile(path, backup)
            import re
            text, n = re.subn(r'(<body\b[^>]*>)', lambda m: m[0] + banner, text, count=1, flags=re.I)
            if not n:
                # Historical pages use an implicit HTML body immediately after their stylesheet.
                text, n = re.subn(r'(</style>)', lambda m: m[0] + banner, text, count=1, flags=re.I)
            assert n == 1, path
            path.write_text(text)
    print(json.dumps({'output': str(OUT), 'samples': 50, 'requests': sum(len(rs) for s in payload['samples'] for rs in s['series'].values())}, ensure_ascii=False))


if __name__ == '__main__':
    main()
