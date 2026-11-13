"""Portable random sample for annotation-code testing; production data stays unchanged."""
import concurrent.futures
import csv
import hashlib
import json
import random
import shutil
import subprocess
from pathlib import Path
import zipfile
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data_corpora/seamless_listen_local_full/manifests/train_primary_keep_candidate.jsonl'
OUT = ROOT / 'data_exports/seamless_annotation_sample50_20261023'
SEED = 2026102301


def write(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def command(args):
    p = subprocess.run(args, capture_output=True, text=True, timeout=600)
    if p.returncode:
        raise RuntimeError(p.stderr[-6000:])
    return p.stdout


def probe(path):
    return json.loads(command(['ffprobe', '-v', 'error', '-show_streams', '-show_format', '-of', 'json', str(path)]))


def uri(ref):
    return f"subfile,,start,{ref['offset']},end,{ref['offset']+ref['byte_size']},,:{ref['archive']}"


def export(task):
    ordinal, row = task
    relative = Path('samples') / f"{ordinal:03d}_{row['sample_id']}"
    folder = OUT / relative
    folder.mkdir(parents=True, exist_ok=True)
    write(folder / 'original_record.json', row)
    evidence = json.loads(Path(row['quality_evidence_file']).read_text())
    original = next(r for r in evidence['clips'] if r['sample_id'] == row['sample_id'])
    write(folder / 'quality_evidence.json', {k: original[k] for k in ('qc3', 'qc4') if k in original})
    sources = {role: probe(uri(row['sources'][role])) for role in ('speaker', 'listener')}
    limit = min(float(s['format']['duration']) for s in sources.values())
    for role in sources:
        with sf.SoundFile(row['normalized_audio'][role]['path']) as wav:
            limit = min(limit, len(wav) / wav.samplerate)
    assert row['end'] <= limit + .04
    context_start, context_end = max(0., row['start'] - 3), min(limit, row['end'] + 3)
    item = dict(ordinal=ordinal, sample_id=row['sample_id'], pair=row['pair'], split=row['split'],
                admission_pool=row['admission_pool'], audio_pool=row['audio_final']['pool'],
                source_start_s=row['start'], source_end_s=row['end'], duration_s=row['duration_s'],
                context_source_start_s=context_start, context_source_end_s=context_end,
                segment_in_context_s=[row['start']-context_start, row['end']-context_start],
                transcript=row['text'], boundary_review_flags=row.get('boundary_review_flags', []),
                audio_final=row['audio_final'], face_geometry=row['face_geometry'],
                crop_proposals=row['crop_proposals'], roles={},
                metadata_file=str(relative/'original_record.json'),
                quality_evidence_file=str(relative/'quality_evidence.json'))
    for role in ('speaker', 'listener'):
        source_video = next(s for s in sources[role]['streams'] if s['codec_type'] == 'video')
        paths = {}
        checks = {}
        for mode, start, end in [('clip', row['start'], row['end']), ('context', context_start, context_end)]:
            # PCM sample boundaries are recorded explicitly; video may differ by <= one frame.
            with sf.SoundFile(row['normalized_audio'][role]['path']) as f:
                rate = f.samplerate
                a, b = round(start*rate), round(end*rate)
                f.seek(a)
                wave = f.read(b-a, dtype='int32')
            assert len(wave) == b-a
            wavpath = folder / f'{role}_{mode}.wav'
            sf.write(wavpath, wave, rate, subtype='PCM_24')
            video = folder / f'{role}_{mode}.mp4'
            command(['ffmpeg', '-v', 'error', '-nostdin', '-y', '-threads', '1', '-filter_threads', '1',
                     '-noautorotate', '-ss', str(start), '-i', uri(row['sources'][role]),
                     '-i', str(wavpath), '-t', str(end-start), '-map', '0:v:0', '-map', '1:a:0',
                     '-vf', 'setpts=PTS-STARTPTS', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '20',
                     '-threads', '2', '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', '192k',
                     '-map_metadata', '-1', '-metadata:s:v:0', 'rotate=0', '-movflags', '+faststart', str(video)])
            check = probe(video)
            v = next(s for s in check['streams'] if s['codec_type'] == 'video')
            audio = next(s for s in check['streams'] if s['codec_type'] == 'audio')
            assert (v['width'], v['height']) == (source_video['width'], source_video['height'])
            numerator, denominator = v['avg_frame_rate'].split('/')
            fps = float(numerator)/float(denominator)
            assert abs(float(v['duration'])-(end-start)) <= max(.085, 1/fps+.025)
            assert abs(float(audio['duration'])-(end-start)) <= .03
            command(['ffmpeg', '-v', 'error', '-xerror', '-threads', '1', '-i', str(video),
                     '-map', '0:v:0', '-map', '0:a:0', '-f', 'null', '-'])
            paths[mode+'_video'] = str(relative/video.name)
            if mode == 'clip':
                paths['audio_wav'] = str(relative/wavpath.name)
            else:
                wavpath.unlink()
            checks[mode] = dict(width=v['width'], height=v['height'], fps=fps,
                                video_duration_s=float(v['duration']), audio_duration_s=float(audio['duration']),
                                pcm_source_start_sample=a, pcm_source_end_sample=b, sample_rate=rate, decoded=True)
        item['roles'][role] = dict(person_id=row[role+'_id'], source_index=row[role+'_index'],
                                   **paths, media_checks=checks)
    write(folder/'annotation_item.json', item)
    print(json.dumps(dict(complete=ordinal, sample_id=row['sample_id'])), flush=True)
    return item


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    selected = OUT/'selected_original_records.jsonl'
    if not selected.exists():
        rng = random.Random(SEED)
        sample = []
        with SOURCE.open() as stream:
            for count, line in enumerate(stream, 1):
                if count <= 50:
                    sample.append((count, line))
                else:
                    slot = rng.randrange(count)
                    if slot < 50:
                        sample[slot] = (count, line)
        assert count == 1200212
        selected.write_text(''.join(line for _, line in sample))
        write(OUT/'sampling.json', dict(seed=SEED, method='uniform reservoir sampling without replacement over clips',
                                       source_manifest=str(SOURCE), population=count, sample_count=50,
                                       selected_line_numbers_1based=[i for i, _ in sample],
                                       visual_filter_applied=False, errors_or_rejects_sampled=False))
    rows = [json.loads(line) for line in selected.read_text().splitlines()]
    assert len(rows) == len({r['sample_id'] for r in rows}) == 50
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        items = list(pool.map(export, enumerate(rows, 1)))
    write(OUT/'items.json', items)
    (OUT/'manifest.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in items))
    (OUT/'labels_template.jsonl').write_text(''.join(json.dumps(dict(sample_id=r['sample_id'],
        audio_quality=None, speaker_leakage=None, listener_role_correct=None, boundary_ok=None,
        crop_ok=None, corrected_listener_crop=None, notes=''), ensure_ascii=False)+'\n' for r in items))
    with (OUT/'samples.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['ordinal','sample_id','pair','source_start_s','source_end_s','duration_s','transcript','listener_video','speaker_video','listener_crop_x','listener_crop_y','listener_crop_size'])
        writer.writeheader()
        for r in items:
            c=r['crop_proposals']['listener']
            writer.writerow(dict({k:r[k] for k in writer.fieldnames[:7]},
                listener_video=r['roles']['listener']['clip_video'], speaker_video=r['roles']['speaker']['clip_video'],
                **{'listener_crop_'+k:c.get(k) for k in ['x','y','size']}))
    write(OUT/'validation.json', dict(samples=len(items), unique_pairs=len({r['pair'] for r in items}),
        total_clip_seconds=sum(r['duration_s'] for r in items), videos=200, wav_files=100,
        all_videos_decoded=True, native_dimensions_verified=True, durations_verified=True,
        crops_unchanged=True, roles_and_source_intervals_unchanged=True,
        listener_crop_status=dict((status,sum(r['crop_proposals']['listener']['status']==status for r in items))
                                 for status in {r['crop_proposals']['listener']['status'] for r in items})))
    shutil.copy2(Path(__file__), OUT/'export_sample.py')
    (OUT/'README.md').write_text('''# Seamless 标注代码测试包：随机 50 条

从训练主池音质通过候选 1,200,212 段中均匀无放回抽取 50 段，固定种子 2026102301。没有按画面、人脸、裁框质量二次挑选；本包不包含专门抽取的拒绝样本，不能用于估计误拒率。

入口为 `items.json`（数组）或 `manifest.jsonl`（每行一条）；所有媒体路径均相对于本 README 所在目录，整包复制即可使用。`samples.csv` 为精简表格，`labels_template.jsonl` 为待填标签。

每条有 speaker/listener 各一份完整画幅片段 MP4、归一化单声道 PCM_24 WAV，以及边界前后各最多 3 秒的完整画幅 context MP4。MP4 内音频为对应人的归一化音频 AAC；精确音频测试请使用 WAV。没有叠加框或文字，也没有执行空间裁剪，适合打标程序自行绘框。

`roles.listener.clip_video` / `roles.speaker.clip_video`：切分片段，从本地 0 秒播放。`roles.*.audio_wav`：同区间音频。`roles.*.context_video`：前后上下文，`segment_in_context_s` 标出真正切分区间。`source_start_s/source_end_s` 是原始视频时间，不能直接用作本地片段时间。截取边界因视频帧离散最多约一帧误差；音频按 round(原时间×采样率) 截取，精确采样边界在 media_checks 内。

`crop_proposals.listener` / `speaker` 为本片段固定参考框；x、y 为对应完整画幅左上角原生像素坐标，size 为正方形边长。视频保持原生编码宽高，禁用自动旋转和缩放，因此坐标可直接绘制。缩放显示时应同步缩放坐标。参考框只针对正式片段生成，不保证覆盖额外上下文。status=proposal_only 不代表最终验收，unmeasured 时无可用坐标。fits_estimated_head 是几何估计，valid_fraction 是有效帧占比；尚未做人脸裁框准入。旧 listener_crop=null 字段不用。

每条 `original_record.json` 保存完整原清单字段（含原 TAR、成员名、偏移、归一化整轨音频路径、转写、角色、边界标记、人脸统计等）；`quality_evidence.json` 保存对应片段 QC3/QC4 证据。上述原始绝对路径仅供溯源，移机测试使用 items.json 中的相对媒体路径。转写是源数据文本，未经新增人工校正。

`sampling.json` 记录随机种子和来源行号；`validation.json` 记录媒体解码、尺寸、时长检查；`SHA256SUMS` 可用于复制完整性校验。`export_sample.py` 是本次导出脚本存档，重新导出需要原服务器依赖。
''')
    checks=[]
    for p in sorted(OUT.rglob('*')):
        if p.is_file() and p.name!='SHA256SUMS':
            checks.append(hashlib.file_digest(p.open('rb'),'sha256').hexdigest()+'  '+str(p.relative_to(OUT)))
    (OUT/'SHA256SUMS').write_text('\n'.join(checks)+'\n')
    archive=OUT.with_suffix('.zip')
    with zipfile.ZipFile(archive,'w',allowZip64=True) as z:
        for p in sorted(OUT.rglob('*')):
            if p.is_file():z.write(p,arcname=str(Path(OUT.name)/p.relative_to(OUT)),compress_type=zipfile.ZIP_STORED if p.suffix in ('.mp4','.wav') else zipfile.ZIP_DEFLATED)
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    print(json.dumps(dict(directory=str(OUT),archive=str(archive),bytes=archive.stat().st_size)),flush=True)


if __name__ == '__main__':
    main()
