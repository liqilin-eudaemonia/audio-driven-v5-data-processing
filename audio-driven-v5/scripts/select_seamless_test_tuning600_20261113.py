"""Extend the frozen Seamless256 subset to 600 using the same sampling rules."""
import collections
import csv
import hashlib
import json
from pathlib import Path
import random
import shutil
import time

from select_seamless_test_tuning256_20261113 import (
    label, read, rows, save, sha, stratum, summary,
)

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / 'experiments/seamless_test_tuning600_20261113'
RETAINED = ROOT / 'experiments/seamless_test_tuning256_20261113'
SEED = 20261113
N = 600


def choose(sources):
    counts = collections.Counter(map(stratum, sources))
    quotas = {k: int(v * N / len(sources)) for k, v in counts.items()}
    order = sorted(counts, key=lambda k: (-(counts[k] * N / len(sources) % 1), k))
    for k in order[:N - sum(quotas.values())]:
        quotas[k] += 1
    old = rows(RETAINED / 'selection.jsonl')
    by_id = {r['sample_id']: r for r in sources}
    selected = [by_id[r['id']] for r in old]
    assert len(selected) == len({r['sample_id'] for r in selected}) == 256
    existing = collections.Counter(map(stratum, selected))
    assert all(existing[k] <= quotas[k] for k in existing)
    used_groups = collections.Counter(r['group_id'] for r in selected)
    used_pairs = {r['pair'] for r in selected}
    assert len(used_pairs) == 256
    buckets = {k: [] for k in counts}
    for r in sorted(sources, key=lambda r: r['sample_id']):
        buckets[stratum(r)].append(r)
    rng = random.Random(SEED)
    for k in sorted(buckets):
        rng.shuffle(buckets[k])
    priorities = {r['sample_id']: i for k in sorted(buckets)
                  for i, r in enumerate(buckets[k])}
    schedule = [k for k in sorted(quotas)
                for _ in range(quotas[k] - existing[k])]
    rng.shuffle(schedule)
    for k in schedule:
        available = [r for r in buckets[k] if r['pair'] not in used_pairs]
        if not available:
            raise ValueError('Cannot satisfy quotas with unique original source pairs')
        r = min(available, key=lambda r: (
            used_groups[r['group_id']], priorities[r['sample_id']]))
        selected.append(r)
        used_pairs.add(r['pair'])
        used_groups[r['group_id']] += 1
    return selected, quotas


def main():
    current = read(ROOT / 'evaluation/SEAMLESS_CURRENT.json')
    parent = Path(current['prepared_full6624']['prepared'])
    parent_seal = read(parent / 'seal.json')
    retained_seal = read(RETAINED / 'prepared_256/seal.json')
    assert sha(parent / 'references.jsonl') == parent_seal['references_sha256']
    assert sha(parent / 'source.test.jsonl') == parent_seal['source_sha256']
    assert sha(RETAINED / 'selection.jsonl') == retained_seal['selection_sha256']
    assert sha(RETAINED / 'prepared_256/references.jsonl') == retained_seal['references_sha256']
    assert retained_seal['parent_references_sha256'] == sha(parent / 'references.jsonl')
    sources = rows(parent / 'source.test.jsonl')
    refs = rows(parent / 'references.jsonl')
    assert len(sources) == len(refs) == 6624
    assert all(r['split'] == 'test' for r in sources)
    assert {r['id'] for r in refs} == {r['sample_id'] for r in sources}
    selected, quotas = choose(sources)
    ids = [r['sample_id'] for r in selected]
    assert ids == [r['sample_id'] for r in choose(sources)[0]]
    assert len(ids) == len(set(ids)) == len({r['pair'] for r in selected}) == N
    assert ids[:256] == [r['id'] for r in rows(RETAINED / 'selection.jsonl')]
    assert collections.Counter(map(stratum, selected)) == quotas
    counts = collections.Counter(map(stratum, sources))
    assert all(abs(quotas[k] - counts[k] * N / len(sources)) < 1 for k in quotas)
    if (WORK / 'COMPLETE.json').exists():
        complete = read(WORK / 'COMPLETE.json')
        assert [r['id'] for r in rows(WORK / 'selection.jsonl')] == ids
        for name, digest in complete['files_sha256'].items():
            assert sha(WORK / name) == digest, f'Frozen artifact changed: {name}'
        print('Existing frozen600 selection and artifact hashes match deterministic recomputation')
        return
    WORK.mkdir(parents=True, exist_ok=True)
    prepared = WORK / 'prepared_600'
    prepared.mkdir(exist_ok=True)
    shutil.copyfile(parent / 'crop_config.json', prepared / 'crop_config.json')
    maps = {name: {r['id']: r for r in rows(parent / name)} for name in (
        'references.jsonl', 'inference_inputs.jsonl', 'annotations.jsonl',
        'generation_map.template.jsonl')}
    old_refs = {r['id']: r for r in rows(RETAINED / 'prepared_256/references.jsonl')}
    output = {name: [] for name in maps}
    selection, checks = [], []
    offset = 0
    with (prepared / 'source.test.jsonl').open('wb') as target, (parent / 'source.test.jsonl').open('rb') as source:
        for rank, r in enumerate(selected, 1):
            sid = r['sample_id']
            original = maps['references.jsonl'][sid]
            ref = original['gt_reference']
            source.seek(ref['offset'])
            raw = source.read(ref['bytes'])
            assert hashlib.sha256(raw).hexdigest() == ref['row_sha256']
            assert json.loads(raw) == r
            if sid in old_refs:
                assert old_refs[sid]['gt_reference']['row_sha256'] == ref['row_sha256']
            target.write(raw)
            newref = dict(ref, manifest=str(prepared / 'source.test.jsonl'),
                          offset=offset, crop_config=str(prepared / 'crop_config.json'))
            offset += len(raw)
            output['references.jsonl'].append(dict(original, gt_reference=newref))
            output['inference_inputs.jsonl'].append(dict(maps['inference_inputs.jsonl'][sid], gt_reference=newref))
            output['annotations.jsonl'].append(maps['annotations.jsonl'][sid])
            output['generation_map.template.jsonl'].append(maps['generation_map.template.jsonl'][sid])
            assert Path(r['inputs']['listener_first_frame']['source']['archive']).is_file()
            assert Path(r['inputs']['speaker_audio']['source_result_file']).is_file()
            assert sha(r['target']['source_file']) == r['target']['source_sha256']
            checks.append(dict(id=sid, parent_row_sha256=ref['row_sha256'],
                               source_archive_exists=True, source_result_exists=True,
                               caption_hash_verified=True))
            selection.append(dict(rank=rank, id=sid, split='test', purpose='checkpoint_tuning',
                pair=r['pair'], group_id=r['group_id'], genre=stratum(r)[0],
                listener_waveform_nonzero=stratum(r)[1],
                duration_bin=['2-4', '4-8', '8-11.95'][stratum(r)[2]],
                duration_s=r['duration_s'], source_start_s=original['source_start'],
                source_end_s=original['source_end'], speaker_id=original['speaker_id'],
                listener_id=original['listener_id'], face_p10_px=original['face_p10_px'],
                stratum=label(stratum(r)), reference_row_sha256=ref['row_sha256'],
                retained_from256=rank <= 256))
    for name, records in output.items():
        (prepared / name).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records))
    (WORK / 'selection.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in selection))
    with (WORK / 'selection.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(selection[0]))
        writer.writeheader()
        writer.writerows(selection)
    for name, subset in [('ids.txt', ids), ('retained256.ids.txt', ids[:256]), ('added344.ids.txt', ids[256:])]:
        (WORK / name).write_text(''.join(sid + '\n' for sid in subset))
    save(prepared / 'seal.json', dict(schema='seamless-approved-subset-v1', dataset='seamless',
        expected_clips=N, parent_prepared=str(parent), parent_seal_sha256=sha(parent / 'seal.json'),
        parent_references_sha256=sha(parent / 'references.jsonl'),
        selected_ids_sha256=hashlib.sha256(json.dumps(sorted(ids), separators=(',', ':')).encode()).hexdigest(),
        references_sha256=sha(prepared / 'references.jsonl'), source_sha256=sha(prepared / 'source.test.jsonl'),
        crop_config_sha256=sha(prepared / 'crop_config.json'), selection_sha256=sha(WORK / 'selection.jsonl'),
        source_version=current['version'], purpose='test_checkpoint_tuning_not_independent_dev'))
    full, sub = summary(sources), summary(selected)
    old200 = {r['id'] for r in rows(Path(current['prepared_200']['prepared']) / 'references.jsonl')}
    policy = ('Hamilton proportional quotas for genre x waveform-zero/nonzero x duration; '
              'seeded random priority within cells; retain original256 including original64; '
              'prefer least represented source group; one clip per original pair; no model outcomes read')
    report = dict(full_test=full, selected600=sub, seed=SEED,
                  stratum_quotas={label(k): v for k, v in sorted(quotas.items())},
                  retained256=256, added344=344, selection_policy=policy,
                  old_diagnostic200_overlap_ids=[sid for sid in ids if sid in old200],
                  audio_definition='Processed waveform zero/nonzero, not human speech ground truth')
    save(WORK / 'DISTRIBUTION.json', report)
    save(WORK / 'PROTOCOL.json', dict(seed=SEED, count=N, parent=str(parent),
        source_version=current['version'], selection_script=str(Path(__file__).resolve()),
        selection_script_sha256=sha(__file__),
        helper_script_sha256=sha(ROOT / 'scripts/select_seamless_test_tuning256_20261113.py'),
        parent_reference_sha256=sha(parent / 'references.jsonl'),
        parent_source_sha256=sha(parent / 'source.test.jsonl'),
        retained_selection=str(RETAINED / 'selection.jsonl'),
        retained_selection_sha256=sha(RETAINED / 'selection.jsonl'),
        selected_ids_sha256=sha(WORK / 'ids.txt'), selection_sha256=sha(WORK / 'selection.jsonl'),
        quotas=report['stratum_quotas'], policy=policy, purpose='test_checkpoint_tuning',
        retained256=256, added344=344, independent_dev=False, does_not_launch_inference=True))
    save(WORK / 'SELECTION_AUDIT.json', dict(success=True, count=N, retained256=256, added344=344,
        all_from_current_full_test=True, deterministic_reselection_equal=True,
        retained256_order_and_gt_unchanged=True, each_stratum_rounding_error_lt_one=True,
        unique_original_pairs=N, source_groups=sub['source_groups'],
        max_per_group=sub['max_clips_per_group'], checks=checks, full_media_decode_performed=False))
    lines = ['# Seamless test 固定600条评测子集', '',
        '来自当前转正版test6624，保留原256条（含原64条）并新增344条；固定随机种子20261113。',
        '按场景×净化listener音轨全零/非零×时长共12层，以最大余数法按全集比例分配配额；',
        '层内固定随机优先级，优先当前入选较少的来源组，每个原始视频对只取一段，不读取预测或评分。', '',
        '**这是test权重筛选子集，不是独立dev。** 完整test6624包含这些调参样本。',
        '原GT行、裁框、时间、角色、首帧及音频配方保持；这里只重建清单引用，不复制媒体。', '',
        '| 项目 | 完整test6624 | 固定600 |', '|---|---:|---:|']
    for field, title in [('genre', '场景'), ('waveform', '音轨'), ('duration', '时长')]:
        for k in sorted(full[field]):
            lines.append(f'| {title}：{k} | {full[field][k]}（{full[field][k]/6624:.2%}） | {sub[field].get(k, 0)}（{sub[field].get(k, 0)/N:.2%}） |')
    lines += [f'| 平均时长 | {full["mean_duration_s"]:.3f}s | {sub["mean_duration_s"]:.3f}s |', '',
        f'600条来自600个原始视频对、{sub["source_groups"]}个来源组，合计{sub["total_duration_s"]/60:.2f}分钟；',
        f'每来源组最多{sub["max_clips_per_group"]}条，与旧固定200重叠{len(report["old_diagnostic200_overlap_ids"])}条。音轨非零不等于人工确认有语音。', '',
        '- [逐条CSV](selection.csv)、[冻结机器清单](selection.jsonl)、[全部ID](ids.txt)。',
        '- [新增344条ID](added344.ids.txt)、[保留256条ID](retained256.ids.txt)。',
        '- [评测参考](prepared_600/references.jsonl)、[生成输入](prepared_600/inference_inputs.jsonl)。',
        '- [生成映射模板](prepared_600/generation_map.template.jsonl)、[seal](prepared_600/seal.json)。',
        '- [协议](PROTOCOL.json)、[分布](DISTRIBUTION.json)、[抽样核验](SELECTION_AUDIT.json)。',
        '- [评测框架接入核验](FRAMEWORK_ACCEPTANCE.json)、[机器入口](../../evaluation/SEAMLESS_TUNING600.json)。', '',
        '## 读取和评测', '',
        '既有OmniResponse Seamless binder已经支持sealed子集，直接把`--prepared`设为本目录的`prepared_600`。',
        '生成时调用`evaluation.seamless.reader.load_conditioning(gt_reference)`，只取speaker音频与listener首帧；',
        'GT视频和净化listener音轨分别使用原`decode_reference`和`load_listener_audio`，不把未来GT或caption传入生成。', '',
        '在OmniResponse根目录，复制映射模板到独立运行目录并填好模型生成路径后，沿用完整评测入口：', '',
        '```bash', 'cd /private/liqi_workspace/project_code/OmniResponse',
        'export PYTHONPATH=/private/liqi_workspace/.cache/paired_video_metrics/python',
        'export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2',
        '/private/software/conda/envs/AnyTalker/bin/python -m evaluation.listener_metrics.suite \\',
        '  --dataset seamless \\', f'  --prepared {prepared} \\',
        '  --generated-map /path/to/filled_generation_map.jsonl \\',
        '  --output /path/to/new_seamless600_scores \\', '  --device cpu --batch-size 16', '```', '',
        '视频指标也可沿用`evaluation.seamless.bind`→`evaluation.video.paired_metrics.run`，同样指定这份prepared。',
        '所有比较必须使用相同600条；FID/FVD按全600条特征汇总，不平均分片分数。GLARE只有视频，音频指标保持N/A。',
        '旧256条结果仅在模型、seed、来源、规划及采样身份完全相同时才能复用，600条分布指标需要重新全量汇总。', '',
        '## 重算与验证', '',
        '在audio-driven-v5根目录执行；已有COMPLETE时仅核对确定性和冻结文件SHA，不覆盖：', '',
        '```bash', 'python scripts/select_seamless_test_tuning600_20261113.py',
        'PYTHONPATH=/private/liqi_workspace/.cache/paired_video_metrics/python \\',
        '  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="" \\',
        '  /private/software/conda/envs/AnyTalker/bin/python scripts/check_seamless_test_tuning600_20261113.py',
        '```', '', '本次只准备数据并验证读取，不启动模型生成或评分，不改动现有256/6624及正在运行的队列。']
    (WORK / 'README.md').write_text('\n'.join(lines) + '\n')
    names = ['selection.jsonl', 'selection.csv', 'ids.txt', 'retained256.ids.txt', 'added344.ids.txt',
             'PROTOCOL.json', 'DISTRIBUTION.json', 'SELECTION_AUDIT.json', 'README.md']
    names += ['prepared_600/' + name for name in [*output, 'source.test.jsonl', 'crop_config.json', 'seal.json']]
    save(WORK / 'COMPLETE.json', dict(success=True, at=time.time(), selected=N,
        selection_sha256=sha(WORK / 'selection.jsonl'), prepared=str(prepared),
        source_version=current['version'], purpose='test_checkpoint_tuning',
        files_sha256={name: sha(WORK / name) for name in names}, inference_started=False, scoring_started=False))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
