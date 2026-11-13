#!/usr/bin/env python3
"""Table-only inventory of retained descriptions and previously unmapped visual evidence.

Descriptions are exact source strings, not a new closed taxonomy. No label, media,
admission, or cache is changed. Observation counts are not physical action counts.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from listen_benchmark.backchannel_face_stats import loads
from p2_listen.contracts import atomic_json, sha

NAMES = dict(head_nod='点头', head_shake='摇头', smile='微笑', laugh='大笑／笑声',
             frown='皱眉', surprise='惊讶', listener_vocalization='音频回应／发声')
LAYERS = {'retained': '后处理整合结果', 'unassigned_visual': '未纳入七类的视觉证据'}


def union(spans):
    result = []
    for a, b in sorted(spans):
        if result and a <= result[-1][1]:
            result[-1][1] = max(result[-1][1], b)
        else:
            result.append([a, b])
    return result


def intervals(event, duration):
    values = event.get('intervals', [{'start': event['start'], 'end': event['end']}])
    spans = []
    for value in values:
        a, b = value['start'], value['end']
        if not all(isinstance(x, (int, float)) and math.isfinite(x) for x in (a, b)):
            raise ValueError('Nonfinite event interval')
        if not 0 <= a < b <= duration + .0011:
            raise ValueError(('Out-of-clip interval', a, b, duration))
        spans.append((a, min(b, duration)))
    if not spans:
        raise ValueError('Empty retained support')
    return union(spans)


def new_stat():
    return dict(records=0, samples=0, union_seconds=0., event_seconds=0.,
                duration_ms=Counter(), previous_categories=Counter(), previous_status=Counter())


def add_event(stats, key, event, spans, example):
    s = stats.setdefault(key, new_stat())
    duration = sum(b-a for a, b in spans)
    s['records'] += 1
    s['event_seconds'] += duration
    s['duration_ms'][round(duration * 1000)] += 1
    s['previous_categories'].update(event.get('categories', []))
    s['previous_status'][event.get('mapping_status', 'retained')] += 1
    s.setdefault('example', example)


def scan(work):
    layer, key, paths, output, fusion = work
    out = Path(output) / 'shards'
    marker = out / f'{key}.done.json'
    if marker.exists():
        m = loads(marker.read_bytes())
        for name, checksum in m['outputs'].items():
            if sha(out / name) != checksum:
                raise ValueError('Changed report shard')
        return m
    stats = {}; cats = {}; totals = Counter(); source_hashes = {}
    event_path = out / f'{key}.events.jsonl.gz'
    ids_path = out / f'{key}.ids.txt'
    summary_path = out / f'{key}.stats.jsonl'
    with gzip.open(event_path, 'wt', encoding='utf-8', compresslevel=1) as events_out, ids_path.open('w') as ids_out:
        for path in paths:
            path = Path(path)
            source_hashes[str(path)] = sha(path)
            if layer == 'retained':
                rows = ((loads(line), None) for line in path.open('rb'))
            else:
                label_path = Path(fusion) / 'labels' / path.name
                source_hashes[str(label_path)] = sha(label_path)
                # Original fusion kept the same visual events and unresolved IDs.
                # Audio is read exclusively from postprocessed integrated files.
                originals = {r['sample_id']: r for r in (loads(line) for line in label_path.open('rb'))}
                rows = ((originals.pop(e['sample_id']), e) for e in (loads(line) for line in path.open('rb')))
            for label, evidence in rows:
                sid = label['sample_id']; duration = label['duration_s']
                if layer == 'retained':
                    selected = label['events']
                else:
                    lookup = {e['event_id']: e for e in evidence['original_events']}
                    ids = [o['observation_id'] for o in label['unresolved_observations']]
                    if len(ids) != len(set(ids)):
                        raise ValueError('Duplicate unresolved observation ID')
                    selected = [lookup[i] for i in ids]
                    if any(e['modality'] != 'visual' for e in selected):
                        raise ValueError('Unfiltered audio must not enter expanded report')
                ids_out.write(sid + '\t' + str(len(selected)) + '\n')
                totals['samples'] += 1; totals['records'] += len(selected)
                totals['samples_with_records'] += bool(selected)
                per_description = defaultdict(list); per_category = defaultdict(list); all_spans = []
                source_offset = label['source']['source_start_s']
                for e in selected:
                    description = e['description'] if layer == 'retained' else e['behavior']
                    if not isinstance(description, str) or not description:
                        raise ValueError('Missing original description')
                    modality = e['modality']; spans = intervals(e, duration)
                    type_key = (layer, modality, description)
                    type_id = hashlib.sha256((modality + '\0' + description).encode()).hexdigest()
                    example = dict(sample_id=sid, event_id=e['event_id'], intervals=spans)
                    add_event(stats, type_key, e, spans, example)
                    per_description[type_key].extend(spans); all_spans.extend(spans)
                    if layer == 'retained':
                        for category in set(e.get('categories', [])):
                            add_event(cats, category, e, spans, example)
                            per_category[category].extend(spans)
                    totals[modality + '_records'] += 1
                    detail = dict(sample_id=sid, event_id=e['event_id'], layer=layer,
                        original_description_id=type_id, description=description, modality=modality,
                        previous_categories=e.get('categories', []),
                        previous_mapping_status=e.get('mapping_status', 'retained'),
                        intervals=spans, source_intervals=[[source_offset+a, source_offset+b] for a,b in spans],
                        source_start_s=source_offset, source_media=label['source'].get('source'),
                        source_label_file=str(path if layer == 'retained' else label_path),
                        original_record=e, human_verified=False)
                    events_out.write(json.dumps(detail, ensure_ascii=False, separators=(',', ':')) + '\n')
                for k, spans in per_description.items():
                    stats[k]['samples'] += 1
                    stats[k]['union_seconds'] += sum(b-a for a,b in union(spans))
                for k, spans in per_category.items():
                    cats[k]['samples'] += 1
                    cats[k]['union_seconds'] += sum(b-a for a,b in union(spans))
                totals['union_seconds'] += sum(b-a for a,b in union(all_spans))
            if layer != 'retained' and originals:
                raise ValueError('Missing original evidence rows')
    with summary_path.open('w') as f:
        for k, s in stats.items():
            f.write(json.dumps(dict(key=k, **s), ensure_ascii=False, separators=(',', ':')) + '\n')
    result = dict(key=key, layer=layer, totals=dict(totals), classes=cats, sources=source_hashes,
                  outputs={p.name: sha(p) for p in (event_path, ids_path, summary_path)})
    atomic_json(marker, result)
    return result


def combine(a, b):
    for k in ('records', 'samples', 'union_seconds', 'event_seconds'):
        a[k] += b[k]
    for k in ('duration_ms', 'previous_categories', 'previous_status'):
        a[k].update(b[k])
    if 'example' not in a:
        a['example'] = b['example']


def quantile(hist, q):
    n = sum(hist.values())
    rank = max(1, math.ceil(n*q)); c = 0
    for value, count in sorted((int(k), v) for k,v in hist.items()):
        c += count
        if c >= rank:
            return value / 1000


HEADERS = ['层级', '模态', '原文描述ID', '原文描述', '涉及片段数', '标注条目数', '区间并集秒', '区间并集小时',
           '条目平均持续秒', '条目持续P10秒', '条目持续P50秒', '条目持续P90秒', '原七类归属计数', '原映射状态计数',
           '示例sample_id', '示例event_id', '示例起止时间秒']


def csv_row(key, s):
    layer, modality, description = key
    return [LAYERS[layer], modality, hashlib.sha256((modality+'\0'+description).encode()).hexdigest(),
            description, s['samples'], s['records'], round(s['union_seconds'], 6),
            round(s['union_seconds']/3600, 9), round(s['event_seconds']/s['records'], 6),
            quantile(s['duration_ms'], .1), quantile(s['duration_ms'], .5), quantile(s['duration_ms'], .9),
            json.dumps(s['previous_categories'], ensure_ascii=False), json.dumps(s['previous_status'], ensure_ascii=False),
            s['example']['sample_id'], s['example']['event_id'], json.dumps(s['example']['intervals'])]


def aggregate(output, manifest):
    out = Path(output); all_stats = {}; classes = {}; totals = defaultdict(Counter)
    scope_db = sqlite3.connect(f"file:{manifest['postprocess']}/scope.sqlite?mode=ro", uri=True)
    expected = {r[0] for r in scope_db.execute('select sid from samples')}; scope_db.close()
    seen = {k: set() for k in LAYERS}
    present = {k: set() for k in LAYERS}
    for marker in sorted((out/'shards').glob('*.done.json')):
        m = loads(marker.read_bytes()); layer = m['layer']; key = m['key']
        totals[layer].update(m['totals'])
        for row in (out/'shards'/f'{key}.ids.txt').read_text().splitlines():
            sid, count = row.split('\t')
            if sid in seen[layer]:
                raise ValueError('Duplicate sample in layer')
            seen[layer].add(sid)
            if int(count):
                present[layer].add(sid)
        for line in (out/'shards'/f'{key}.stats.jsonl').open('rb'):
            s = loads(line); k = tuple(s.pop('key'))
            combine(all_stats.setdefault(k, new_stat()), s)
        for k, s in m['classes'].items():
            combine(classes.setdefault(k, new_stat()), s)
    if any(ids != expected for ids in seen.values()):
        raise ValueError('Full corpus scope mismatch')
    previous = loads(Path(manifest['baseline_report'], 'summary.json').read_bytes())
    for c, s in classes.items():
        old = previous['classes']['open'][c]
        if s['records'] != old['events'] or s['samples'] != old['samples'] or not math.isclose(s['union_seconds'], old['union_seconds'], abs_tol=.001):
            raise ValueError(('Seven-class reconciliation failed', c))
    if totals['retained']['records'] != previous['totals']['open_events']:
        raise ValueError('Retained event count differs')
    if totals['unassigned_visual']['records'] != previous['totals']['unresolved_observations']:
        raise ValueError('Unassigned visual evidence count differs')
    top = {}
    for layer in LAYERS:
        ordered = sorted(((k,s) for k,s in all_stats.items() if k[0] == layer), key=lambda x:(-x[1]['samples'], x[0]))
        with (out/f'{layer}_全部原文类型.csv').open('w', encoding='utf-8-sig', newline='') as f:
            writer = csv.writer(f); writer.writerow(HEADERS)
            writer.writerows(csv_row(k,s) for k,s in ordered)
        top[layer] = [dict(layer=k[0], modality=k[1], description=k[2], **s) for k,s in ordered[:100]]
        totals[layer]['unique_verbatim_types'] = len(ordered)
    with (out/'七类时间对照.csv').open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f); w.writerow(['类别','片段数','条目数','区间并集秒','区间并集小时','平均持续秒','P10秒','P50秒','P90秒'])
        for k in NAMES:
            s = classes[k]
            w.writerow([NAMES[k],s['samples'],s['records'],round(s['union_seconds'],6),round(s['union_seconds']/3600,6),
                        round(s['event_seconds']/s['records'],6),quantile(s['duration_ms'],.1),quantile(s['duration_ms'],.5),quantile(s['duration_ms'],.9)])
    all_present = set.union(*present.values())
    summary = dict(scope=previous['scope'], layers=totals, classes=classes, top_original_descriptions=top,
                   samples_with_any_retained_or_additional_observation=len(all_present),
                   samples_without_any_retained_or_additional_observation=len(expected-all_present),
                   total_distinct_verbatim_types=len({k[1:] for k in all_stats}), no_new_semantic_taxonomy=True,
                   all_samples_retained=True, source_text_rewritten=False, source_labels_changed=False,
                   charts_generated=False, duration_quantile_resolution_seconds=.001)
    atomic_json(out/'summary.json', summary)
    # A queryable catalog avoids spreadsheet limits and keeps every exact source string.
    db = sqlite3.connect(out/'description_catalog.sqlite')
    db.execute('CREATE TABLE descriptions(layer TEXT,modality TEXT,description TEXT,samples INT,records INT,union_seconds REAL,statistics_json TEXT,PRIMARY KEY(layer,modality,description))')
    db.executemany('INSERT INTO descriptions VALUES(?,?,?,?,?,?,?)',
                   ((k[0],k[1],k[2],s['samples'],s['records'],s['union_seconds'],json.dumps(s,ensure_ascii=False)) for k,s in all_stats.items()))
    db.commit(); db.close()
    write_readme(out, summary)
    atomic_json(out/'COMPLETE.json', dict(status='passed', samples=len(expected), layers=totals,
        validation='both exact sample ID sets; seven-class counts and union time; all unresolved visual observations',
        report_files={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name not in ('COMPLETE.json','progress.json')}))


def write_readme(out, s):
    lines = ['# 全量开放描述与事件时间统计（无图）', '',
        '范围：722,706 段，全部保留，包括无七类事件及原先未获训练准入的记录。原始标签、准入标记和缓存不变。', '',
        '本报告不再把七类当作全部行为。按 **模态＋逐字保留的原文描述** 建立开放目录；同义句、大小写和不同详细程度仍可能分列，不能把原文类型数当成语义类别数。', '',
        '现有模型只完成了七类语义归类，没有“所有语义类别”的可靠统一分类。本次完整列出原文，不用关键词硬造新类别。额外视觉记录是模型观察候选，可能是普通动作、状态或跨窗重复；不自动认定为 backchannel。', '',
        '| 统计层 | 涉及片段数 | 标注条目数 | 不同原文描述数 |', '|---|---:|---:|---:|']
    for layer,t in s['layers'].items():
        lines.append(f"| {LAYERS[layer]} | {t['samples_with_records']:,} | {t['records']:,} | {t['unique_verbatim_types']:,} |")
    lines += ['', f"至少有一条上述记录：{s['samples_with_any_retained_or_additional_observation']:,} 段；两层均无记录：{s['samples_without_any_retained_or_additional_observation']:,} 段。这是有无模型观察，不是有无真实回应的判定。", '', '## 数据入口', '',
        '- [后处理整合结果：全部原文类型与时长](retained_全部原文类型.csv)',
        '- [额外视觉证据：全部原文类型与时长](unassigned_visual_全部原文类型.csv)',
        '- [原七类时间对照](七类时间对照.csv)',
        '- [可查询的完整原文目录](description_catalog.sqlite)',
        '- [汇总及高频原文](summary.json)',
        '- `shards/*.events.jsonl.gz`：逐事件起止时间，支持流式读取；`retained-*` 为整合后结果，`evidence-*` 为未纳入七类的视觉证据。', '',
        '## 时间口径', '',
        '- `intervals`：相对当前片段的起止秒；`source_intervals`：相对原始源视频的起止秒。分离的音频保留区间保持分离，不用外包区间填补静音。',
        '- 每一原文类型、每一片段先求时间并集，再累计为该类型覆盖时长。同一条记录跨越多个保留区间时，持续时长为这些区间之和。',
        '- 原七类表同样在片段内按类求并集；不同类别可以同时发生，各类别时长不能直接相加当总时长。',
        '- 原文目录同义描述可能重叠，目录各行时长也不能简单相加。P10/P50/P90 是标注条目持续时长，使用最近秩定义、1 毫秒分辨率；不代表边界具有毫秒级准确度。',
        '- 条目数不是物理动作次数；额外视觉证据没有重新跨窗整合。统计没有把它们强行并入训练标签。',
        '- 音频只使用后处理保留结果，没有复活被静音规则排除的音频标注。视觉保留原证据中的所有未纳入条目。', '',
        '## 原七类时间对照（用于与旧报告核对，不是类别上限）', '',
        '| 类别 | 覆盖片段数 | 累计覆盖小时 | 条目持续 P50 秒 |', '|---|---:|---:|---:|']
    for k,name in NAMES.items():
        c=s['classes'][k];lines.append(f"| {name} | {c['samples']:,} | {c['union_seconds']/3600:.3f} | {quantile(c['duration_ms'],.5):.3f} |")
    for layer in LAYERS:
        lines += ['', f'## {LAYERS[layer]}：高频原文示例', '', '| 原文描述 | 片段数 | 区间并集小时 |', '|---|---:|---:|']
        for r in s['top_original_descriptions'][layer][:25]:
            desc=r['description'].replace('|','\\|').replace('\n',' ')
            lines.append(f"| {desc} | {r['samples']:,} | {r['union_seconds']/3600:.3f} |")
    lines += ['', '## 复现', '', '```bash', 'python scripts/report_open_backchannel_times.py \\',
        '  --postprocess experiments/p2_full_audio_postprocess_crop384_20261026 \\',
        '  --fusion experiments/listen_full_fusion_crop384_20261025 \\',
        '  --baseline-report experiments/p2_full_backchannel_report_20261027 \\',
        f'  --output experiments/{out.name} --workers 16', '```', '',
        '仅 CPU 统计，不运行模型，不生成图。输入 SHA、逐分片输出 SHA 和集合核对保存在 manifest / shards / COMPLETE。']
    (out/'README.md').write_text('\n'.join(lines)+'\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('--postprocess',type=Path,required=True)
    p.add_argument('--fusion',type=Path,required=True);p.add_argument('--baseline-report',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=16)
    a=p.parse_args(); out=a.output.resolve();out.mkdir(parents=True,exist_ok=True);(out/'shards').mkdir(exist_ok=True)
    manifest=dict(postprocess=str(a.postprocess.resolve()),fusion=str(a.fusion.resolve()),baseline_report=str(a.baseline_report.resolve()),
        script_sha256=sha(Path(__file__)),postprocess_complete_sha256=sha(a.postprocess/'COMPLETE.json'))
    if (out/'manifest.json').exists() and loads((out/'manifest.json').read_bytes()) != manifest:
        raise ValueError('Report identity changed; use a new output directory')
    atomic_json(out/'manifest.json',manifest)
    if (out/'COMPLETE.json').exists():
        print('Report already complete');return
    paths=sorted((a.postprocess/'integrated').glob('*.jsonl'))
    work=[('retained',f'retained-{i//32:04d}',[str(p) for p in paths[i:i+32]],str(out),str(a.fusion)) for i in range(0,len(paths),32)]
    work += [('unassigned_visual',f'evidence-{p.stem}',[str(p)],str(out),str(a.fusion)) for p in sorted((a.fusion/'evidence').glob('*.jsonl'))]
    started=time.time()
    with ProcessPoolExecutor(a.workers) as pool:
        futures=[pool.submit(scan,w) for w in work]
        for i,future in enumerate(as_completed(futures),1):
            future.result()
            if i%20==0 or i==len(work):
                state=dict(completed_shards=i,total_shards=len(work),elapsed_seconds=time.time()-started)
                atomic_json(out/'progress.json',state);print(json.dumps(state),flush=True)
    aggregate(out,manifest)
    print('Complete:',out,flush=True)


if __name__ == '__main__':
    main()
