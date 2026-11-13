"""Publish descriptive comparison only after all frozen requests are audited."""
import json
from pathlib import Path

R = Path(__file__).resolve().parent


def main():
    a = json.load(open(R / 'analysis.json'))
    assert a['all_samples_complete'] and (R / 'COMPLETE.json').exists()
    names = {'image8': '多图＋时间戳 8fps', 'video8': '原生视频 8fps', 'video16': '原生视频 16fps'}
    pct = lambda x, n: f'{100*x/n:.1f}%' if n else '—'
    lines = ['# C prompt：多图与原生视频输入的50条对照结果', '',
        '已完成50/50条、879次请求。三组同样本、同prompt、同窗口；8fps两组使用相同图像帧，16fps包含全部8fps帧。', '',
        '[打开带声音的同样本对照](http://localhost:8876/listen_benchmark_review/prompt_versions_20261029/?comparison=inputs)', '',
        '## 当前判断', '',
        '原生视频显著改变了时间区间的划分，减少了部分逐帧碎片；但也会把短动作概括成宽区间，或生成规则重复的运动序列。不能据此认定原生视频全面优于多图，也不能认定16fps优于8fps。', '',
        '建议保留当前C＋多图8fps为基线，重点复核原生8fps的动作覆盖和边界，再决定是否替换；不因候选变少或小数位变少就批量切换。没有运行语义去重、没有覆盖训练标签。', '',
        '## 所有窗口统计', '',
        '每组293次请求，含整段与重叠短窗；下表条目不是去重后的真实事件数。推演词筛查不是语义准确率；短区间或长区间也不自动判错。', '',
        '| 输入 | 有效请求 | 候选条目 | <0.2秒 | 同请求同句重复 | 区间中位数 | ≥窗口80%的条目 | 推演词线索请求 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for arm, name in names.items():
        s = a['stats'][arm]
        n = s['valid_events']
        lines.append(f"| {name} | {s['valid_requests']}/{s['requests']} | {n} | {pct(s['shorter_than_0_2s'],n)} | {pct(s['exact_repeated_descriptions_within_request'],n)} | {s['median_event_duration_s']:.3f}s | {pct(s['interval_over_80pct_window'],n)} | {pct(s['requests_with_explicit_markers'],s['valid_requests'])} |")
    lines += ['', '## 仅整段输出', '', '| 输入 | 有效请求 | 候选条目 | <0.2秒 | 同请求同句重复 |', '|---|---:|---:|---:|---:|']
    for arm, name in names.items():
        s = a['full_view_stats'][arm]
        n = s['valid_events']
        lines.append(f"| {name} | {s['valid_requests']}/{s['requests']} | {n} | {pct(s['shorter_than_0_2s'],n)} | {pct(s['exact_repeated_descriptions_within_request'],n)} |")
    lines += ['', '## 需要人工复核的取舍', '',
        '- S1032：整段多图52条，视频8fps为3条，视频16fps为2条。碎片减少，但还需看是否遗漏真实动作。',
        '- S0404：多图整段列出多次短暂闭眼、嘴部动作，视频整段主要概括头部和目光状态。不能只因文本更简洁就判更好；需同时查看短窗有无补充。',
        '- S1103：三组均描述微笑、右手遮嘴、放下手。可比较抬眉／睁眼细节，以及举手、放手的起止边界。',
        '- S0343：视频16fps整段出现约每0.5秒交替抬头／低头的规则序列。需要检查是否真有这些周期，不能把条目增加视为召回提升。',
        '- S1033：整段多图29条、视频8fps为15条、视频16fps为71条。16fps并没有稳定消除碎片化，请核对密集短区间是否对应真实多次动作。',
        '- S1279：三组对目光、嘴部、闭眼的解释存在差异。需要核对运动部位与次数。', '',
        '上述是从输出中发现的复核点，不是完整人工真值或定量召回评估。', '',
        '## 时间、计算与运行口径', '',
        f"- 原生metadata按整数源帧位置表示，最大取整误差为{a['max_native_timestamp_rounding_error_s']*1000:.3f}毫秒。这不是模型标注误差。",
        '- 当前vLLM原生视频按两帧组织，时间戳取帧对平均时刻并格式化为一位小数；多图输入逐帧文字时间戳保留六位小数。输出精度外观不能当作准确率。',
        '- 固定512×512且不二次抽帧；视频8fps的视觉token通常少于多图8fps，视频16fps再增加时序信息。实际prompt token数逐请求保存。',
        '- 三组每窗同批生成，只有批次耗时，不能用它声称某一组独立推理提速多少。',
        '- 最终三机各3路推理、每台1路源视频解码；完成后释放模型。node12尾段无检查点推进的分片迁移到其他两机续跑，失败／续跑日志保留。',
        '- 所有无效／截断响应均保留于results和analysis.errors，没有改写时间修复。输入8fps与旧实验的JPEG摘要逐条一致。', '',
        '详见[实验规格与运行说明](README.md)、[结构与配对审计](analysis.json)、[完成清单](COMPLETE.json)。']
    (R / 'RESULT.md').write_text('\n'.join(lines) + '\n')
    notes = ['# 整段原文对照：用于人工复核，不是GT', '']
    for code in ['S1103', 'S0404', 'S1032', 'S0343', 'S1033', 'S1279']:
        row = next(x for x in a['full_comparisons'] if '_'+code+'_' in x['sample_id'])
        notes += ['## '+row['sample_id'], '']
        for arm, name in names.items():
            r = row['arms'][arm]
            notes += [f"### {name} · {r['status']} · {len(r['events'])}条", '']
            for e in r['events']:
                notes.append(f"- [{e['start']:.6f}, {e['end']:.6f}] {e['behavior']}")
            notes += ['']
    (R / 'REVIEW_NOTES.md').write_text('\n'.join(notes) + '\n')
    print('RESULT.md and REVIEW_NOTES.md written')


if __name__ == '__main__':
    main()
