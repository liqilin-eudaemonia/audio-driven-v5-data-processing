"""Evidence-based before/after presentation for the complete 24-GPU rerun."""
from collections import Counter
import html
import json
from pathlib import Path
import re


def retest_comparison(run, data):
    plan = json.loads((run / 'retest_plan.json').read_text())
    baseline = Path(plan['baseline'])
    old = json.loads((baseline / 'final_comparison.json').read_text())
    rows, native_before = [], {}
    for previous in old['rows']:
        name = previous['system']
        summary = json.loads(Path(previous['summary']).read_text())
        audits = [r for r in summary['results'] if r['stage'] == 'audit']
        counts = Counter(e['type'] for r in audits if r['status'] == 'schema_pass'
                         for e in r['parsed']['events'])
        positive = Counter(t for r in audits if r['status'] == 'schema_pass'
                           for t in {e['type'] for e in r['parsed']['events']})
        current = data['systems'][name]
        changes = []
        for r in audits:
            after = data['results'][name][r['sample_id']]['audit']
            before_events = r.get('parsed', {}).get('events')
            after_events = (after.get('parsed') or {}).get('events')
            if r['status'] != after['status'] or before_events != after_events:
                changes.append(r['sample_id'])
        rows.append(dict(system=name, name=current['name'], auxiliary=current['auxiliary'],
                         old_phase=previous['phase'], new_phase=current['phase'],
                         old_pass=previous['complete_schema_pass'], new_pass=current['complete_schema_pass'],
                         old_events=dict(counts), new_events=current['event_counts'],
                         old_positive=dict(positive), new_positive=current['positive_clips'],
                         changed_audit_samples=changes))
        if name == 'qwen3_omni_native':
            for r in summary['results']:
                native_before.setdefault(r['sample_id'], {})[r['stage']] = dict(
                    status=r['status'], parsed=r.get('parsed'),
                    raw=r.get('raw_response', {}).get('text', ''))
    return dict(baseline=str(baseline), rows=rows, native_before=native_before,
                human_ground_truth=False, change='explicit native video grid time: 0.25 seconds')


def retest_template(template, data):
    native = data['systems']['qwen3_omni_native']
    old = next(r for r in data['retest']['rows'] if r['system'] == 'qwen3_omni_native')
    nod, voice = native['event_counts'].get('head_nod', 0), native['event_counts'].get('listener_vocalization', 0)
    joy, vl = data['systems']['joy_visual'], data['systems']['qwen3_vl_visual']
    lead = f'''<div class="lead"><div><span class="tag">24 卡完整重测 · 16 配置 / 50 样本</span>
<h2>时序已修正并重测，<br>点头质量仍需逐段验收。</h2>
<p>Qwen 原生视听：发声事件从 {old['old_events'].get('listener_vocalization',0)} 个变为 {voice} 个，点头从 {old['old_events'].get('head_nod',0)} 个变为 {nod} 个。事件数量变化不是准确率；本轮没有修改点头 prompt 或提高采样帧率。</p>
<div class="actions"><button class="primary" id="openDisagreement">打开点头分歧样本 →</button><a class="mini-action" href="选型分析.md" download>下载重测分析</a></div></div>
<div class="lead-aside"><small>原生视听报告的发声事件数</small><div><strong>{old['old_events'].get('listener_vocalization',0)} → {voice}</strong></div><p>上轮时序错误 → 本轮修正</p><p class="foot">全部 1,450 次请求重新执行。原始样本、prompt、生成预算及各模型推理引擎保持原轮口径。</p></div></div>'''
    template = re.sub(r'<div class="lead">.*?(?=\n<div class="metrics">)', lead, template, count=1, flags=re.S)
    rows = []
    for row in data['retest']['rows']:
        if row['auxiliary']:
            continue
        a, b = row['old_events'], row['new_events']
        rows.append(f"<tr><td>{html.escape(row['name'])}</td><td>{row['old_pass']} → {row['new_pass']}</td><td>{a.get('head_nod',0)} → {b.get('head_nod',0)}</td><td>{a.get('listener_vocalization',0)} → {b.get('listener_vocalization',0)}</td><td>{len(row['changed_audit_samples'])}/50</td></tr>")
    panel = f'''<div class="panel"><h2>重测前后变化</h2><p>本表是模型输出变化，不是人工验收成绩。其他配置也全部重新执行，用于检查重复运行的稳定性。</p><div class="table-scroll"><table><thead><tr><th>配置</th><th>两阶段结构通过 /50</th><th>点头数</th><th>发声数</th><th>audit 状态或事件内容变化</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
<div class="hint warning">Joy 本轮报告 {joy['event_counts'].get('head_nod',0)} 个点头，Qwen3-VL 报告 {vl['event_counts'].get('head_nod',0)} 个，修正版原生视听报告 {nod} 个。没有逐事件人工真值，不能据此选择准确率冠军。点头漏检仍需专项验证。</div>
<p class="foot">逐样本对照默认展示本轮结果；原生视听栏可展开“上轮输出”查看修正前回答。灰色表示未声明有效覆盖，结构通过不表示质量合格。</p><a href="retest_comparison.json" download>下载前后变化数据</a></div>'''
    template = re.sub(r'<div class="panel"><h2>目前可以决定什么</h2>.*?(?=\n</section>)', panel, template, count=1, flags=re.S)
    template = template.replace('最终纳入版本已与旧版失败分开。MiniCPM 修正真实音频尾段；Captioner 改回默认模板；Qwen 原生视听采用单请求、关闭分块预填充的独立 vLLM 引擎。没有隐藏旧版记录，也没有把无输出当无事件。',
                                '本轮 16 配置全部重新执行。Qwen 原生视听修正交错时序，继续采用单请求、关闭分块预填充；其他配置延续上一轮设置。空闲卡转给剩余任务，MiniCPM 和 Qwen2-Audio 使用多副本；资源曲线包含加载和收尾。')
    return template


def retest_analysis(data):
    rows = data['retest']['rows']
    native = next(r for r in rows if r['system'] == 'qwen3_omni_native')
    n = native['new_events']
    lines = ['# 24 卡完整重测分析', '',
             '16 种配置、同一批 50 条样本、1,450 次请求全部重新执行。所有标签仍待人工验收，当前没有可确认的最终最佳方案。', '',
             '## 原生视听修正的影响', '',
             f"修正 vLLM 原生视听的音视频 token 时间网格后，发声事件从 {native['old_events'].get('listener_vocalization',0)} 个变为 {n.get('listener_vocalization',0)} 个，点头从 {native['old_events'].get('head_nod',0)} 个变为 {n.get('head_nod',0)} 个。两阶段结构通过 {native['old_pass']}/50 → {native['new_pass']}/50。",
             '时序参数由缺省 2 秒改为显式 0.25 秒（8 fps、每组 2 帧）；预处理顺序已与 HF 处理器核对。模型、样本、prompt 和生成预算保持一致。新增事件是否正确、时间边界是否准确，仍需人工验证。', '',
             '## 全部配置前后对照', '',
             '| 配置 | 两阶段结构通过 | 点头数 | 发声数 | audit 状态或事件内容变化 |',
             '|---|---:|---:|---:|---:|']
    for r in rows:
        if r['auxiliary']:
            continue
        a, b = r['old_events'], r['new_events']
        lines.append(f"| {r['name']} | {r['old_pass']} → {r['new_pass']} | {a.get('head_nod',0)} → {b.get('head_nod',0)} | {a.get('listener_vocalization',0)} → {b.get('listener_vocalization',0)} | {len(r['changed_audit_samples'])}/50 |")
    lines += ['', '## 选型建议', '',
              '音频分路与修正版原生视听继续作为待审候选，先核对同一声音的有无判断和边界。点头端不推荐直接生产：事件报告差异依然需要人工真值解释；数量更多也可能是误报。',
              '本轮是同配置重测，没有进行短窗口、高帧率或点头专项 prompt 实验。下一步应先给明确漏标片段建立人工事件边界，再控制变量比较这些改动；头部姿态轨迹可辅助定位，不自动等同于点头。',
              '灰色未声明有效覆盖不表示模型没有收到媒体；结构校验通过不证明备注与事件语义一致。旧 Joy 的备注／事件矛盾作为诊断证据保留，不自动视为本轮同样错误。', '',
              '## 证据与限制', '',
              '- `retest_comparison.json`：逐配置统计及发生变化的样本 ID，并保留上轮原生视听回答。',
              '- `native_alignment_diagnosis.json`：CPU 预处理回归，不是事件准确率测试。',
              '- `final_comparison.json` 与原始记录：本轮全部输出和失败。',
              '- `resource_summary.json`：三台机器资源实测，包含模型加载与空闲收尾。',
              '其他配置的重复运行差异需结合批次、浮点计算及推理实现解释；不能把差异全部归于本次时序修正。未建立独立人工真值，不报告 precision / recall / F1。']
    return '\n'.join(lines) + '\n'
