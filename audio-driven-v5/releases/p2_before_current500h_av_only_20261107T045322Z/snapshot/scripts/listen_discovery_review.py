"""Analyze frozen open descriptions; text diagnostics are not verified labels."""
from collections import Counter
import html
import json
import re

EYE_PATTERN = r'blink|eyelid|eyes? clos|eyes? open|closes? eyes?|opens? eyes?'


def analyze_discovery(data):
    samples = {s['id']: s for s in data['samples']}
    rows = []
    for name, model in data['systems'].items():
        if model['auxiliary']:
            continue
        counts = Counter()
        errors = Counter()
        per_sample = {}
        for sid, stages in data['results'][name].items():
            result = stages['discovery']
            if result['status'] != 'schema_pass':
                errors[(result.get('error') or {}).get('message', 'unknown')] += 1
                try:
                    raw = json.loads(result['raw'])
                    counts['failed_with_event_text'] += bool(raw.get('events'))
                except (ValueError, AttributeError):
                    pass
                per_sample[sid] = dict(status=result['status'], event_count=None)
                continue
            counts['passed'] += 1
            events = result['parsed']['events']
            counts['positive_samples'] += bool(events)
            counts['events'] += len(events)
            long_count = eye_count = repeat_count = 0
            for event in events:
                counts[event['modality']] += 1
                long_count += (event['end'] - event['start']) >= samples[sid]['duration'] * .8
                eye_count += bool(re.search(EYE_PATTERN, event['behavior'], re.I))
                counts['uncertain_events'] += bool(event['uncertainty']) or event['owner'] == 'uncertain'
            # Exact same behavior/evidence on adjacent intervals, per modality.
            for modality in ('visual', 'audio'):
                previous = None
                for event in sorted((e for e in events if e['modality'] == modality), key=lambda e:(e['start'],e['end'])):
                    normalize = lambda s: ' '.join(s.lower().split())
                    key = (normalize(event['behavior']), normalize(event['evidence']))
                    if previous and previous[0] == key and abs(event['start'] - previous[1]) <= .05:
                        repeat_count += 1
                    previous = key, event['end']
            counts['long_events'] += long_count
            counts['eye_text_events'] += eye_count
            counts['adjacent_repeat_extras'] += repeat_count
            per_sample[sid] = dict(status=result['status'], event_count=len(events),
                                  long_events=long_count, eye_text_events=eye_count,
                                  adjacent_repeat_extras=repeat_count)
        rows.append(dict(system=name, name=model['name'], branch=model['branch'],
                         counts=dict(counts), failures=dict(errors), per_sample=per_sample))
    def case(pattern, title, note, members, inspection):
        sid = next(sid for sid in samples if pattern in sid)
        return dict(sample_id=sid, title=title, note=note, members=members, inspection=inspection)
    cases = [
        case('S2043_I00000338', '张嘴露齿 → 抬手遮嘴：描述有用，时间却可能过宽',
             'Qwen3-VL 把张嘴露齿、手开始抬起、遮嘴分成不同区间；Omni 原生视听把“遮嘴”写成整个 0–3.041 秒。采样画面中，手到约 2 秒才进入脸旁，不能把整段当作遮嘴。Joy 原文也描述了两段变化，但 reviewed 只写到 1.9 秒，导致后段事件校验失败。',
             ['qwen3_vl_visual','joy_visual','qwen3_omni_native'], '已抽看原片 0–2.77 秒的采样帧；未核验声音或精确动作边界。'),
        case('S1806_I00000480', '片尾微笑：开始时间相近，持续时间分歧很大',
             'Joy 报 7.1–8.223 秒，Qwen3-VL 报 7.1404–7.9079 秒，Omni 视觉报 7.140467–7.273933 秒。采样帧中约 7.1 秒起嘴角上扬，随后仍可见笑容；Omni 视觉的短区间更像变化发生的一刻，未覆盖持续状态。',
             ['joy_visual','qwen3_vl_visual','qwen3_visual'], '已抽看原片 6.3–8.08 秒采样帧；属于定性复核，不是逐帧人工真值。'),
        case('S0699_I00000500', '同一微笑重复切片，事件数量被放大',
             'Omni 视觉将相同的 Smiling 描述与证据连续重复 18 段，多数每段 0.5 秒。Joy 又把 0–4 秒描述为“eyes closed”，但该区间多个采样帧可见睁眼。原生视听把“说 Okay 两次”合为覆盖全片的一条声音事件。三种问题分别是重复切片、画面描述不符和未拆分声音边界。',
             ['qwen3_visual','joy_visual','qwen3_omni_native'], '已抽看约每 0.8 秒一帧；声音次数与内容仅为模型报告，尚未听审。'),
        case('S3370_I00000221', '“yeah”候选：听到了什么与发生在何时要分开验收',
             '纯音频 discovery 报 6.27–6.62 秒的短发声，并明确表示可能是 yeah、词语不确定；原生视听写出 Yeah，却把声音区间设为整个 0–8.78 秒。切换 audit 还能检查同一模型的另一套时间判断。两份模型回答均不能代替听审。',
             ['qwen3_audio','qwen3_omni_native','minicpm_audio'], '这里只核对原始回答；未听审，不判定哪个时间正确。'),
        case('S1460_I00000182', '开放描述与审计也会相互矛盾',
             'Joy discovery 主要描述手掠过脸部、嘴和眼的变化，备注说大部分时间相对静止；audit 备注却写开头至约 2.4 秒持续点头，events 仍为空。这是回答之间及字段之间的矛盾，不是开放描述已经补全点头的证据。',
             ['joy_visual','qwen3_vl_visual','qwen3_omni_native'], '回答一致性检查；未据此认定实际有没有点头。'),
    ]
    return dict(rows=rows, cases=cases, human_ground_truth=False,
                source_run=data['run_id'], source_sha256=data['provenance'],
                methodology=dict(counts='Only schema_pass discovery events; failed events excluded, not negatives.',
                                 long_events='event duration >= 80% of clip duration; diagnostic, not an error label',
                                 eye_text_events='Regex on behavior only: '+EYE_PATTERN,
                                 adjacent_repeat_extras='Same normalized behavior and evidence, same modality, consecutive intervals within 0.05 s; count extras after first.',
                                 visual_inspection='Three selected clips inspected through sampled original-preview frames; no audio listening or exhaustive human event GT.'))


def discovery_markdown(analysis):
    lines = ['# 现有开放描述结果分析', '',
             '结论：开放描述已包含有用的动作变化，值得作为下一步候选；目前更接近“带时间的观察记录”，不能直接当作已验收的 backchannel 标签。',
             '分析使用完整重测的既有 discovery 原始回答，没有重新推理或修改模型标签。统计只包含通过结构校验的回答；失败不视为阴性。', '',
             '| 配置 | 结构通过 /50 | 有描述样本 /50 | 视觉 / 音频条目 | 长区间条目 | 眼部词语匹配 | 连续重复额外段 |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for row in analysis['rows']:
        c=row['counts']
        lines.append(f"| {row['name']} | {c.get('passed',0)} | {c.get('positive_samples',0)} | {c.get('visual',0)} / {c.get('audio',0)} | {c.get('long_events',0)} | {c.get('eye_text_events',0)} | {c.get('adjacent_repeat_extras',0)} |")
    lines += ['', '长区间指覆盖片长至少 80%，本身不一定错误；眼部统计仅为 behavior 文本正则匹配，不是动作真值或无效反馈统计。连续重复要求相邻段的行为和证据文字相同，不自动合并或删除。', '',
              '## 主要发现', '',
              '- Qwen3-VL 的开放结果比其点头审计丰富，也能描述张嘴、遮嘴、微笑等变化；适合作为优先人工审阅的视觉候选，不能据此宣布最准确。',
              '- Qwen3-VL 的 254 条中有 125 条文本涉及眨眼/闭眼/睁眼；Joy 为 70/159，Omni 视觉为 120/214。同一个眨眼还可能分成闭眼和睁眼两条，数量并非独立反馈次数。',
              '- Omni 视觉有重复短片段描述；原生视听有概括整段的描述。前者可能虚增事件数，后者可能无法提供所需的细时间边界。',
              '- MiniCPM 音频 discovery 在 49 条结构通过回答中没有任何事件，但同轮 audit 报告 57 个发声事件。这说明提示任务和输出阶段对结果影响很大，不能把空 discovery 当作片段没有声音。',
              '- Joy 的格式失败样本仍可能含可读且部分符合画面的描述；应把格式/覆盖问题与感知问题分开分析，原文保留供复核。', '',
              '## 具体样本']
    for item in analysis['cases']:
        lines += ['', '### '+item['title'], '', '`'+item['sample_id']+'`', '', item['note'], '', '核查范围：'+item['inspection']]
    lines += ['', '## 当前建议', '',
              '先以 Qwen3-VL 视觉＋Qwen3-Omni 纯音频的开放描述作为优先审阅对象，Joy 和 Omni 视觉作对照；这里是审阅顺序，不是生产选型。保留所有动作候选，但将普通动作、可能的反馈和语境不明分开复核；眨眼不自动排除。',
              '先核验事件内容，再校准时间、合并同一持续事件的重复描述；单次粗区间不能冒充细边界。后续需要明确动作“变化的一刻”与“持续状态”的标注口径。',
              '尚无完整人工逐事件真值；未报告准确率、召回率或自动标注合格率。']
    return '\n'.join(lines)+'\n'


def discovery_panel(analysis):
    rows=[]
    priority={'qwen3_vl_visual':0,'joy_visual':1,'qwen3_visual':2,'qwen3_audio':3,'qwen3_omni_native':4}
    for r in sorted(analysis['rows'],key=lambda r:priority.get(r['system'],10)):
        c=r['counts']
        rows.append(f'''<tr><td>{html.escape(r['name'])}</td><td>{c.get('passed',0)}/50</td><td>{c.get('positive_samples',0)}/50</td><td>{c.get('visual',0)} / {c.get('audio',0)}</td><td>{c.get('long_events',0)}</td><td>{c.get('eye_text_events',0)}</td><td>{c.get('adjacent_repeat_extras',0)}</td><td><button class="mini-action" data-open-model="{r['system']}">看描述 →</button></td></tr>''')
    cases=[]
    for i,c in enumerate(analysis['cases']):
        cases.append(f'''<div class="panel discovery-case"><span class="tag">复核样本 {i+1}</span><h3>{html.escape(c['title'])}</h3><p>{html.escape(c['note'])}</p><p class="foot">{html.escape(c['inspection'])}</p><button class="mini-action" data-open-case="{i}">打开同样本对照 →</button></div>''')
    return f'''<div class="lead"><div><span class="tag">现有 discovery 结果 · 本次未重新推理</span><h2>模型描述了不少动作，<br>需要分清观察、反馈与时间边界。</h2><p>Qwen3-VL 已描述张嘴露齿、遮嘴、微笑等变化；但眨眼拆分、重复短片段和整段概括都会影响事件数量与定位。先看下方具体样本，再判断哪些能作为 backchannel 候选。</p><div class="actions"><button class="primary" data-open-case="0">先看“张嘴露齿 → 遮嘴”</button><a href="开放描述分析.md" download>下载开放描述分析</a></div></div><div class="lead-aside"><small>Qwen3-VL 开放描述条目</small><strong>254</strong><p>其中 125 条文本涉及眨眼／闭眼／睁眼</p><p class="foot">这是词语匹配，不是 125 个错误，也不是 254 个已确认的反馈。原始输出保持不变。</p></div></div>
<div class="panel"><h2>开放阶段独立统计</h2><p class="muted">分母始终是同一批 50 条；条目只统计结构通过的回答。视觉与音频分别计数，辅助 Captioner／ASR 不混入事件数。</p><div class="table-scroll"><table><thead><tr><th>模型配置</th><th>结构通过</th><th>有描述样本</th><th>视觉 / 音频条目</th><th>长区间 ≥80%</th><th>眼部词语匹配</th><th>连续重复额外段</th><th></th></tr></thead><tbody id="discoveryStats">{''.join(rows)}</tbody></table></div><p class="foot">长区间不自动等于错误；重复段仅检测相邻且行为和证据文字相同的情况，未自动修改标签。详细统计口径见 <a href="discovery_analysis.json" download>分析 JSON</a>。</p></div>
<div class="panel"><h2>本轮怎么看</h2><p>优先审阅 Qwen3-VL 视觉与 Omni 纯音频的开放描述，用 Joy／Omni 视觉交叉核对。这是审阅顺序，不是已证明的准确率排名。现有结果的主要问题是普通动作混入、重复切分、整段概括以及不同阶段回答不一致。</p><div class="hint warning">MiniCPM 音频：discovery 的 49 条结构通过回答均无事件，但 audit 报告 57 个发声事件。空列表不能直接当可靠负例。下方三个视觉样本已抽看原片采样帧；声音仅比较模型报告，未听审。</div></div>
<div class="columns2">{''.join(cases)}</div>'''
