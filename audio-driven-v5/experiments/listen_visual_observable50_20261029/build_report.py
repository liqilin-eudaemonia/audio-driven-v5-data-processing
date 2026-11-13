"""Build text-only final report from independently audited paired outputs."""
import collections,csv,hashlib,json,statistics
from pathlib import Path
R=Path(__file__).resolve().parent
A=json.loads((R/'analysis.json').read_text());assert A['all_samples_complete'] and A['completed_samples']==50
arms=['baseline','observable'];ss=A['stats'];samples=json.loads((R/'samples.json').read_text())
rows=[];alltexts={a:json.loads((R/f'descriptions_{a}.json').read_text()) for a in arms}
for sample in samples:
 sid=sample['sample_id'];r=json.loads((R/'results'/f'{sid}.json').read_text());row={'sample_id':sid,'duration_s':r['duration_s']}
 for a in arms:
  records=[v for v in r['records'] if v['arm']==a];texts=[v for v in alltexts[a] if v['sample_id']==sid]
  row.update({f'{a}_requests':len(records),f'{a}_valid_requests':sum(v['status']=='valid' for v in records),f'{a}_candidate_events':len(texts),f'{a}_flagged_descriptions':sum(bool(v['flags']) for v in texts)})
 rows.append(row)
with (R/'per_sample.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def metric(k,den=None):
 return [f"{ss[a].get(k,0):,}"+(f" / {ss[a][den]:,}（{ss[a].get(k,0)/max(1,ss[a][den])*100:.1f}%）" if den else '') for a in arms]
lines=['# 50条视觉prompt配对验证结果','',
'结论：新增“只描述直接可见动作／表情，禁止心理、意图及交际意义推演”后，推演措辞减少，但仍存在明确违反这条约束的原始输出。此次简单追加不能保证解决问题，也没有解决重复短区间和动作描述覆盖变化。不能据此认定所有prompt优化都无效。','',
'## 试验条件','',
'- 从新50h（29,975条）中随机抽50条，seed=20261029，50个不同来源pair，共321.777秒；本次抽中的片段均来自V03，不代表其余分区。',
'- Qwen3-VL-8B-Instruct视觉分支；同一listener裁框、512×512、8fps、整段＋3秒/1秒滑窗。两组各293个请求，共586次输出。音频与ASR未重跑。',
'- 原prompt两行完整保留，新版只追加以下两句；输出字段、模型和生成参数相同。无额外system prompt。',
'', '```text',
'Describe only directly visible actions and expressions. Do not infer psychological states, intentions, motivations, or communicative meanings.',
'```','',
'完整原prompt与启动命令见[README](README.md)，参数见[protocol.json](protocol.json)。这是重新运行的配对试验；并发调度与旧全量运行不同，历史响应不能直接当作本轮基线。','',
'## 量化结果','',
'下表的推演词检查是固定正则诊断，不是人工语义真值。`indicating / suggesting / implying / signalling / conveying / as if / perhaps / possibly`为较明确的语言线索；广义检查还包含attentive、engagement、amusement等词，也可能误命中表情描述。只统计通过原有结构和时间检查的响应，不对失败输出静默修补。','',
'| 指标 | 旧prompt | 新prompt |','|---|---:|---:|']
for label,k,den in [
 ('通过结构与时间校验的请求','valid_requests','requests'),('输出截断请求','truncated',None),('包含明确推演词线索的请求','requests_with_explicit_markers','valid_requests'),('包含广义推演词线索的请求','requests_with_flags','valid_requests'),('含广义推演线索的样本数','clips_with_flags',None),('有有效事件输出的样本数','clips_with_events',None),('有效原始候选条数（尚未去重）','valid_events',None),('明确推演词线索条数','explicit_marker_descriptions','valid_events'),('广义推演词线索条数','flagged_descriptions','valid_events'),('小于0.2秒的原始候选','shorter_than_0_2s','valid_events'),('同一次请求内完全同文的额外条目','exact_repeated_descriptions_within_request','valid_events')]:
 vals=metric(k,den);lines.append(f'| {label} | {vals[0]} | {vals[1]} |')
lines+=['','事件总数包含整段与重叠窗口，不能作为独立事件数量或召回率。同文条目也可能是真实重复动作；此处仅作为检查线索，未自动删除。小于0.2秒不自动视为错误，8fps本身允许约0.125秒采样间隔。','',
'## 可直接核对的例子','',
'### 1. 推演明显减少，但重复仍在：S1052','',
'样本：`naturalistic_V03_S1052_I00000325_local_9949df5d9ab1`，整段输出。0.3932–0.5267秒：','',
'- 旧：`Smiling and opening mouth slightly, indicating positive engagement or amusement.`',
'- 新：`Smiling with mouth slightly open, showing teeth.`','',
'新描述只包含可见外观，但整段仍输出5个分散短区间，5条文字完全相同；仅改这两句没有自动形成合理的持续微笑区间。','',
'### 2. 明确禁止后仍有意图推断：S0127','',
'样本：`naturalistic_V03_S0127_I00000135_local_142d65a9d1ad`，新版整段仍输出：','',
'> Listener\'s mouth begins to move slightly, suggesting preparation to respond or react to the speaker\'s words.','',
'“准备回应”不能由嘴部轻微活动直接确定。这是实际残留例子，不依赖关键词比例来推断。','',
'### 3. 整段描述与短窗覆盖不一致：S1103','',
'样本：`naturalistic_V03_S1103_I00000150_local_f031c9f810fd`。新版整段仅列出抬手／捂嘴相关动作，并将同一句 `Lifts right hand completely from mouth, fingers still curled.` 重复分配到许多相邻短区间。',
'',
'内部抽帧检查可见前约3秒有明显露齿微笑，新版整段未描述，但w000、w001等短窗口仍识别微笑。因此这是整段输出覆盖变化，不能宣称整条样本漏检；也说明不能仅凭描述变短就认定标注更准确。短窗口仍有 `indicating amusement or enjoyment` 等残留。',
'',
'## 动作词覆盖诊断','',
'下表是“50条中有多少条的有效输出提到该动作词”，不是人工确认类别，也不是召回率。词形、同义描述、否定句和无效响应都会影响结果；例如“no nodding”也会被计为提到nod，不能理解成检测到了点头。','',
'| 动作词组 | 旧prompt样本数 | 新prompt样本数 |','|---|---:|---:|']
for k in ['nod','shake','smile','laugh','hand','brow','eye','tilt','mouth']:
 lines.append(f"| {k} | {ss['baseline']['clips_with_action_mentions'].get(k,0)} | {ss['observable']['clips_with_action_mentions'].get(k,0)} |")
lines+=['','新版3条含nod的描述中，2条明确说没有点头，仅1条写有subtle head nod。旧版中也含否定／不确定表述。该变化需要回看动态视频区分“旧版臆测减少”和“新版漏掉真实动作”，不能把词频差直接当作召回率下降。','','## 使用建议','',
'保留这条源头约束有帮助，但本轮结果不支持“加上两句就直接重跑全量、取消后处理”。当前仍需使用原文证据的语义去重、推演片段删除、动作保留及时间校验。后续若继续优化prompt，应保持同一批样本做对照，并同时检查遗漏与区间碎片化；本次没有擅自改变其他要求。',
'',
'此次没有逐帧人工边界真值，不能给出动作准确率、召回率或证明±0.2秒达标。','',
'## 产物与验收','',
'- [逐样本对照CSV](per_sample.csv)：50条成员、时长、有效请求数、候选数和推演线索数。',
'- [详细分析JSON](analysis.json)：逐窗口配对诊断、错误清单和原文例子。',
'- [旧版原始描述](descriptions_baseline.json)／[新版原始描述](descriptions_observable.json)：有效响应中的全部候选，start/end已换算到片段局部时间。',
'- `results/<sample_id>.json`：完整原始响应（包括未通过校验者）、prompt、帧SHA256、窗口偏移和输出状态；其中事件时间为窗口局部时间。',
'- `samples.json`：确切成员、来源媒体／裁框和历史整合事件。',
'- `code_manifest.json`、`model_metadata/`：推理代码快照与模型配置。',
'',
'检查通过：50条随机成员无增删、新旧所有窗口一一配对、两组媒体元数据和帧摘要一致、prompt仅相差指定后缀、有效输出逐条通过原有时间与字段校验。无效响应如实保留并计数，未补造事件。原生产结果、训练标签和缓存未覆盖。']
(R/'RESULT.md').write_text('\n'.join(lines)+'\n')
print('Wrote RESULT.md and per_sample.csv')
