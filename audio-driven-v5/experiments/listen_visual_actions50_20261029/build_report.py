"""Render descriptive results; conclusion and media review are appended separately."""
import csv,json
from pathlib import Path
R=Path(__file__).resolve().parent
A=json.loads((R/'analysis.json').read_text());C=json.loads((R/'cross_run.json').read_text());assert A['all_samples_complete'] and A['completed_samples']==50
arms=['observation_v1','observed_actions_v2'];ss=A['stats'];samples=json.loads((R/'samples.json').read_text());rows=[]
alltexts={a:json.loads((R/f'descriptions_{a}.json').read_text()) for a in arms}
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
lines=['# 允许动作命名的观察式prompt：50条配对验证结果','',
'仅测试视觉prompt的动作名称与覆盖提示，未执行方案A文本整理。新50h同一批50条，Qwen3-VL-8B-Instruct；512×512、8fps、整段＋3秒窗口／1秒步长。新旧两组均本轮重新运行，每组293次输出；模型、schema与生成参数一致。完整prompt和复现命令见[README](README.md)。','',
'## 统计','',
'只有通过原有字段与时间校验的响应进入描述统计。原始失败输出保留，不自动修补。推演关键词仅是线索，不是人工准确率；例如as if speaking可能是对嘴部动作的不确定描述，并不一定是心理推断。候选数含重叠窗口，尚未合并去重。','',
'| 指标 | 观察式v1对照 | 允许动作命名的新版prompt |','|---|---:|---:|']
for label,k,den in [('结构与时间校验通过','valid_requests','requests'),('输出截断','truncated',None),('有有效事件的样本','clips_with_events',None),('含明确推演词线索的请求','requests_with_explicit_markers','valid_requests'),('含广义推演线索的请求','requests_with_flags','valid_requests'),('含广义推演线索的样本','clips_with_flags',None),('有效原始候选数','valid_events',None),('含明确推演词线索的候选','explicit_marker_descriptions','valid_events'),('含广义推演线索的候选','flagged_descriptions','valid_events'),('短于0.2秒的候选','shorter_than_0_2s','valid_events'),('同次请求中完全同文的额外条目','exact_repeated_descriptions_within_request','valid_events')]:
 vals=metric(k,den);lines.append(f'| {label} | {vals[0]} | {vals[1]} |')
for label,k in [('候选持续时间中位数（秒）','median_candidate_duration_s'),('每次有效请求候选数中位数','median_candidates_per_valid_request'),('平均描述字符数','mean_description_chars')]:
 lines.append(f"| {label} | {C['stats']['observation_v1'][k]:.3f} | {C['stats']['observed_actions_v2'][k]:.3f} |")
lines+=['','“明确线索”包括indicating、suggesting、implying、signalling、conveying、as if、perhaps、possibly；广义检查还包括attentive、engagement、amusement等。完整正则见analyze.py。重复原文不自动等于同一次事件，小于0.2秒也不自动等于错误。','',
'## 动作词诊断','',
'以下仅为各动作词出现在多少样本的有效输出中，包含否定句和不确定句，不是动作类别统计或召回率。“moves head down then up”等同义描述也可能没有nod这个词。','',
'| 词组 | 观察式v1样本数 | 新版样本数 |','|---|---:|---:|']
for k in ['nod','shake','smile','laugh','hand','brow','eye','tilt','mouth','shoulder','gesture','posture','gaze','head_turn']:
 lines.append(f"| {k} | {ss['observation_v1']['clips_with_action_mentions'].get(k,0)} | {ss['observed_actions_v2']['clips_with_action_mentions'].get(k,0)} |")
lines+=['','## 复现与对照边界','',
 f"50条成员和全部媒体帧摘要均与上轮相同。观察式v1的293个重跑请求中，{C['control_raw_identical']}个与上轮原文逐字相同，{293-C['control_raw_identical']}个有变化；因此本轮以同期观察式v1作为主要对照。temperature=0不保证不同批内生成过程中的输出逐位一致，差异的具体数值原因未单独归因。",
 '',
'这次同时增加了动作名称许可、非穷尽提示、持续表情和点头运动说明，衡量的是整个候选prompt，不能单独归因于其中一句。观察式描述仍需判断哪些适合作为最终backchannel标签；普通眨眼和嘴部动作并不自动具有回应意义。',
'',
'没有逐帧人工边界真值，不能给出准确率、召回率或保证±0.2秒达标。未覆盖正式标注、训练标签或缓存。',
'',
'## 文件','',
'- [逐样本CSV](per_sample.csv)：50条完整成员及逐条统计。',
'- [完整统计与异常](analysis.json)：字段、区间、逐窗口配对检查。',
'- [跨轮输入与观察式v1复现检查](cross_run.json)。',
'- [新版完整有效描述](descriptions_observed_actions_v2.json)／[观察式v1完整有效描述](descriptions_observation_v1.json)：时间已加窗口偏移，统一为片段局部秒。',
'- `results/<sample_id>.json`：所有原始响应（包括失败）、精确prompt、帧SHA256、结束原因与解析状态；其中事件时间是窗口局部秒。',
'- [实验方案与命令](README.md)及[参数](protocol.json)。']
lines[2:2]=[(R/'CONCLUSION.md').read_text().strip(), '']
(R/'RESULT.md').write_text('\n'.join(lines)+'\n')
print('Wrote report and CSV')
