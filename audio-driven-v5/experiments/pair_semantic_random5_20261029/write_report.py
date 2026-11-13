"""Publish the measured results and the scope of the fixed policy."""
import json
from pathlib import Path
EXP=Path(__file__).resolve().parent;ROOT=EXP.parents[1]
read=lambda p:json.loads(p.read_text())
m=read(EXP/'review/manifest.json');stats=m['stats'];browser=read(EXP/'browser_validation.json');selection=read(EXP/'selection.json')
base='http://127.0.0.1:8875/listen_full_source/'
lines=['# J 切分规则：新增随机五条验证与固定版本','',
'本轮应用用户此前认可的同一份 J 代码，没有边跑边改切分规则。抽样先于结果：seed=20261031，对原始 catalog 打乱，排除此前 7 条；只要求已有原全量结果与音频缓存可比较，不按时长、题材或改善效果挑选。遇到 1 条缺少所需旧结果而跳过，过程见 selection.json。五条均为 naturalistic，是此次随机结果。','',
'结论：固定当前切分规则为 `listen-pair-segmentation-j-v1`。H→J 初始候选均为 150 段，140 段（93.33%）说话人与起止时间完全相同；10 处变化均为句末保护，起点不变、收回下一句残尾。本次未触发短前缀软停顿调整，没有新切进对齐词内部。','',
'旧全量已经保留的可靠对齐词，在 J 完整初始候选中，前四条没有减少；340 秒来源只有孤立残尾 I / I / He 不再纳入。该覆盖检查不等同于这些内容都已通过音质。','',
'**最终通过不是基本不变：旧全量 131→J 113 段，92 段起止相同，共 36 组差异。H 自动通过为 133 段。J 有 37 个未自动通过的终端候选，其中 18 个因恢复子段不能确认完整而保留原候选及原 QC 结论。新增范围、重新分段、质量待复核与排除都分列，不把待复核当成已通过，也不把通过数量减少当成额外优化证据。**','',
'固定的是本版切分与保守恢复行为；音质规则、2～11.95 秒、5 秒停顿阈值均未改变。保留完整候选供人工复核，不自动解除音质阻断。没有启动全量、替换生产结果或训练数据。','',
'## 五条 HTML','', '| 来源 | 秒 | H→J 初始候选 | 初始不变 | 旧全量/H/J 自动通过 | 旧全量→J 不变 | 入口 |','|---|---:|---:|---:|---:|---:|---|']
for i,r in enumerate(stats,1):
 url=base+f'?pair={r["pair"]}&fixed=1'+('&compare='+r['entry_case'] if r['entry_case'] else '')+'#fixedPairReview'
 lines.append(f'| {i} · {r["pair"]} | {r["source_seconds"]:.0f} | {r["initial_h"]}→{r["initial_j"]} | {r["initial_unchanged"]} | {r["old_full_kept"]}/{r["h_kept"]}/{r["j_kept"]} | {r["production_unchanged"]} | [打开对照]({url}) |')
lines+=['','三列为旧全量 P／历史规则 H／固定 J，均使用实际 pair 视频及源时间。所有 H→J 初始变化、三版本最终通过差异、J 未自动通过候选均覆盖。时间表含不变片段，完整三版本明细为 [intervals.csv](intervals.csv)。视频按各列的源时间排列。','', '## 初始切分的全部 10 处变化','', '| 来源 | H 起止（秒） | J 起止（秒） | J 保留的句末 | 不再追加 |','|---|---|---|---|---|']
for n,r in enumerate(read(EXP/'initial_inspection.json'),1):
 for c in r['cases']:
  a,b=c['before'][0],c['after'][0];suffix=a['text'][len(b['text']):].strip()
  lines.append(f'| {n} | {a["start"]:.6f}–{a["end"]:.6f} | {b["start"]:.6f}–{b["end"]:.6f} | …{b["text"][-85:]} | {suffix} |')
lines+=['','## 验证与复现','',f'- 切分模块 SHA256：`{read(EXP/"results/contract.json")["repair_sha256"]}`，与前七条所用 J 相同。',
'- 原生产 38 份源码及五条旧结果指纹不变。原输入、规则、音质参数均留档。',
'- 5 项切分回归测试通过；全部初始变化、三版本最终差异、未自动通过候选均检查覆盖。',
f'- 浏览器串行完成 {len(browser["played"])} 次对照视频播放；时间、转写、三列标签、Range 请求、手机布局及历史两条 J 入口通过。',
'- H 计算复用 131 段、重算 41 段；J 复用 148 段、重算 4 段。重型任务均串行，未扫全量媒体。',
'- 固定版本：[manifest](../../releases/listen_pair_segmentation_j_v1_20261029/manifest.json)；源码与归档校验：`python releases/listen_pair_segmentation_j_v1_20261029/verify.py`。',
'', '已有结果可按 `prepare.py` → `run.py h` → `run.py j` → `finish.sh` → `validate.py` → `check_browser.py` 重现。浏览器依赖见已有运行环境，校验过程保存在 browser_validation.json。冻结脚本 freeze.py 只在数据与浏览器验收成功后运行。',
'', '用户已认可此前展示的 J 规则与案例；这五条的判断基于真实切分、对齐词、转写与可播放文件验证，尚未把用户未提交的试听判断记录为人工通过。']
(EXP/'README.md').write_text('\n'.join(lines)+'\n')
release=ROOT/'releases/listen_pair_segmentation_j_v1_20261029'
(release/'README.md').write_text('''# 固定切分规则 J v1

入口：`scripts/listen_local_recovery_semantic_trial.py`，规则 revision 继续保留 `sentence-integrity-two-source-trial-6`，避免改名改变已验证代码指纹。正式固定标识为 `listen-pair-segmentation-j-v1`，详见 manifest.json。

在先切完整候选的流程中，保留句末保护、短前缀软停顿前瞻、短尾再分配、局部 listener backchannel 分类、句首短回应补接及失败回退。局部音质恢复不能确认子段完整时，保留原整段与原 QC 结论。硬停顿 5 秒，pair 2～11.95 秒；音质规则不变。

此前七条方案及展示案例经用户认可；新增五条真实随机来源中，150 个初始候选有 140 个不变，10 处仅收回完整句末后的残尾。旧全量内容在初始候选中只减少 I / I / He 三个孤立残尾词。自动通过数量由旧全量 131 变为 113，完整候选与待复核状态分开保留，不能将它们自动计入通过集。

[五条完整对照与入口](../../experiments/pair_semantic_random5_20261029/README.md)

`code/` 保存规则及基础依赖、实验运行入口、测试的源码归档，不是含模型与数据的独立运行环境。运行仍使用项目目录和既有资产，并先执行：

```bash
python releases/listen_pair_segmentation_j_v1_20261029/verify.py
```

后续修改需形成新版本并重新验证，不直接改写本次指纹。该固定动作未启动全量，未切换现有生产或训练结果；待复核／排除候选没有自动放行。
''')
root=ROOT/'README.md';text=root.read_text();marker='**J 切分规则已固定为 v1'
entry='**J 切分规则已固定为 v1（2026-10-29）：** 在用户认可此前七条后，再随机抽取五条未检查来源（seed=20261031）。H→J 初始候选 150→150，140 段起止不变；10 处均为保护完整句末、收回下一句残尾。旧全量已保留内容在 J 完整候选中只减少 I／I／He 三个孤立残尾词。最终自动通过仍须区分：旧全量 131→J 113，92 段不变；37 段未自动通过，其中 18 段保留整段供复核。未启动全量或自动放行音质疑点。[五条 HTML、全部起止差异与验证](experiments/pair_semantic_random5_20261029/README.md) · [固定版本与源码指纹](releases/listen_pair_segmentation_j_v1_20261029/README.md)。\n\n'
if marker not in text:root.write_text(text.replace('# audio-driven-v5 总索引\n\n','# audio-driven-v5 总索引\n\n'+entry,1))
print('REPORTS UPDATED')
