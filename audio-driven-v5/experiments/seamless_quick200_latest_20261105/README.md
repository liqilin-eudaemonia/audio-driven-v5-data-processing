**方向修复后的当前版本（2026-11-12）：** 本目录为历史冻结版本，4条仍绑定转正前素材；后续使用 [CURRENT.json](CURRENT.json) 指向的新200／核心48入口。不能直接沿用这4条的旧首帧、条件或生成。[修复记录](../seamless_upright_eval_fix_20261112/README.md)。

**当前测评入口（2026-11-05）：** 同一固定200条现在由[三机24卡十组队列](../p2_seamless200_tenway_24gpu_20261105/README.md)执行，接替此前暂停的node8九组；数据、原始标签、裁框及音轨未改变。

# Seamless 200条快速诊断测试集：11月5日最新标签

固定选集为 **200段、1199.3653秒（19.99分钟）**，另含固定48段核心集（4.93分钟）。全部来自已接入评测的6624条独立test，不加入train/dev。选集用于pipeline快速比较，经过分层和稀有行为线索补充，不作为全量6624条的自然分布估计。

## 数据入口

- [逐条选择与原文CSV](selection.csv)、[完整选择JSONL](selection.jsonl)、[统计](summary.json)。每条记录选择理由、原文中的匹配子串、原最新标签文件与SHA、完整清单字节偏移及行SHA。
- **实际推理／评测入口**：[prepared_200/references.jsonl](prepared_200/references.jsonl)；[生成映射模板](prepared_200/generation_map.template.jsonl)。数据集名仍为`seamless`。
- [推理条件定位](prepared_200/inference_inputs.jsonl)：speaker音频及其源时间、最终listener首帧引用；[最新完整原文标签](prepared_200/annotations.jsonl)。标签不作为生成模型输入。
- 核心48段使用[prepared_core48/references.jsonl](prepared_core48/references.jsonl)，它是200段的固定子集，不是另抽48条。
- [完整最新标签引用](latest_labels.test200.jsonl)、[输入兼容验收](PREPARED_AUDIT.json)、[全量来源扫描与隔离核对](CANDIDATES_AUDIT.json)。

## 筛选规则与规模

来源是`../listen_j_labels_manifest_20261105/labels.jsonl`，标签身份`e0fa13b3987ff1585c53a6a6074982d5d705fb7db2cfa44fec715f975c1eab39`。该清单test共12284条；本次取它与既有6624条评测清单的交集。原完整清单只串行扫描一次，完整SHA通过；train/dev的group与样本ID隔离核对为0交叉。没有重新打标或使用旧行为统计。

准入要求：最新生成检查通过且无复核标记、正常结束、画质retain、人脸短边p10≥160。每个源pair仅取一段，单group及listener代码最多6段；优先大脸、不同人物／来源／时长、动作组合和重复/同时行为。CPU正向文本匹配附原文子串，只用于抽样覆盖，不改写caption，不输出新的硬类别标签或精确事件次数。

| 项目 | 200段 | 核心48段 |
|---|---:|---:|
| 实际全零listener波形 | 64 | 16 |
| 短声学活动 | 116 | 24 |
| 较长连续声学活动 | 20 | 8 |
| 人脸短边p10≥224 | 180 | 46 |
| 人脸p10在160–224 | 20 | 2 |
| naturalistic／improvised | 167／33 | 39／9 |
| 独立pair | 200 | 48 |
| 独立group | 49 | 36 |

原文覆盖点头、摇头、微笑、笑、皱眉／压眉、抬眉、眨眼、闭眼、视线变化、头倾斜／转动、手部动作、张嘴／闭嘴、抿唇及姿态变化等线索；这些是待模型和人工核验的原标签内容，不当作人工行为真值。

音频分组逐条重放最新标签实际采用的官方词／句／VAD并集配方，波形SHA与原打标输入一致。全静音指波形严格全零；活动采用20ms RMS窗／10ms步长、−50dBFS阈值。非静音需至少40ms活动；最长连续活动≥1.44s划为较长组，否则为短组。较长活动不是“确认语音”，更不代表必然满足UTMOS门槛；其中仅4条连续声学活动≥3.01s，正式语音适用性仍由评分器单独判断。

原计划的较长发声配额根据实际可用波形调整为20条，短组相应增加到116条，没有把过短声音凑作长语音或将静音拼接。保留了64条实际静音来检查误发声。

## 最终裁框、音轨和评测读取

200条均逐一核对：最新标签与原评测GT引用中的源视频、双方身份、起止时间、最终修正版512×512裁框、净化音轨配方和波形SHA相同。视频保持25fps、原pair时长；保留静音间隔，不新增8n+1尾部到GT。

为复用已验收的媒体读取器，`source.test.jsonl`保留原通过清单中逐字节不变的媒体溯源行，其中的旧caption仅供旧读取器校验来源；实际最新标签在`annotations.jsonl`和每条reference的`latest_annotation`。筛选和复核用最新caption，推理只接speaker音频＋listener首帧，不将任何版本的GT描述、未来视频或listener音轨作为输入。

`prepared_*/seal.json`固定父6624条清单、子集ID、媒体几何和行SHA；扩展的`OmniResponse/evaluation/seamless/bind.py`只允许经父清单验证的子集，原完整6624条默认约束保留。4项CPU检查覆盖成员／核心集／独立pair、最新原文和字节行SHA、重封seal后篡改帧数仍拒绝、否定动作不作正向抽样线索。未全量人工复核，不宣称标签语义完全正确。

没有复制原视频或WAV；读取仍依赖共享盘原TAR、normalized audio、最终画质测量及新旧溯源文件。全部200条实际推理读取及完整GT解码还会分别由推理与评分阶段验证。

## 复现

使用基础Python环境（含orjson、numpy、soundfile、PyAV），选择新的输出目录：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/select_seamless_quick200.py extract --output experiments/NEW_SUBSET
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/select_seamless_quick200.py select --output experiments/NEW_SUBSET
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/select_seamless_quick200.py prepare --output experiments/NEW_SUBSET
```

九组推理及评分接续：[运行记录](../p2_seamless200_nineway_20261105/README.md)。


本轮累计CPU检查10项通过：新选集／来源与守护6项、原Seamless adapter4项；见`VALIDATION.json`。准备和输入验收不代表1800条生成／评分已完成，正式完成以运行目录`COMPLETE.json`为准。
