# P2 打标信息原文整合：规则、代码与恢复

当前版本：`listen-existing-result-fusion-verbatim-v2`，2026-10-25。任务是整理已有打标结果，保留原文，减少同一次事件的跨窗重复，选择已有候选时间，并保留来源分歧；不重新识别音视频，不使用GPU，不用人工反馈替程序填写标签。

## 当前入口

- 50段HTML：http://127.0.0.1:8876/listen_benchmark_review/existing_fusion_verbatim/ （旧existing_fusion入口也指向本版）。
- 当前结果：`experiments/listen_existing_fusion_verbatim_20261025/`。
- GitHub代码、规则和可重放输入：[integration_verbatim](https://github.com/liqilin-eudaemonia/audio-driven-v5-data-processing/tree/main/event_labeling/integration_verbatim)。
- 本地备份：`../audio-driven-v5-data-processing/event_labeling/integration_verbatim/`。该目录README提供脱离原项目的CPU重放命令。

## 输入与职责

1. `experiments/listen_response_episodes_20261024T120000Z/observations.jsonl`：682条上游观测，包含原始来源描述、已有开放事件和既有类别。整合不改写此文件。
2. 同目录`review.json`：已有模型来源、窗口、语义关联及媒体引用。
3. `experiments/listen_split_validation_20261024T074733Z/word_alignment/`：本轮使用307份整段／滑窗ASR和逐词时间记录。

当前七类字段与候选范围沿用上游模型归类，不在本步骤重新做类别硬映射。七类之外和未决记录仍保留为证据；并非682条都会成为独立候选。这是现有50段输入格式的实现，尚未编写将全量原始响应批量接入此输入格式的适配器，也未应用到全量打标。

## 实际规则

### 描述原文

- ASR候选采用参考转写的原词或原短句，分句只用于整理事件单位；不添加`Says`模板、不改词。完整转写和原始词记录仍保留。
- 视觉／非语言音频描述优先取`source_descriptions`中的模型原文；保存所有来源版本，用其中一条作主显示。没有来源描述时使用输入观测原文。
- 不把“嘿嘿”改成笑声，不用局部`Yeah`替换`Ed`。冲突来源各自保留，参考序列的选择不证明其中内容正确。
- 原文可能同时描述多个动作。某段原文与某小区间有关联，不代表整句话的每个细节都在该小区间内成立。

### 同次候选去重

- 开放事件使用已有类别／语义标识及原文辅助匹配。仅跨窗口来源、不共享同一来源窗口、内容键一致、区间重叠，且起止差均不超过0.2秒时，才作为重复候选。
- 组内两两满足匹配条件；有多个可能归属时不强行合并。选择一条已有代表区间，不平均时间，也不把所有区间外包成一个长区间。
- 同一来源中的重复词保持独立；长间隔的相同词不合并。0.2秒是匹配容差，不是已验证的标注精度。

### ASR词序和时间

- 优先用有词记录的整段ASR作参考；没有时选词记录最多的窗口，并以窗口长度辅助排序。参考之外的原始转写继续保留。
- 按已有标点与词序整理短句，例如`Uh huh`作为一项；相同词连续出现仍拆成各次，不用类别词典改写内容。
- 汇总有效ASR及音频模型明确引用原词的已有区间。相同内容的跨窗候选按重叠和0.2秒容差整理；模型词区间若完全落在对应ASR时间支持之外，单列保留。
- 按词序选择互不倒置的正时长候选区间，优先保留可定位的次数，再比较来源支持、窗口边缘、长区间和部分短句等代价。短参考区间提供弱位置约束，长或零时长记录不作为可靠锚点。
- 不均分宽区间、不凭空插值。找不到可用有序区间的词单列失败，不用空时间假装已完成的事件。

### 多来源与视听关联

- 词语描述、通用发声描述及重叠的其他音频解释可作为相关／分歧证据，不额外断言一次新发声；原文、原区间与来源仍保存。此关联不是内容真伪判定，也不证明两个记录一定属于同一次发声。
- 视觉和音频保留各自区间，只建立可能同时发生的关系。回应组外包区间不作为连续活动标签。
- 未标出的时间保持unknown；所有结果`training_admitted=false`。

## 代码与数据导航

| 职责 | 文件 |
| --- | --- |
| 当前自动整合、原文恢复与时间选择 | `src/listen_benchmark/existing_result_fusion.py` |
| 复用的跨窗匹配、事件与证据结构 | `src/listen_benchmark/integration_pilot.py`（当前只传correction=None） |
| ASR窗口偏移、有效性及候选重复索引 | `src/listen_benchmark/response_review.py` |
| 引用词提取依赖 | `src/listen_benchmark/open_merge.py` |
| 导出、代码快照和输入哈希 | `scripts/build_listen_existing_fusion.py` |
| HTML模板／真实结果与页面检查 | `scripts/listen_existing_fusion.html`、`scripts/check_listen_existing_fusion.py` |
| 原文保留、次数、缺时间等回归 | `tests/test_listen_existing_fusion.py` |

结果目录中的`labels.candidate.jsonl`保存候选、组关系和证据；`review.json`包含完整原文和对照；`observations.jsonl`是原样输入；`policy.json`记录策略；`manifest.json`和`code/`固定来源与源码；`validation.json`记录当前50段与浏览器检查。`description_source`／`description_sources`指向原文出处，`timing_evidence`指向已选用区间及各来源边界。

## 复现与验收

从源项目根目录执行，两个输出目录必须为空：

```bash
python scripts/build_listen_existing_fusion.py \
  --source experiments/listen_response_episodes_20261024T120000Z \
  --align experiments/listen_split_validation_20261024T074733Z/word_alignment \
  --out /tmp/listen-verbatim-replay/results \
  --preview /tmp/listen-verbatim-replay/preview
python -m unittest discover -s tests -p test_listen_existing_fusion.py
```

标准库Python即可生成数据和HTML；可选字体参数需要fontTools。浏览器自动检查另外需要Playwright、Chromium以及原媒体和支持Range请求的服务器；不需要GPU。GitHub备份中的输入已包含重放所需内容，无须媒体即可重建候选JSON和页面文本，媒体播放需接回原文件。

本轮50段得到175条候选（134视觉＋41音频），2个词无法定位单列，15段没有独立候选但不能当作负例。682条输入观测逐字节保留，130条ASR词证据整理为85个候选索引，不等于真实发声次数。10项单元测试、50段结果约束和浏览器检查通过；两次Okay仍分开，Right两次和Yeah五次仍保留。已删除要求“嘿”必须变成笑声或Ed必须变成Yeah的旧断言，改查不改写来源内容。

当前限制：原文保留不消除模型误识别；已有上游区间可能偏宽；全段与窗口词序可能有分歧；未验证所有边界达到±0.2秒。完整训练标签及全量推广仍待后续确定。
