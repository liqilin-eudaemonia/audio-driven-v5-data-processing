# 四版视觉 prompt 的 50 条对照复核

**新增 C 的输入方式对照：** [直接打开三组](http://localhost:8876/listen_benchmark_review/prompt_versions_20261029/?comparison=inputs)。多图8fps、原生视频8fps、原生视频16fps，全部使用同一 C prompt，50条／879次请求完成；与原三轮prompt实验分开标识。三组在宽屏并排显示，手机逐组显示。重点复核 S1032 的碎片减少、S0404 的短动作覆盖、S0343 的规则重复运动、S1033 的16fps碎片增多。[完整结果](../../../experiments/listen_visual_input50_20261029/RESULT.md)。

入口：[打开复核页](http://localhost:8876/listen_benchmark_review/prompt_versions_20261029/)。在 Cursor 转发远端 **8876**。原审核首页、原文整合页、方案 A 页顶部也有入口。复用已有服务，没有重启或替换历史页面。

## 展示内容

同一批新 50h 子集中的 50 条，三轮实验、六组实际输出、共 1,758 次请求：

| 轮次 | 左组 | 右组 | 来源 |
|---|---|---|---|
| 1 | A 原始简单 prompt | B 仅追加禁止心理／意图推演 | `experiments/listen_visual_observable50_20261029` |
| 2 | A 原始简单 prompt 同期重跑 | C 新观察式 v1 | `experiments/listen_visual_observation50_20261029` |
| 3 | C 新观察式 v1 同期重跑 | D 允许动作命名＋非穷尽动作清单 | `experiments/listen_visual_actions50_20261029` |

默认四版总览采用第一轮 A/B、第二轮 C、第三轮 D。跨轮结果不等同于同期配对，严谨比较应切换每轮对照。全量样本、完整原始描述、实际 prompt、失败／截断响应和整段／短窗均可查看。短窗事件展示时间 = 窗口起点 + 原始窗口内时间；展开的原始 JSON 不改写。

左侧视频复用现有 25fps、512×512 缓存文件，只新增符号链接；模型使用 8fps JPEG 输入。已逐条验证缓存与实验输入的源起点、时长、裁框一致，并验证六组输入帧时间、PTS 和 JPEG 哈希相同。缓存播放帧与模型抽帧并非逐帧相同。本轮模型只输入图像；页面补充试听帮助人工复核，不代表模型接受过音频输入。

### 同步声音（2026-10-29 补齐）

- 默认 **原始 listener**：对应参与者的独立音轨，未施加本项目新增的静音规则，可能有残留串音。
- **处理后 listener**：严格使用新 50h 子集的 `audio_postprocess/recipes/<sample_id>.json`，复用 `listen_audio_postprocess.recipes.masks/read_listener_audio`。构建时验证原 WAV 哈希，导出两个阶段的 float32 掩码；浏览器执行相同的两次 float32 乘法，以内存 Blob 播放，没有保存新 WAV、没有改动训练缓存。这 50 条中 25 条处理后全静音，页面明确提示，不当作播放器故障。
- **双人对话参考**：同时播放原始 listener、speaker，各自 50% 音量，仅帮助理解对话；speaker 声音不是 listener 回应。
- 另有静音选项和独立音量。视频是同步主时钟，播放、暂停、拖动、倍速、区间循环均带动音轨；切换样本立即停止前一条声音。

音轨复用 `p2_features_full_crop384_20261025/media/<sample_id>/{listener,speaker}.wav`，核对角色、样本时长和配方来源。处理后模式不静默回退到原音轨。50 条浏览器生成 PCM 的 SHA256 须与 Python 训练读取结果逐条一致；原始与 speaker 共 100 个媒体地址须支持 HTTP 206。音频同步与这些检查纳入同一浏览器验收报告。

这批 50 条与历史 `existing_fusion_verbatim/` 的 50 条不同，未混入历史标签或媒体。

## 人工复核重点

建议先看 S1103（动作完整性）、S0404（长区间包住多次动作）、S1032（重复短条）、S2147（推演残留）、S1049（否定点头措辞）、S0343（动作名称与真实运动），再抽查其他片段。页面还提供 S1052、S1279、S0263、S2076 快捷入口。

只需判断动作是否完整、描述是否可见、时间是否合适。不要求重新手工打标。旧版输出不是人工真值，条目数／关键词数不是检出率。相同描述出现在不同时间不必然是重复；短于 0.2 秒也不必然错误。词语提示仅帮助定位，既非穷尽检查也非自动判错。

意见按“样本＋对比轮次＋输入范围”保存在浏览器，点击保存后可导出 JSON。导出明确保留 `human_ground_truth=false`、`training_admitted=false`；不回填生产标签。

## 代码与复现

- 构建：[`scripts/build_listen_visual_prompt_versions_review.py`](../../../scripts/build_listen_visual_prompt_versions_review.py)
- 模板：[`scripts/listen_visual_prompt_versions_review.html`](../../../scripts/listen_visual_prompt_versions_review.html)
- 同步声音与浏览器内静音处理：[`scripts/listen_visual_prompt_review_audio.js`](../../../scripts/listen_visual_prompt_review_audio.js)
- 浏览器检查：[`scripts/check_listen_visual_prompt_versions_review.py`](../../../scripts/check_listen_visual_prompt_versions_review.py)

在项目根目录执行：

```bash
python scripts/build_listen_visual_prompt_versions_review.py
/tmp/p2-review-qa/bin/python scripts/check_listen_visual_prompt_versions_review.py
```

构建不调用模型，不修改标签；历史页面只添加入口，修改前备份为相邻的 `index.before_visual_prompt_versions_20261029.html`。所有结果源文件 SHA256 见 [build_audit.json](build_audit.json)。

验收：[浏览器报告](browser_validation.json)、[50 条媒体检查](media_validation.json)。已检查：50 条四版整段描述逐字一致、六组同期映射、短窗时间换算、区间循环、0.5 倍速、失败响应保留、复核导出与重载、50 个视频 HTTP 206、历史三个页面 HTTP 200、390px 手机布局和无 JS 异常。测试意见只存在隔离浏览器上下文中，没有写入人工复核数据。
