# 高召回点头时间线索融入 Omni 整段自然语言描述

2026-10-30 完成同一批 50 条小样本试验。目标是把已选 GLARE recall 点头候选及起止时间作为辅助视觉证据，交给 Qwen3-Omni，融入既有自然语言段落的动作顺序和同时发生关系。模型仍读取同一裁剪视频和净化 listener 音轨，不把输出改成事件时间戳表。

**后续 prompt 简化（2026-10-30）：** 按用户要求，当前 `src/listen_benchmark/native_caption_nod.py` 已删除 25 fps／8 fps 的说明，改为仅明确 `start`、`end` 是相对当前 pair 起点的秒数（0 = pair 开始）。检测与采样配置不变。本页下述结果、`omni/prompts/`、`omni/code/` 及 HTML 仍对应此前实际使用的旧 prompt；此次没有重新推理，不将旧结果记作新版 prompt 的结果。

同日进一步简化：发给 Omni 的点头条目仅包含 `id/start/end/observation` 四个字段，删除提示条目中的 `status`，并紧接时间说明解释四个字段。`id` 是引用编号，不代表点头次数；`observation` 沿用英文 `head nodding`（点头）。原始检测记录的候选状态仍保留，未改写历史请求或重跑结果。

## 结果

- 点头检测 50/50 成功；38 条有候选，共 50 个区间。区间不是单次点头次数，候选不是人工真值。
- 38 条重新调用 Omni：29 条明确写入 nod／nodding；4 条改变了头部动作描述但没使用 nod 词；5 条原段落未改变。
- 12 条无点头候选，直接复用上一轮段落，逐字不变。
- 本轮八张卡均实际生成结果，0–7 卡分别处理 5/5/5/5/4/5/4/5 条。前四卡当时有其他任务，使用 0.55 显存比例叠加；后四卡空闲，使用 0.85。未停止其他进程。模型、解码与输入规格相同。
- [自动验收](omni/audit.json)通过：50 条输入音频与 8 fps 视频帧证据与上一轮一致，点头来源/时间/裁框身份正确，无候选的 12 条原文一致，所有实际生成正常结束。该验收不证明动作真实或融合位置正确。

[有声复核 HTML](http://localhost:8876/listen_benchmark_review/native_simple/natural_caption_nods/)。原 `native_simple` 页面顶部新增入口。左侧播放媒体，右侧包含点头区间按钮、融合后段落、融合前段落和完整 prompt，可导出人工复核。默认显示有点头候选的 38 条，可切换“有候选但原文未变”的 5 条，以及全部 50 条。没有覆盖上一轮自然语言页面。

## 处理方法

1. 点头输入与 Omni 使用同一 listener、起点、时长和 512×512 裁框。31 条复用原 25 fps 缓存，19 条从历史 clip 以同一裁框和既有采样规则补齐。只生成无声视觉检测缓存，不新增净化 WAV。
2. MediaPipe CPU 逐帧姿态＋GLARE `recall`，参数逐项匹配 `configs/listen_visual_native8_nod_v1.json`；仅运行点头，不运行专用摇头及其他类别检测。
3. 有候选的样本：Omni 输入原 8 fps 视频数组、净化 listener 波形、上一轮原段落、带 ID 的片段内点头区间。模型把线索融合为自然语言，不在正文输出检测器名称或时间戳。
4. Prompt 明确：25 fps 证据用于补充 8 fps 的微小动作；有明确画面矛盾时不能虚构动作；不要把同一头部下沉和点头写成两个动作；不要把检测区间数当成精确点头次数；移除与点头冲突的“没有头部运动”，其他观察与听清的原词尽量保留。
5. 原始段落、检测轨迹、区间、输入证据和新版段落分别保存。代码验证来源和时间边界；自然语言是否正确融入、是否漏掉某个区间仍须看媒体复核。

## 已发现的局限

- S0805 加入发声期间和接近结尾的点头，保留 Right 和连续 yeah；S0699 加入微笑期间点头。这说明辅助时间线索能够影响段落，但不等于已证实对应关系正确。
- 4 条写成头部下沉等描述：S5268、S1464、S0866、S1875。不能仅用 nod 关键词判断动作是否被完全遗漏。
- 5 条有候选但原文未变：S1719、S1643、S0835、S0432、S1748。不能推断候选必然错误，也不能宣布它们已融合成功。
- 新段落仍可能出现过度确定次数、推演或非点头细节变动。例如 S3264 的 `nod of acknowledgment`，S1713 的再次触碰颈部和多次动作，需要人工复核。
- 上一轮段落存在的发声／意图推演错误可能被继承。此次不把新增点头当作整段感知已经正确，也不将本试验候选直接覆盖到训练数据。

## 文件与代码

- `media_index.json`：50 条 25 fps 输入索引及来源；`media25/` 为 19 条补齐的视觉缓存。
- `nod/protocol.json`、`nod/results/`、`nod/landmarks/`、`nod/scores/`：原始检测配置、逐帧姿态、强度及候选区间。
- `omni/manifest.json`：原音视频、静音配方、旧段落、点头区间和来源哈希。
- `omni/prompts/`：实际逐样本完整 prompt；`omni/results/`：原始新结果与复用记录；`omni/integration_cases.json`：逐样本文字变化统计。
- `omni/code/`、`omni/code_hashes.json`：本次推理源码快照；`delivery_code/`：交付的页面、验收和准备代码快照。
- `browser_validation.json`：播放、点头区间跳转、浏览器与 Python 50 条净化波形哈希一致性验证。
- 准备入口：`scripts/prepare_native_caption_nods.py`、`scripts/prepare_native_caption_nod_revision.py`；检测复用 `scripts/run_glare_nodding_pilot.py`。
- 提示词：`src/listen_benchmark/native_caption_nod.py`；GPU 入口：`scripts/run_listen_native_caption_pilot.py`。

```bash
cd /private/liqi_workspace/project_code/audio-driven-v5
python scripts/audit_native_caption_nod_revision.py --run experiments/listen_native_caption_nod50_20261030
python scripts/build_listen_native_caption_review.py --run experiments/listen_native_caption_nod50_20261030/omni --output-name natural_caption_nods
```

本轮没有启动全量处理或训练，也没有声称检测器时间是精确人工边界。
