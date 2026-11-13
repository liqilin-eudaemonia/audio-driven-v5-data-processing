# P2 当前视觉打标：原生视频 8 fps＋高召回点头补充

确认时间：2026-10-29。**当前选择是 Qwen3-VL-8B-Instruct＋观察式 C prompt＋原生视频 8 fps，再补充 GLARE 高召回点头候选。** 不默认运行 16 fps、多图输入对照或其他五类专用检测器。本次整理与验证不启动全量打标、训练或 GPU 推理。

## 1. 方案边界

| 部分 | 当前配置 |
|---|---|
| 视觉语言模型 | `/private/ckpts/Qwen3-VL-8B-Instruct` |
| 输入 | listener 原视频按已确认的预裁框裁至 512×512，8 fps，原生 video tensor＋metadata |
| 时间覆盖 | 整段＋3 秒窗口、1 秒步长，保持历史验证规格 |
| 点头补充 | 同一 listener、同一起点与裁框的 512×512／25 fps 缓存视频；MediaPipe 逐帧姿态＋GLARE recall 参数 |
| 默认停用 | 摇头；微笑、视觉大笑、皱眉、惊讶专用检测器仅保留历史实验 |
| 原 VLM 内容 | 所有原文与时间保留，包括 VLM 自身描述的任何可见动作，不限定七类、不按关键词删改 |
| 声音 | 本方案是视觉分支，不输入音轨；既有音频打标与官方区间静音后处理沿用原流程 |
| 输出定位 | 保留来源的候选包，不直接等于最终去重后的生成 prompt 或训练标签 |

这里的“停用摇头”是停用专用检测器自动追加的摇头候选，不是删除 VLM 原文中可能存在的摇头／转头描述。

用户复核认可点头高召回可作补充；摇头方向检查剩余两条 S0157／S1236 仍被用户判为误报，因此不继续通过调阈值强行纳入。其他四类专用检测结果尚未获确认，也不进入默认方案。8 fps 是当前工程选择，尚无完整准确率评测证明它全面胜过 16 fps；独立测速未完成。

## 2. 当前代码入口

- [配置](../configs/listen_visual_native8_nod_v1.json)：唯一当前方案配置，明确 `supplement_enabled=["nodding"]` 及停用原因。
- [视觉输入与 prompt](../src/listen_benchmark/visual_native8.py)：从已验证实验提取，只准备 8 fps，不再准备 16 fps 或图片分支。
- [任务与导出入口](../scripts/listen_visual_native8_nod.py)：`vlm` 显式启动 GPU 推理；`export` 只读已有结果，导出当前方案。
- [CPU 点头入口](../scripts/run_glare_nodding_pilot.py)：增加 `--input-media-index`、`--sample-ids`、`--output`、`--profile recall`；默认历史三参数模式仍保留，当前方案必须显式选 recall。
- [方案验证](../scripts/check_listen_visual_native8_nod.py)：已有 15 段原文与点头逐项核对；`--decode-one` 串行检查一段真实源视频的输入一致性，不调用 GPU 推理。
- [历史复核页面](http://localhost:8876/listen_benchmark_review/prompt_versions_20261029/glare_nodding/)：六类历史实验仍可查看，不代表都获准纳入当前输出。

GLARE 原算法、MediaPipe 模型资产位于 `related/visual_reaction_annotation/`。保留原仓库文件，包装代码在本项目 `scripts/`，没有修改原始检测器。

## 3. 完整 C prompt

`{duration}` 替换为当前整段或窗口时长，窗口内时间从 0 开始。模型使用默认 chat template，无额外自定义 system message；`enable_thinking=False`。

```text
Watch this {duration}-second listener clip. Describe all directly visible movements and facial-expression changes you recognize.
For each occurrence, give its start time, end time (seconds from clip start), and a brief description of what visibly changes.
Describe physical observations only, without interpreting their psychological or communicative meaning.
Represent a continuous action or state as one interval, not separate entries for successive frames. Keep genuinely separate occurrences separate.
Return JSON with an events array. Each event has start, end, modality ("visual"), and behavior (description).
```

参数：temperature=0，seed=20261029，max_tokens=4096，max_model_len=49152，bfloat16，JSON schema 约束。原生视频关闭再次抽帧和缩放；沿用历史 JPEG85 帧传输。源 metadata 的时间取整误差逐帧记录；模型输出小数位不代表标注精度，±0.2 秒尚未经过完整边界真值检验。

## 4. 点头参数与输入

MediaPipe 资产：`related/visual_reaction_annotation/nod_headshake/face_detection/face_landmarker_v2_with_blendshapes.task`。

检测器：`related/visual_reaction_annotation/reaction_detector/nodding/run_nodding_intensity_mt_v2.py`。

```json
{
  "smooth_window": 7,
  "baseline_window": 35,
  "th_on": 0.45,
  "th_off": 0.30,
  "min_event_len": 6,
  "min_turns_per_event": 1,
  "yaw_penalty_weight": 0.35,
  "min_pitch_peak_to_valley_deg": 1.8,
  "min_peak_pitch_vel_deg": 0.10,
  "min_cycles_per_event": 0
}
```

以上窗口长度以帧计，速度是度/帧，所以点头输入保持 **25 fps**，不能直接换成 8 fps 沿用同一参数。VLM 与点头输入是同一片段的两个采样率，不是改变裁框或人物。

原算法门控后分数大于零的连续帧形成候选区间，边界为帧号÷25，末尾裁至源片段时长。一个区间可能含多个点头；强度不是概率，不能拿来与其他检测器跨类别竞争。缺失人脸和失败单独记录，不把失败当作无动作。

点头输入索引为 JSON：`{"samples":[...]}`，每条含 `id`、`video`（相对索引文件路径或绝对路径）、`duration`、`crop`。`crop` 使用当前完整 `video_transform` 结构。视频旁须有 `media.json`，用于核对 `sample_id/duration_s/frames/video_transform`。现有复核媒体索引已符合此格式：

`data_previews/listen_benchmark_review/prompt_versions_20261029/data.json`

对应缓存位于：`experiments/p2_features_full_crop384_20261025/media/<sample_id>/video.mp4`。缺缓存或无效裁框应先补齐，不能退回原画整帧。

## 5. 新任务命令（本次未启动）

从项目根目录运行，使用新的输出目录。`vlm` manifest 为 JSON 数组，每条 `sample_id`＋`source`，结构与 `experiments/listen_visual_input50_20261029/samples.json` 一致；source 必须包含 listener 视频源定位、起止、normalized_audio.sample_rate、video_transform。`export` 可接受同样格式，或 `{"ids":[...]}`。

```bash
# 原生 8 fps GPU 分支。下面为单 worker 示例，不是自动占满 GPU。
CUDA_VISIBLE_DEVICES=0 \
LD_PRELOAD=/private/software/conda/envs/vllm/lib/libstdc++.so.6 \
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
/private/software/conda/envs/vllm/bin/python scripts/listen_visual_native8_nod.py vlm \
  --manifest /absolute/path/source_samples.json \
  --output /absolute/path/new_run/visual8 \
  --rank 0 --shards 1 --memory .20

# CPU 点头：只跑已选中的 recall，不运行 FER 或摇头。
# --limit 0 表示所给索引中的全部样本；默认 limit=50，适用于小样本检查。
/tmp/glare-nod-pilot-env/bin/python scripts/run_glare_nodding_pilot.py \
  --input-media-index /absolute/path/cropped_media_index.json \
  --sample-ids /absolute/path/selected_ids.json \
  --output /absolute/path/new_run/nod \
  --profile recall --workers 2 --limit 0

# CPU 导出：需要两路都完成；缺少点头结果时明确失败，不填空冒充无点头。
python scripts/listen_visual_native8_nod.py export \
  --manifest /absolute/path/selected_ids.json \
  --vlm-results /absolute/path/new_run/visual8/results \
  --nod-results /absolute/path/new_run/nod/results \
  --output /absolute/path/new_run/bundles
```

GPU worker 按固定 manifest 的 `rows[rank::shards]` 分配；所有 worker 共用输出目录和相同 shards，协议不一致拒绝续跑。每个窗口保存检查点，结果完成后跳过；原始无效/截断输出保留，不自动改时间或假装成功。每输出目录每台主机只允许一路源解码，模型初始化串行；不同输出目录的任务不共享这把锁，不能靠任意新增目录规避并发控制。CPU worker 保持 1–2 路。

新包装改变了请求调度为单分支串行，不能保证 GPU 数值细节与历史三组同批实验逐字相同。**本次仅验证 CPU 输入准备和已有结果导出，未运行新入口的 GPU 端到端测试。** 正式扩展前先小样本启动确认 vLLM 环境与吞吐。

## 6. 结果如何用于后续整合

每段导出三部分：

- `vlm_observations`：仅 video8，完整保留整段和各滑窗的原始响应、局部时间、窗口偏移、parsed/status、输入证据。
- `vlm_full_events`：整段的原始事件数组，描述和时间逐项相同。
- `nod_candidates`：独立点头候选、片段内绝对时间、帧边界、强度、来源；描述 `Head nodding` 明确标为检测器标签，不冒充 VLM 原句。

当前不把所有窗口拼成一个事件列表，不把同时出现的 VLM 头部运动和点头候选重复拼进生成 prompt。后续语义整合应保留来源并解决同一次动作的重复；这次只固定检测方案，未重新制定全量整合规则。整段无效会阻止导出，短窗无效原样保留。导出检查两路人物 ID、时长、源起点、裁框、25 fps 和 recall 参数一致。

## 7. 已完成的样例与验证

样例目录：[experiments/listen_visual_native8_nod_20261029](../experiments/listen_visual_native8_nod_20261029/README.md)。

- `bundles.jsonl`／`results/`：15 段，94 条整段 VLM 事件＋16 个独立点头候选；摇头自动补充为 0。
- 原始 video8 共 82 个整段/滑窗响应，其中 2 个无效窗口保留；没有把 94＋16 解释为去重后的真实事件数。
- `audit.json`：原文、时间、点头区间、默认禁用项与配置一致性核对。
- 串行读取 1 段真实源视频验证：新抽帧、JPEG 元数据、视频张量、渲染后 prompt 与历史 video8 的所有窗口输入完全一致。
- `nod_replay_check/`：新 CPU 参数化入口单段复跑核对。
- `code/`、`code_manifest.json`：本次代码及文档的本地快照与哈希；没有推送 GitHub。

现有视觉原始结果共 50 段，但专用点头只在这 15 段完成。**其余 35 段尚未形成两路完整候选包，全量数据也没有按本方案重跑。**
