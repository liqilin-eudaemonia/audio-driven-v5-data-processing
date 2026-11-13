# P2 当前视觉打标：原生视频 8 fps＋高召回点头补充

确认时间：2026-10-29。**当前选择是 Qwen3-VL-8B-Instruct＋观察式 C prompt＋原生视频 8 fps，再补充 GLARE 高召回点头候选。** 不默认运行 16 fps、多图输入对照或其他五类专用检测器。已完成小样本 GPU 加速测试（见文末实测链接），没有启动全量打标或训练。

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
  --rank 0 --shards 1 --memory .20 \
  --batch-size 4 --batch-frame-budget 256 --prefetch 1 --scheduling balanced

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

GPU worker 默认按整段及全部滑窗的预计输入帧数和每请求固定开销，用最大工作量优先的贪心算法均衡分配。所有 worker 共用完整 manifest、输出目录、shards 和执行参数；协议不一致拒绝续跑。每批完成后保存其中每个窗口，最终结果按原始窗口顺序排列；中断后只补未保存的窗口。原始无效/截断输出保留，不自动改时间或假装成功。每输出目录每台主机只允许一路源解码，模型初始化串行；不同输出目录的任务不共享这把锁，不能靠任意新增目录规避并发控制。CPU 点头 worker 保持 1–2 路。

新包装只处理 video8，现改为同片段内多窗口批量推理。批量调度不能保证 GPU 数值细节与串行或历史三组同批实验逐字相同。基础封装最初仅完成 CPU 验证；随后已在 node17 完成 8 组各 50 条的 GPU 推理测试。当前仍未测得原视频读取起始的全流水线最优配置，详见[加速实测](P2_原生视频8fps打标加速实测_20261029.md)。

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

## 8. 2026-10-29 打标调度加速

用户授权修改打标速度。本次只改执行调度；模型、C prompt、8 fps、512×512 裁框、JPEG85、原始 PTS、整段及 3 秒／1 秒滑窗、4096 输出 token 上限和点头参数不变。没有 T5 编码器，也没有预计算或替换 Qwen3-VL 视觉特征。

- [调度模块](../src/listen_benchmark/visual_schedule.py)：确定性工作量均衡、有界后台预取、按窗口长度组批。
- [运行入口](../scripts/listen_visual_native8_nod.py)：真正向 vLLM 提交多条请求及逐条 SamplingParams；每条仍用自己的窗口时长限制 JSON schema。
- [CPU 回归](../tests/test_visual_annotation_schedule.py)：11 项测试，覆盖输入/响应归属、批量与串行原文对应、失败续跑、截断保留、预取异常、分配完整性及帧预算。

执行参数以输出 `protocol.json.execution` 为准，覆盖旧固定配置中的单序列执行值，固定配置文件保留不动以兼容历史结果哈希：

| 参数 | 新默认 | 含义 |
|---|---:|---|
| `--batch-size` | 4 | 同一片段内最多同时提交 4 个窗口；同步多请求批处理，不是跨片段异步服务 |
| `--batch-frame-budget` | 256 | 每批预计输入帧数预算；过长单请求独立处理，绝不裁掉帧来凑预算 |
| `--prefetch` | 1 | 后台提前解码下一片段；可设 0/1/2，每 worker 仅一条后台解码线程，仍受同主机读锁约束 |
| `--scheduling` | balanced | 按完整 manifest 做固定工作量均衡；没有跨卡同步屏障，也没有动态抢占任务 |
| `--max-num-batched-tokens` | 4096 | vLLM 每次调度 token 预算，启用 chunked prefill；不是输出长度限制 |
| `--memory` | .20 | 保留原显存比例默认值；这不是叠加其他任务的授权，也不是独占卡的最优配置 |

三机 24 卡运行时，各卡唯一 rank 0–23，统一 `--shards 24`、manifest、输出和执行参数。24 卡须全部运行对应 rank；不能只启动部分 rank 而期望其他卡自动接管。不自动登录节点或启动进程，不改动其他任务。

对照串行基线可在**新输出目录**使用 `--batch-size 1 --prefetch 0 --scheduling round-robin`。增加批量可试 2/4/8，但应先用相同 50 条验证，显存不足时不能减少帧或输出预算冒充等价加速。发生引擎异常会退出，已完成批次保留；重新启动时需要相同代码、输入、卡数及参数。改变参数必须换目录，本次未实现跨协议迁移。

`schedule/rank-NNN.json` 保存分配和工作量估计。每段结果的 `performance` 记录本次尝试的解码、等读锁、等待预取、准备、批量推理耗时、实际批大小、续跑窗口数。多个窗口的 `elapsed_s` 是共享批次的墙钟时间，标为 `elapsed_scope=shared_batch_wall_time`；不能逐窗口相加当成总推理耗时，应读 `performance.generate_s`。重启后的该指标只统计本次尝试，不能当成完整历史成本。新增埋点不作为已经完成的 GPU 测速结论。

```bash
python -m unittest discover -s tests -p test_visual_annotation_schedule.py -v
python scripts/check_listen_visual_native8_nod.py
```

当前验收：11 项 CPU 回归通过，已有 15 段输出核对通过；另以单段真实源视频核对输入。未叠加 GPU 程序，实际提速比例、最佳 batch/显存比例及批量生成的内容稳定性等待独占资源实测。

## 9. 后续八卡实测与跨片段批处理

已完成 batch64/128/256 的八组各50条实测，当前最快为 batch256、token预算16384、每组50个片段。新增跨片段组批、可选原样帧缓存和解码槽位；13项 CPU 回归通过。原视频解码至输出的整机吞吐仍待实测。完整结果、输入/输出差异、推荐命令及资源冲突限制见[P2 原生视频8fps打标加速实测](P2_原生视频8fps打标加速实测_20261029.md)。上一节为初次加速版本记录，当前实测推荐参数以新文档为准。
