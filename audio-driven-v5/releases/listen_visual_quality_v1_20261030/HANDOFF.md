# 视觉质量与裁框提案 v1：代码交接

2026-10-30。将用户已确认的试验流程整理成可复用入口，不改变 J v1 的 pair 时间边界。未启动全量筛选，未覆盖生产数据，也未更新总索引 README。

## 入口与分工

- [统一 CLI](../../scripts/run_listen_visual_quality.py)：`measure` 从原媒体测量；`classify` 从已存测量重新判定。
- [测量](../../src/listen_visual_quality/measurement.py)：原 TAR 成员按偏移读取、CPU 解码、人脸与原始像素清晰度测量；不导入 experiments 脚本。
- [规则](../../src/listen_visual_quality/rules.py)：清晰度区间、裁框平移和受限扩大。
- [判定](../../src/listen_visual_quality/decision.py)：保留／排除／待复核，区分检测失败、测量不足和程序错误。
- [运行与恢复](../../src/listen_visual_quality/cli.py)：有界进程池、缓存、输入指纹、原子写入、单写入者锁。
- [固定参数](../../configs/listen_visual_quality_v1.json)；[人工确认依据](../../configs/listen_visual_quality_reviewed_evidence_20261030.json)。人工依据绑定完整来源、时间、裁框的 SHA256，不能按相同 ID 套用到不同片段。
- [回归测试](../../tests/test_listen_visual_quality.py)；[验证记录](../../experiments/visual_quality_release_validation_20261030/validation.json)。

## 当前规则

1. 原图 5 fps 检测；同时检测扩大到约 1.5 倍的上下文区域。原图检测失败或人脸中心落在当前裁框外时采用上下文结果。坐标映射回原图。
2. 清晰度在原生像素中的面部内部区域测量，内部区域任一边不足 48 像素视为无法测量。有效区域缩放至 128×128，统计 Laplacian 方差和 Sobel 梯度能量，相对同段 P80 归一化。3 点中值平滑后，两项同时低于 0.45／0.70 才记低清晰度，计入持续至少 0.6 秒的区间。未测量帧不填零。
3. **最长低清晰度区间 ≥2 秒，或累计低清晰度占整段 ≥25%，任一满足即排除。** 旧 holdout 测量脚本中的 AND 条件为历史版本，不作为当前入口。
4. 检测成功执行但整段没有检出人脸，排除。已人工确认的横转／检测失败片段，通过来源绑定的依据配置排除。短暂低头、手部遮挡不因单帧检测失败就被排除。
5. 检出部分人脸但清晰度有效测量覆盖不足 80%，转待复核；80% 不是已验证的全量排除阈值。检测程序执行异常进入待复核并在续跑时重试，不当作画质差。
6. 裁框先按整段观测头部范围加估计余量求最小平移；确需扩大时边长最多 +5%，对应固定输出分辨率下人脸线性尺寸最多约缩小 4.8%。超过上限或源边界／几何支持不足时保持原框。输出提案，不直接修改训练裁框。

## 是否需要 GPU

**不需要。此入口使用 CPU：PyAV 视频解码、MediaPipe 人脸关键点检测、OpenCV 清晰度计算和 NumPy 裁框计算。** 不依赖 CUDA；启动器清空 `CUDA_VISIBLE_DEVICES`。已有测量的 `classify` 不加载人脸模型，也不需要 MediaPipe/PyAV。

默认 2 个 worker，可用 `--workers` 调整。每个 worker 的 OpenCV、解码器、BLAS 设为 1 线程；MediaPipe 自身仍可能创建内部线程，因此不宣称整个 worker 严格只有一个线程。原 120 段双 worker 测量约 192.2 秒，不含审核视频生成，不能直接据此承诺全量耗时。加速优先复用测量并根据 CPU、存储吞吐调整 worker；本版本未实现 GPU 检测后端。

已验证环境：Python 3.10.20，`/tmp/glare-nod-pilot-env/bin/python`，主依赖见 [requirements-cpu.txt](requirements-cpu.txt)。该临时环境的 PyAV 17.1.0 及其二进制库复用本机 qwenomni 环境文件，不应作为跨机器可搬运的环境。新机器应按依赖清单安装独立环境，并确保原媒体、来源结果和模型路径可读。

模型：`related/visual_reaction_annotation/nod_headshake/face_detection/face_landmarker_v2_with_blendshapes.task`；检测选项及姿态转换复用同目录 `detect_face_multithread.py`。模型和外部代码指纹记录在本发布 manifest 中。

## 运行与恢复

在项目根目录执行。输入为已有 J 标注行：JSONL 每行一个 `{sample_id, source}`；也支持 JSON 行列表或 `{rows: [...]}`。`source` 必须包含原 TAR 定位、片段时间、原图尺寸、现有裁框、音频采样率及来源 J 结果路径和 SHA256；不是仅提供原视频路径的接口。

```bash
# 先检查代码和模型指纹。
/tmp/glare-nod-pilot-env/bin/python releases/listen_visual_quality_v1_20261030/verify.py

# 原媒体测量；示例为已验证的 5 段，默认不生成审核视频。
/tmp/glare-nod-pilot-env/bin/python scripts/run_listen_visual_quality.py measure \
  --input experiments/visual_quality_release_validation_20261030/selection5.json \
  --output experiments/visual_quality_release_validation_20261030/measured5 \
  --workers 2

# 已有 120 段测量回放，使用最新 OR 规则。
/tmp/glare-nod-pilot-env/bin/python scripts/run_listen_visual_quality.py classify \
  --input experiments/visual_quality_holdout120_20261030/selection.json \
  --measurements experiments/visual_quality_holdout120_20261030/data.json \
  --output experiments/visual_quality_release_validation_20261030/reclassified120_final

/tmp/glare-nod-pilot-env/bin/python -m unittest discover -s tests -p test_listen_visual_quality.py -v
```

续跑重复原 `measure` 命令即可，可调整 worker 数。每段结果单独原子保存；成功且指纹匹配的直接复用，缺失或执行失败的重算。输入、代码、规则、人工依据、模型或依赖变化时拒绝混用旧目录，应使用新输出目录。一个输出目录只允许一个调度进程。尚未提供四机调度功能；不要多个调度器写同一目录。

JSONL 测量输入按行读取、任务队列不超过 worker 数的两倍。`classify` 的聚合测量 JSON 在内存中加载，适用于此次缓存回放，不是大规模流式聚合入口。

## 输出与应用范围

| 文件 | 内容 |
|---|---|
| `contract.json` | 输入、代码、规则、人工依据及运行环境指纹 |
| `results/<两位分桶>/<sample_id>.json` | 逐段原始测量、清晰度区间、裁框提案和错误记录 |
| `decisions.jsonl` | 每段决定及具体原因 |
| `retained_rows.jsonl` | 保留的原始输入行，原裁框未改写 |
| `excluded_rows.jsonl` / `review_rows.jsonl` | 原行与排除／复核依据 |
| `crop_proposals.jsonl` | 独立裁框提案，`applied=false` |
| `progress.json` | 每 10 段更新的运行中进度 |
| `summary.json` | 完成凭证、计数、缓存复用与耗时；成功完成后写入 |

排除仅写清单，不删除源数据。`retain` 表示此规则未排除，不等同人工验收合格；`training_admitted=false`、`production_applied=false`。裁框正式应用时还需同步下游渲染／特征缓存，不能只改坐标而沿用旧缓存。

## 验证与已知边界

- 10 项回归测试通过：120 段的清晰度统计和裁框提案在固定依赖环境下完全一致，最新判定仍为 **保留 115、排除 5、待复核 0**。排除 N020、N039、N049、N052、N081。
- 5 段原视频实际重新解码检测：原始人脸记录、清晰度样本、统计与裁框均与旧记录一致；随后验证全缓存复用及仅完成 4/5 的恢复。
- **N052 说明更正**：实际 16/16 采样点检出了人脸，但面部内部像素区域过小，清晰度测量为 0/16；它不是“完全没检出人脸”。画面横转且无法测量，仍按用户确认排除。审核页说明已更正，历史快照保留。
- 未实现全面的自动横转识别；已确认样本可排除，但不保证检出未知横转素材。也尚未将脸移出画面的连续时长、遮挡或黑屏全部转成经用户校准的自动排除条件。
- 同段 P80 相对指标可能漏掉全程均匀模糊；运动、头发与表情变化也会影响分数。原始脸尺寸有记录，但除测量所需的 48 像素支持外，没有新增全量分辨率准入阈值。
- 头顶余量来自面部关键点外推，不等同头发分割。5% 是当前试验上限。此次整理冻结实现与证据，不意味着这些尚未校准的条件已经补齐。

`code/` 保存此次入口、模块、参数与测试副本；`manifest.json` 记录 SHA256；`verify.py` 同时核对归档、工作区和外部模型／检测代码。
