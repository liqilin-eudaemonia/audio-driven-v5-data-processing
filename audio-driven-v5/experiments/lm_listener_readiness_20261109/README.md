# LM Listener 评测接入与阻塞核查

2026 年 11 月 9 日已完成官方测试名单的逐字段／逐张量比对，并在 CPU 上执行官方评分器的 GT 自对照。又确认5个原始公开视频来源，并取回其中1个视频及音频，导出首条候选片段和listener首帧。**当前模型尚未生成或取得本数据集分数。主要阻塞已细化为剩余源媒体映射、与官方系数的时间核对及speaker音轨验证；Stephen缺单独文件不再是名单阻塞。**

[完整审计](AUDIT.json) · [4107 行派生清单](test_manifest.jsonl) · [官方评分器验收](SCORER_ACCEPTED.json) · [可复跑核查](audit.py)。派生清单保留原行序号、来源、30fps 时间与半开帧区间，不修改发布数据。

## 去重叠测试名单

| 来源 | 发布行数 | 独立帧区间 | 来源视频数 |
|---|---:|---:|---:|
| Trevor 独立 no overlap 文件 | 2853 | 2853 | 20 |
| Conan 独立 no overlap 文件 | 446 | 446 | 11 |
| Stephen 原 test 文件 | 808 | 804 | 32 |
| 三人合并 no overlap 文件 | 4107 | 4103 | 63 |

三份人物名单的全部字段、转写及八组 tensor 按包含重复次数的多重集比较，**精确等于官方合并 no overlap 文件**。因此 Stephen 的原 test 808 行已被官方合并去重叠名单保留，不必为了名单继续等待不存在的单独文件。这是本地发布文件之间的等价性，不额外证明源视频内容不存在任何其他重叠。

Stephen 有 4 行重复帧区间。派生清单保留重复和源行号；正式生成／汇总前须固定重复权重，不能静默删除后仍声称按 4107 行评分。官方评分器用视频名＋帧号构造映射，同一区间会覆盖，输出则按找到的预测文件统计，原始行数不自动等于实际得分条数。

全部行的系数 tensor 比 `[split_start_frame, split_end_frame)` 多一帧。官方 scorer 使用半开区间；接入时不把末端额外系数算进评分，也不把任意 8 秒截取当成原始完整 segment。派生清单记录二者长度。

## 官方评分器已实测

从 Trevor 测试第一条取 60 帧 GT 56 维系数，按官方预测文件路径组织后运行未修改的 `evaluate_listener.py`，所得：

| 指标 | GT 自对照 |
|---|---:|
| MSE | 0 |
| windowed valence MSE | 0 |
| FD | 7.45e-9 |
| paired FD | -2.28e-6 |
| 预测／GT Var | 均为 0.26662275 |

paired FD 的微小负值是原 scipy sqrtm 路径的数值误差，未改分或截零。diversity 的预测和 GT 独立随机抽样，自对照也不要求完全相等。EMOCA-emorec 权重可加载，现有 protobuf／可选 SWIN 警告没有阻止本次运行；`--help` 成功与实际打分成功已分别验证。

这是一条合成预测文件的技术自对照，不是我们模型成绩，也不是官方论文整表复现。原始日志和 JSON 在 [gt_self_check](gt_self_check/) 及 [official_scorer.log](official_scorer.log)。

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  /private/software/conda/envs/lm-listener/bin/python audit.py
```

上面的核查命令在本目录运行。实际官方评分器命令从项目根目录运行：

```bash
source evaluation/lm-listener/env.sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  python evaluation/lm-listener/evaluate_listener.py \
  --output_dir experiments/lm_listener_readiness_20261109/gt_self_check \
  --segments_path experiments/lm_listener_readiness_20261109/gt_self_check/segments.pth \
  --mean_std_path evaluation/lm-listener/dataset/trevor
```

## 当前模型缺什么

官方 ZIP 共 36 个条目，原始 WAV／MP3／视频／图片条目数为 **0**。`segments_*.pth` 只有系数、转写、原始文件名及时间，不提供当前音频驱动 RGB 模型所需的波形和 listener 首帧。全盘按少量样例文件名检索因耗时已停止，不能据此宣称所有挂载盘均无原视频；当前状态是**尚无经过验证的本地解析映射**。

已进一步检查现有 RealTalk 的692份 ASR，找到11个文本锚点候选，但其目录对应 The Skin Deep，文本相同不足以确认访谈媒体身份，未将任何候选提升为评测输入。[检索明细](MEDIA_CANDIDATES.json)和[63个来源列表](source_videos.csv)已保存。

又从原文件名恢复7个可试探的视频ID，其中**5个已通过YouTube官方元数据核对标题，覆盖118条发布测试记录**，原文件名及URL见[YOUTUBE_SOURCES.json](YOUTUBE_SOURCES.json)。这把部分素材从“来源未知”推进到“找到原公开视频”，但尚未完成下载后的30fps时间、人物角色和裁框核对；标题一致不能代替媒体对齐。

首个April Ryan／Stephen源视频已通过备用CDN下载到视频和音频，并导出545–597帧对应的候选片段、16kHz WAV和512像素listener首帧；[文件摘要与状态](RECOVERED_MEDIA.json)。已视觉确认Stephen在左侧，首帧使用记录在JSON中的手工诊断裁框；它不是作者裁框的复现。音频仍是原视频混合音轨，尚未验证说话人分离和时间等价，不能直接作为已验收speaker输入。原视频为30000/1001fps，官方系数按30fps，不能直接以原帧号代替时间对齐。

候选文件：[双人音视频](recovered_source/candidate_545_597.mp4) · [候选speaker WAV](recovered_source/HxeoApgK4pk_545_597.wav) · [listener首帧](recovered_source/listener_first_frame_candidate.png)。本次取得的360p版本用于接入诊断，不能冒充官方原分辨率画质测试。

下一步继续恢复其余源文件映射，下载并验证实际视频，再按63个source建立speaker分轨、listener首帧和时间绑定。文字转写不能代替真实波形，动作系数渲染图不能冒充原listener参考图。

媒体接通后，才执行小样本当前模型生成、输出 RGB 的同空间动作提取，再接官方评分器并验证逐条覆盖。现有 ViCo 的 EMOCA 特征不能自动认定与这套 GT 系数同版本；须核对提取器与预处理。

## 与 DIM 表格的关系

DIM 论文 §4.1 写明其 LM Listener 比较只使用 **Trevor Noah 一人，train／val／test＝2366／222／543 段**，并非本次发布的三人 4107 行测试。见本地 `writing/17_DIM_ECCV2024.pdf` 及[提取页文本](../baseline_training_readiness_20261109/DIM_table1.txt)的 Table 2；段数来自论文实验设置。

DIM Table 2 还使用 Exp／Pose 分列的 FD、P-FD、MSE、SID、Var、rPCC；原 LM Listener scorer 使用完整 56 维动作和 valence 等指标。测试范围、2026 去重叠更新及指标定义均须分开标明，不能把官方 scorer 输出直接填进 DIM Table 2。
