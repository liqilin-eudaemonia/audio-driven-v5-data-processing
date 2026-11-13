**后续新增CPU流水化优化（2026-11-07）：** 保持本页原生指标与GT缓存，在独立入口提前读取下一条并计算像素指标。[新实现与同样本实测](../../related/P2_评测CPU预取与像素计算流水化提速_20261107.md)。

# ResponseNet评测加速与接续（2026-11-02）

## 最终采用配置与完成状态

北京时间 **2026-11-02 14:25:22**，本轮四权重×586条ResponseNet的生成、评分和最终汇总全部完成。2,344条视频／音频结果、128个评分分片完整，视频与音频执行错误均为0；[最终完成标记](../p2_testsets_4ckpt_24gpu_20261102/COMPLETE.json)、[四权重汇总结果](../p2_testsets_4ckpt_24gpu_20261102/RESULTS.json)、[本次验收](FINAL_VERIFIED.json)。UTMOS的44个GT语音窗口均不满足长度门槛，每个权重可评分窗口为0，结果为N/A，不填0分。Seamless仍未执行。

下面是本次验证后推荐复用的配置；机器可读版本见[recommended_config.json](recommended_config.json)。worker数量没有做穷尽式搜索，不能据此声称所有硬件／数据上的全局最优。

| 项目 | 最终设置 |
|---|---|
| Torch视频指标 | `--device cuda:0` |
| 人脸检测／106点关键点 | `--face-device cuda:0`，CUDA provider失败时明确报错 |
| DeepFace表情 | `--emotion-device cuda:0`，原模型与32帧batch |
| LPIPS／FID batch | **64**；本次128未显示稳定提速 |
| PSNR／SSIM并行 | `--pixel-workers 4`，保持原公式和帧顺序 |
| 每进程线程 | `--threads 4`；`OMP_NUM_THREADS=4`、`OPENBLAS_NUM_THREADS=2` |
| 四台释放的推理机 | **每GPU最多4个worker**，每机8卡；任务不足时按需启动 |
| 172.29.189.119 | **每空闲GPU最多8个worker**；本轮只有GPU7空闲 |
| GT复用 | 共享`--gt-cache`，缓存FID/FVD特征及未归一化人脸／关键点／情绪 |
| ORT CUDA | 关闭TF32，卷积算法搜索`HEURISTIC` |
| TensorFlow GPU | 显存按需增长、关闭TF32 |
| 评测采样 | FID最多32帧；FVD中间32帧、stride1；其他原全帧指标不降采样 |
| ResponseNet检脸边框 | `--face-padding 64` |
| 调度 | 每权重32分片；推理全部完成后才接入四机GPU；控制器重启可接管存活worker |
| 最终汇总 | 在172.29.189.119合并全量特征重算FID/FVD，汇总CPU线程4 |

当前任务的共享缓存为`/private/liqi_workspace/project_code/audio-driven-v5/experiments/p2_metrics_acceleration_20261102/gt_cache`。新实验可沿用该目录，但模型、预处理、代码或GT内容变化会产生不同缓存键。调用示例仍见下方“通用调用”；通用CLI默认CPU后端并未被悄悄改成GPU，复用时须显式传上述参数。

生成配置保持：四机32卡、每卡单样本原生采样，未微调Qwen-Omni-3B规划，seed137，30步、video CFG3／audio CFG7，512×512、原FPS及原时间窗；本次只优化评分，没有更换生成协议。下方按时间保留2→4 worker、单卡→多机等历史过程，以本节为最终配置。

针对当前四权重×586条ResponseNet，已实现GPU人脸检测／106点关键点、GPU情绪分类、共享GT特征缓存、并行CPU PSNR／SSIM，以及推理完成后使用释放GPU的评分调度。指标公式、全帧PSNR／SSIM／LPIPS、FID/FVD采样、检测门槛、裁框和短语音协议保持原设置。

## 已验证的改动

- InsightFace两个ONNX模型可指定CUDA；实际provider记录进contract。如果请求CUDA却回退到CPU，初始化明确失败。CUDA关闭TF32，使用HEURISTIC卷积算法搜索。
- DeepFace原EmotionClient可指定TensorFlow GPU，开启显存按需增长、关闭TF32；沿用原权重、预处理和32帧表情batch。默认仍CPU，兼容旧调用。
- `--pixel-workers 4`并行执行原`pixel_scores`，保持帧顺序和原汇总公式。
- `--gt-cache`仅缓存GT FID/FVD特征、未归一化人脸框／关键点／情绪标签／检测数。缓存绑定实际RGB、采样参数、模型权重、代码、版本和执行设备；不缓存生成视频特征，也不缓存依赖双方框大小的关键点归一化结果。原子发布＋文件锁支持多进程共享。
- 新增每条`timings`、`gt_cache_hits`与contract执行设备记录。

## 实测与数值一致性

[两条完整视频对照](BENCHMARK.json)，同一评测机空闲GPU7、相同真实样本（107和324帧）：

| 模式 | 包含启动的总耗时 | 两条逐样本耗时 |
|---|---:|---|
| 原CPU人脸／表情、串行像素指标 | 144.45秒 | 31.27／104.48秒 |
| GPU人脸／表情＋4线程像素指标，首次缓存 | 46.64秒 | 12.86／24.30秒 |
| 同上，复用GT缓存 | 42.82秒 | 11.16／21.86秒 |

小批总耗时约3.1–3.4倍加速，不能直接当作完整多机吞吐倍数。解码也存在I/O时间波动。PSNR／SSIM／LPIPS、全部FID/FVD特征逐值一致；人脸检测覆盖、情绪标签与汇总一致。GPU相对CPU的ΔResp绝对差为4.21e-5和5.28e-6。缓存首次／复用的全部指标一致。

[额外8片段检脸对照](FACE_PARITY.json)：每侧每片段等距8帧，共128帧，包括历史紧裁框检测问题来源。检测数、有效性、情绪类别均一致；框坐标最大差0.000214像素，关键点最大差0.04143像素。没有宣称CPU/GPU逐位一致或全数据集标签绝对不会变化。

39项原回归与新增缓存测试通过，见[tests.log](tests.log)；8项队列／协议测试通过，见[queue_tests.log](queue_tests.log)，共47项。新增测试覆盖缓存复用、像素／参数／模型变动失效、调用方修改不污染缓存、提取失败不发布结果；现有指标公式、聚合、绑定和音频测试保留。调度器另外检查了生成完成标记前不开放推理GPU、完成后开放、每类机器容量及关闭开关。

## 当前后台调度

主目录仍为`../p2_testsets_4ckpt_24gpu_20261102/`，实际推理四机32卡。此次优化没有中断视频推理。当前评测独立控制器见该目录`metrics_launch.json`，实时进度见`metrics_status.json`。

- 在172.29.189.119空闲GPU先以8个评测进程运行，其他占用中的GPU不分配。
- 仅在`GENERATION_COMPLETE.json`写入之后，自动将node17／node12／node4／node8空闲GPU加入评测，初始每卡2进程、最终每卡4进程；不会与尚未完成的视频生成抢卡。
- 分片仍为每权重32份，共128份；各分片只调度一次，失败可按原规则重试。GT缓存跨权重／机器共享。
- 最终FID/FVD使用所有分片特征重新计算，不平均分片分数；最终合并仍在172.29.189.119执行。
- 旧CPU评分、部分结果及完成记录保存在[old_cpu_scores](old_cpu_scores/)，原代码快照在[baseline_paired_metrics](baseline_paired_metrics/)。为避免新旧数值后端混合，正式四权重统一重新评分；视频和生成音频全部复用。
- GPU后端的两样本验证替换了旧CPU评测smoke标记，原音频后端／短回应规则未改动。

该目录保存[启用记录](ACTIVATED.json)、优化前协议／状态、逐模式log、逐样本结果、分布特征和检脸中间数组。最终是否完成及实际多机吞吐必须读实时文件，不能将本文视为全部评分已结束。

## 通用调用

在`/private/liqi_workspace/project_code/OmniResponse`下使用已绑定的配对清单：

```bash
CUDA_VISIBLE_DEVICES=7 \
PYTHONPATH=/private/liqi_workspace/.cache/paired_video_metrics/python \
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=2 \
/private/software/conda/envs/AnyTalker/bin/python -m evaluation.video.paired_metrics.run \
  --manifest /path/to/pairs.jsonl --output /path/to/new_output \
  --assets evaluation/video/paired_metrics/assets.local.json \
  --device cuda:0 --face-device cuda:0 --emotion-device cuda:0 \
  --threads 4 --pixel-workers 4 --batch-size 64 --face-padding 64 \
  --gt-cache /path/to/shared_gt_features
```

`evaluation.listener_metrics.suite`同样支持`--face-device`、`--emotion-device`、`--pixel-workers`、`--threads`与`--gt-cache`，传给视频阶段。它的音频选项沿用原入口，不因本次视频优化改变。

## 多机实际接续验证

2026-11-02 06:16:52 UTC快照：[LIVE_VERIFIED.json](LIVE_VERIFIED.json)。2,344条视频全部生成／导出；评分72路并行，172.29.189.119为8路、node17／node12／node4／node8各16路。四台新增机器的32卡均实际加载并开始产出，最高单卡显存抽查约11.3GiB。已完成774条视频评分，执行错误0条；80份评分contract的代码／权重一致，全部使用GPU人脸与表情。GT的FID／FVD／人脸特征各复用423次。

当时按多机实际吞吐估算剩余评分和汇总约10–20分钟；该估算是此快照对应的剩余时间，不是永久状态或完成标记。完整结果仍以主运行目录COMPLETE.json／RESULTS.json为准。

## 用户要求提高并发后的更新

2026-11-02 06:23:15 UTC：[WORKER_EXPANSION.json](WORKER_EXPANSION.json)。四台推理机器评分容量从每GPU 2进程提升到4进程，评测机仍为每空闲GPU 8进程。仅重启控制器，采用已有SSH／本机worker并读取完成标记，运行中的评分没有停止或重算；后续并发容量可直接从protocol实时读取。新增测试覆盖采用worker后的存活、成功退出和失败退出。

全部排队分片已启动；核对时完成视频评分2265／2,344，执行错误0，已完成分片89／128、仍运行39、排队0。此时部分GPU空闲属于分片收尾，继续增加容量不会产生额外待运行任务。

[LPIPS batch64／128对照](BATCH64_VS128.json)：真实324帧视频，分别预热后ABBA，运行环境同时有正式评分。64平均1.889秒，128平均2.276秒；最大逐帧分数差5.96e-8。并发负载会影响计时，本批没有证据说明128更快，正式batch保持64。FID每段最多32帧，64→128也不会扩大其实际帧批次。
