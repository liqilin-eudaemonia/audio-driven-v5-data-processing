# L2L / ListenFormer / DyStream / Avatar Forcing / DualTalk：ResponseNet与Seamless

2026-11-12开始接入，**双200全部计划推理与评分已完成，COMPLETE.json已生成（24份结果JSON）**。**用户已确认：本轮仅测历史固定ResponseNet200＋Seamless200，共400条输入。** protocol.json保存已核验的完整来源清单，RUN_SCOPE.json将实际执行限制为其中固定400条，其余样本不启动生成或评分。

## 权重与输入

L2L四套官方人物权重均保留，视频标注L2L＋GPAvatar；ListenFormer使用已完成200轮的本地ViCo重训best.pt；DyStream、Avatar Forcing和DualTalk使用已核验的官方权重。本轮不重新训练，不使用500h短跑诊断权重替代正式模型。

复用已冻结的speaker音轨、listener首帧和GT reader。DyStream和AF保持ViCo共同输入适配：listener自身音频为零，AF user视觉为零。L2L、ListenFormer、DualTalk需要额外speaker动作条件，单独从实际speaker视频提取并记录；不得从listener未来或GT描述构造生成输入。DualTalk仅原生动作已接通，未经验证的RGB/全局头部映射不填入视频表。

## 评测与执行

使用原OmniResponse视频八项：PSNR、SSIM、LPIPS、FID、FVD、Emo-Acc、Emo-Div、ΔResp。ResponseNet face-padding64、原GT帧率；Seamless face-padding0、25fps。视频统一512评分，原生低分辨率方法须披露上采样。FID/FVD按各评测范围的全部特征汇总，不平均分片分数。缺人脸/支持不足沿用官方N/A和覆盖记录；技术错误重试或报错，不填造结果。五个方法均不生成listener音轨，音频指标N/A。

用户新增授权后，在node11物理GPU4和GPU7准入新任务，按剩余显存限量共享；不停止其他会话任务。按片段/固定生成批次保存指纹与回执；复用已有DyStream双200需核验权重、输入、视频与评分来源。输入缓存只读复用，新增产物均位于本目录。

源入口：[evaluation说明](../../evaluation/README.md)、[五方法交接索引](../../related/method_handoffs/README.md)。进度、具体执行命令和结果文件随真实阶段完成更新；有准备记录不代表已经生成。

## 结果与进度入口

- [数值对比表](RESULTS.md)：两套固定200分别列视频八项，另列原生动作诊断。
- [图像指标CSV](image_metrics.csv)、[动作指标CSV](native_motion_metrics.csv)：只写已完成结果，不用局部均值代替完整200。
- [实时状态](status.json)：动作、视频、特征、活跃任务与失败重试计数。
- `results/`：逐方法、逐数据集的完整JSON，含覆盖与N/A；`metrics/`保存25条评分分片。
- `COMPLETE.json`仅在所有计划结果确实完成后生成。没有该文件时，本轮仍在进行中。

## 推理配置与必要适配

| 方法 | 本轮模型与输入 | 输出及说明 |
|---|---|---|
| L2L | 官方Conan/Fallon/Stephen/Trevor四套模型；真实speaker DECA与mel128；seed137 | 固定每数据集8个25条批次；官方自回归/采样，30fps动作；使用公开GPAvatar渲染，明确标注L2L＋GPAvatar，不代表论文原渲染器 |
| ListenFormer | 本地ViCo重训200epoch的best.pt、训练集统计；speaker BFM、45维音频和listener首帧BFM | 45维音频由同一16k speaker波形重采样44.1k后调用原提取器；原生PIRender256/30fps，解码后PIL bicubic升512，按GT时间戳取帧评分 |
| DyStream | 官方权重、speaker音频和listener首帧；listener音频置零 | 已逐条核验历史400条输入与输出；复用原视频和原评分，不重复生成 |
| Avatar Forcing | 官方完整权重；speaker音频、listener首帧；user视觉和listener音频置零 | BF16、KV cache、NFE10、aCFG2、uCFG1、seed137；原生512/25fps，沿用原RGB导出规范化 |
| DualTalk | 官方模型；speaker DECA与音频，listener自身音频置零 | 原生25fps动作；DECA代理映射单独诊断，neck/global head含义尚未验证，因此视频八项N/A |

所有模型文件位置与SHA见[模型清单](MODEL_MANIFEST.json)。各方法原始官方命令及历史交接仍见[方法索引](../../related/method_handoffs/README.md)。新数据集适配不是论文原测试集复现，也不能仅凭跨数据集成绩认定官方代码有问题。

`features/`只使用speaker视频/音频及listener首帧。`evaluation_gt/`只用于评分，生成器不读取。Seamless从封存清单定位实际speaker视频、参与者与时间段；ResponseNet读取对应speaker片段。统一按30fps提取动作，保留来源时间戳、首帧/音频哈希及缺脸时的官方回退。完成回执先于下游任务准入。

## 执行与断点恢复

工作目录为本目录，协调器为`controller.py`。当前启动记录：
`../p2_vico_additional_baselines_20261107/jobs/reference_dualtest_20261112_controller_finish_v2/`。

在node11运行（协调器为CPU进程，按任务设置物理GPU4或GPU7；已运行时文件锁会拒绝第二个协调器）：

```bash
cd /private/liqi_workspace/project_code/audio-driven-v5
CUDA_VISIBLE_DEVICES=7 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1   compare/.envs/vico_more/bin/python   experiments/p2_reference_methods_dualtest_20261112/controller.py
```

协调器自动复用已完成片段/固定批次/评分分片；只有明确属于本轮的已知进程会被接管。各阶段日志在`processes/<阶段>/run_XX.log`，真实退出码为`exit_XX.json`；原先启动的AF双worker与首批评分日志在上述jobs目录的同名前缀下。单阶段技术失败最多自动重试3次后留存错误，其他可完成阶段继续。

准入按GPU UUID与剩余显存检查，只使用node11 GPU4/GPU7，启动准入时预留32GiB余量（共享卡运行中的其他任务占用会变化）；GPU7最多10个、GPU4最多4个阶段进程；2个特征worker、AF先2路，L2L动作就绪后降为1路以接续渲染、2个评分worker。所有输入清单只读缓存，固定400范围由`RUN_SCOPE.json`强制约束。L2L输入一次搬到GPU，GPAvatar身份跨四模型共享，生成/渲染/评分按就绪样本衔接；FID/FVD在最终200条特征上重新计算，绝不平均分片分数。

环境沿用现有AF、L2L、vico_more及AnyTalker环境；PyAV等评分依赖使用`/private/liqi_workspace/.cache/paired_video_metrics/python`中的Python3.10构建。不可混入其他任务的Python3.11版PyAV。

GPAvatar视频导出沿用已验证的两帧有界异步编码，25fps与30fps的串行/异步输出SHA完全相同，见[编码一致性记录](checks/export_parity/COMPLETE.json)。这项检查只验证编码重叠不改像素，模型是否跑通仍以真实生成回执为准。

GPU4新增后，GT动作提取可直接在GPU4接续，不再等待AF生成400条；已有GT缓存保留。

官方GPAvatar首帧追踪若明确返回无身份，保留`TRACK_UNAVAILABLE.json`并将该条视频标为N/A，计入400个请求的覆盖缺口；不填零、不借用其他身份，也不让渲染器无限等待。技术异常仍走失败重试。原生L2L动作照常保存评分。

DualTalk批量接入曾在8.8秒片段触发帧数断言：官方源码使用`ceil(audio_samples / 16000 * 25)`，浮点运算可能得到221而不是220帧。适配层现保留完全相同的运算顺序；已通过断言的旧产物不受影响，失败条目由原模型重新推理，不截断或伪造输出。

首条真实视频检查：[INITIAL_MEDIA_CHECK.json](INITIAL_MEDIA_CHECK.json)。AF、ListenFormer与四套L2L＋GPAvatar均核验512×512、原GT帧率与完整帧数；这仅证明链路实际产出，不代表全400已完成。DualTalk的8.8秒边界条目已由修正后的官方运算顺序成功重推，实际输出221帧。历史`FAILURE.json`/重试计数保留，成功与否以对应完成回执及当前进度为准。


## 2026-11-12 新增GPU4

用户明确授权新增node11 GPU4。检查时该卡不是空闲卡：利用率96%，约74GiB已用、66GiB可用。按容量叠加本轮任务，其他会话进程保留；未使用其他未授权卡。

GPU7原有生成/渲染进程继续运行；GPU4新增最多2个GPAvatar渲染worker，另接GT动作提取与视频评分。两张卡共用逐样本文件锁与完成回执，不重复生成已完成片段，固定双200清单、模型、seed和评分公式保持。GPU4启动位置与UUID校验见`GPU_EXPANSION.json`、`devices/`；`status.json`增加`gpu_free_mib_by_gpu`和`active_gpus`。

GPU4渲染已完成真实新增视频，评分已启动两路。GT提取旧工具中还有GPU7专用断言，本目录仅替换该函数的设备许可/UUID检查，原DECA模型初始化、权重与TF32设置不变；旧失败记录保留，新任务为`gt_gpu4_v2`并按容量接续。评分在GPU4/GPU7中选择有进程槽且显存余量较大的卡，避免一张卡有空位而另一张卡排队。

## 2026-11-12 Seamless配置编码恢复

共享speaker特征曾停在204/400（ResponseNet200＋Seamless4）：旧进程读取含中文的crop配置时采用ASCII编码，两个worker各重试3次后退出。新启动任务明确设置`PYTHONUTF8=1`和`PYTHONIOENCODING=utf-8`；原媒体reader、样本清单、模型、采样和已完成特征不变。已在C locale下用原reader读取全部200条Seamless配置通过，见[UTF8_READ_CHECK.json](UTF8_READ_CHECK.json)。新worker为`features_utf8_0/1`，旧失败计数保留并标为已替换阶段；实际恢复进度以status和新日志为准。

UTF8修复后，原先失败的Seamless条目`improvised_V01_S0223_I00000130_local_2bc1ad32236c`已实际完成172帧特征生成，见[UTF8_RECOVERY.json](UTF8_RECOVERY.json)。两路新worker均已启动，后续动作与视频按完成回执接续。

## 2026-11-12 收尾修复

Seamless GT动作提取的`p2_listen`包与同名CLI脚本冲突，现于加载提取器前明确导入`src/p2_listen`包；原模型和裁剪逻辑不改，新任务`gt_gpu4_v3`沿用200条已完成GT缓存。

ListenFormer Seamless评分的第00分片使用旧reader，而01–07分片期间共享reader增加了upright数据入口等逻辑，造成代码指纹不同。保留严格的分片代码/权重一致性检查，将旧00分片归档，仅重算对应25条视频的评分，不重新生成视频；见`READER_RESCORING.json`。全部分片一致后再按200条合并FID/FVD。

原生动作评分现按ResponseNet和Seamless各自200条就绪后独立启动，避免ResponseNet等待Seamless GT提取。指标定义和聚合范围保持；动作表仍与统一RGB八项分开。

完整Seamless6624为独立后续运行：[全量运行与Monitor](../p2_reference_methods_seamless6624_20261112/README.md)。本目录继续保留双200固定范围。
