# P2 接入 OmniResponse 评测

用户已明确授权准备参考工作的评估代码，正式评测数据稍后提供。此次交付为可运行代码、输入契约、离线依赖/模型核对、测试和使用说明；不改正在训练的源码/环境，不启动正式 dev/test 生成。

## 方案

直接运行原目录脚本需要人工重命名/拼接路径，且会跳过坏文件，无法保证 E0/E1 的同一评测集合。重写所有模型公式容易偏离参考。选用独立适配层：固定参考源码哈希、直接调用其 FVD/FD/文本数学实现，适配 manifest、设备、缓存、缺失状态与聚合；使用已存在的独立 metrics/sync Python 和离线权重，默认 CPU 验证。

## 范围与口径

- FVD：I3D 原权重、uint8 RGB、一次 bilinear 224 缩放、rescale=True/resize=False、population covariance、原 eps=1e-6 sqrtm。支持 32/256 帧连续窗口、stride=1；样本 ID 和固定 crop seed 决定起点，GT/生成共享起点，各 generation seed 分别算完整集合。无补帧；短片段有显式 eligibility 与覆盖率，不伪装全量结果。用户已确认默认 FVD-32，保留显式 256 帧配置；正式数据未提供前不产生正式分数。
- FD：直接对用户提供的同语义 T×C 特征调用参考 sample-covariance FD，manifest 必须声明 feature schema；不是 Inception FID，不从训练 Connector 特征擅自替代 3DMM。
- UTMOSv2：离线原模型，固定随机 seed、显式 CPU/GPU。读取 generated/GT listener 独立音频，绝不误用 speaker 音轨。输出逐片段 MOS、两侧均值、静音/失败覆盖。阈值 RMS<1e-5 的音频标为不适用；此为本项目的明确适用集，不称论文完整复现。
- SyncNet：同 S3FD、SyncNet 权重、MFCC/5视频帧/20音频帧、vshift=15。现存 tracker 支持 CPU；使用其裁脸流程并固定 min_track=5 适配短视频，要求单 listener，过短/无脸/静音有明确状态；device 适配只替换 .cuda()，距离直接用参考 calc_pdist。输出 LSE-D、LSE-C、offset、每轨迹覆盖，按参考将轨迹汇总并另报每片段均值。同步音轨必须为对应 listener，不接受 speaker 当参考音轨。
- 文本可选：METEOR、ROUGE-L、BERTScore P/R/F1、Distinct-1/2。需要显式 generated_text/gt_text 与来源说明；不自动以 speaker 文本作为生成文本，不偷偷启动 ASR。空文本计缺失并报告，不丢失行。

## 工程接口

新 src/p2_omni_eval + scripts/p2_omni_eval.py。接收 pairs manifest（sample_id、sample_seed、generated/gt_video、独立 generated/gt_audio，可选 features/text）；可以从已有 GLARE generation work 导出，并加载其 generation/complete 身份，GT audio 从单独 targets 文件加入。支持 E0 新 preflight release 的生成入口，原 GLARE evaluator 通过独立启动器加载对应训练包，不改原训练源码。

prepare 冻结 manifest 内容、源码、参数、环境版本、引用模型与媒体哈希。evaluate 按 sample/seed 运行各指标，完成标记最后写出，损坏缓存拒绝复用、失败可重试。每项单独报告成功/不适用/失败计数与样本 ID，计划集合缺失时 full value 为 null，并可另报清楚命名的 supported-subset 值；多 seed 不把 GT 重复放入同一个分布池。JSON/Markdown 输出及配对比较都检查共同 GT、ID、种子、协议。所有必要失败导致非零退出，不拿部分成功冒充完整结果。

## 验证

公式对照原函数、窗口确定性与短视频、分母与空集、多 seed、GT/生成绑定、静音、坏音轨/错配、特征维度、非有限数、协议变更/缓存损坏/断点；真实 CPU 运行 I3D、UTMOS、SyncNet 及文本模型，复用小批 debug 作为工程样例，不报告为正式 E1/E0 效果。保留原训练源文件哈希并检查当前训练持续推进。

## 独立审阅已采纳

显式 listener 选轨；CPU 子进程屏蔽 GPU；SyncNet 短轨迹有界平滑与低支持标记；固定 GT 窗口；轨迹/片段/帧/语料的不同统计单位；共同支持集合重新评分；原模型/辅助模型/环境/ffmpeg 固定。真实测试还修复上游 AVI 默认 MP3 延迟导致的帧数增加，保留 PCM 并复核时间轴。
