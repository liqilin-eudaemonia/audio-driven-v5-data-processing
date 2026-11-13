# Seamless 全量事件打标：8 小时执行

2026-10-23 14:52 UTC 左右收到用户要求，目标约 22:52 UTC 交付已完成有效结果和剩余清单。用户已确认：处理正式清单主池 `primary/keep_candidate`，listener 有效裁剪框；不混入 onset_reserve，保留 training_admitted=false。train/dev/test 身份分别记录，不把 dev/test 自动标签视为人工 GT。

**2026-10-24 最终状态补充**：本轮已结束，22:45 停止领取新样本，最终导出约 22:51 完成。已处理 57,038 条，结构完整 54,820 条，双类检查支持均非零 41,363 条，失败 2,218 条，剩余 1,160,908 条；最终文件条数已经核对。尚未续跑，标签仍待审核。代码、数据位置、逐请求证据与复验方法统一见 [交接检查说明](P2_事件打标交接检查说明.md)。下文各带时间的小节保留为运行过程记录。

## 已授权方案

先对现有 Joy、MiniCPM-o-4.5、Qwen3-Omni 及新增 Qwen2-Audio-7B-Instruct 做相同 listener 媒体对比；按格式稳定性、可用覆盖、可见/可听证据冲突、时间定位和吞吐选择批量方案。没有人工事件真值时只能给工程选择和质量待审结论，不能编造准确率排行榜。此前小样结果见 [验证报告](P2_事件打标_真实数据与模型小样验证.md)。

批量只读取 `sources.listener` 视频及 `normalized_audio.listener` 独立音轨，核对 person_id 与 source member；按源秒区间截取，按原像素参考框取整裁剪、bicubic resize 512×512。模型接收局部秒时间，不送 speaker 转写，也不使用 speaker 媒体冒充 listener。原清单、源媒体、裁框提议和质检结果只读。

开放 discovery 和目标 audit 分开请求。失败结果原样归档；未知区间不能补成负例。每条输出绑定来源、时间、框、prompt、模型与代码版本，写入独立结果目录。生产采用共享任务索引、原子完成写入和可恢复认领；跳过缺框和无效来源时保留原因，不记作成功。通过结构校验的草稿也不等于已人工验证的训练标签。

在 zz-dev / node8 的可用资源上逐步增加并发，根据显存余量、实际吞吐和失败率调节，不终止其他作业。优先使用已有可用 vLLM 环境；不覆盖共享依赖。预处理与推理并行，避免为 120 万段预先输出原画视频。

## 监测与恢复

计划每 20 分钟输出一个完整进度快照，包含候选/有效框/已处理/结构通过/有效覆盖/失败/剩余、完成速度、预计时长、每卡资源及 worker 心跳。已知暂时性服务/连接故障有界重启与续跑；未知错误保留栈和受影响样本，由当前代理检查修复。监测脚本本身不冒充能理解和修复任意代码 bug 的代理。

当前会话没有可直接定时唤醒此聊天代理的通用调度工具。检测脚本将持续保存进度和报警文件，当前代理在活跃执行期间读取并处理；不能将落盘日志说成已自动推送到此会话。运行配置、PID、日志和最终方案会在启动验证后补充本文件。

八小时交付按实际完成数量计算；不能将未处理、截断、坏 JSON、unknown 或被跳过样本计为完成。训练主池原始候选为 1,200,212 段、2,002.627 小时，是否能八小时全量完成取决于实测吞吐；用户已同意吞吐不足时交付完成部分及剩余清单。

## 2026-10-23 15:25 UTC：选型与正式运行

推荐批量组合：**JoyAI-VL-Interaction 视觉 + Qwen3-Omni-Instruct 纯 listener 音频**，8 fps、每帧先裁到 512×512，开放发现与独立目标检查分开调用，使用 vLLM 的 JSON schema 约束解码。没有改为固定 backchannel 类别表；只有后续训练 audit 仍检查点头/发声。原始输出、失败、模型及输入身份均保留。

全部模型已在相同 50 条样本测试，另做相同前 12 条的约束解码对比，随后对所选双路复验完整 50 条。[比较明细](../experiments/listen_labeling_full_20261023/model_comparison_report.json)包括原始文件链接、失败原因、覆盖时长和请求耗时；这些样本已经用于接口调试，不能作为独立准确率测试集。

| 分支 | 原格式 50 条完整两阶段 | 约束解码 12 条完整两阶段 | 所选分支约束解码 50 条复验 |
|---|---:|---:|---:|
| Joy 视觉 | 35/50 | 11/12 | 44/50 |
| MiniCPM-o-4.5 视觉 | 9/50 | 8/12 | 未扩大 |
| MiniCPM-o-4.5 音频 | 24/50 | 9/12 | 未扩大 |
| Qwen2-Audio-7B-Instruct 音频 | 0/50 | 8/12 | 未扩大 |
| Qwen3-Omni 音频 | 36/50 | 12/12 | 50/50 |
| Qwen3-Omni 视听联合 | 2/50 | 11/12 | 未扩大 |

Qwen-Audio 已下载至 `/private/ckpts/Qwen2-Audio-7B-Instruct`；[下载版本记录](../experiments/listen_labeling_full_20261023/qwen_audio_download.json)。它在当前长结构化 prompt 下经常输出单引号字典，约束解码后仍有倒序区间；本轮没有因为模型小就默认选择它。MiniCPM 有覆盖区间过少、截断及文字与事件列表不一致的问题。联合 Omni 的检查支持与稳定性未超过分路方案。

**表中只是结构通过率，不是识别准确率。** 所选 50 条复验中，Joy 有 42 条、Omni 音频有 36 条同时满足两阶段成功且 audit 支持大于零；它们的交集才可能提供双路完整监督。没有人工真值，不能报告 precision/recall 或宣称模型客观最优；当前选择是这套任务接口下更合适的工程方案，标签仍待复核。

生产另加入最多一次语义校验重试：如模型报告了事件却未把其区间列入 reviewed，给模型校验错误并重新提供原媒体，要求重新判断。不会直接扩大 reviewed、修剪事件边界或将 unknown 改为 negative。每次响应和实际 prompt 均保存；仍失败则单列。重试属于新增生产行为，不能套用上述未重试的通过率冒充生产成绩。

索引最终读取 **1,225,293 条主池候选**，有效框队列 **1,217,946 条**：train 1,193,597，dev 10,206，test 14,143；缺框 7,347 条单列。有效框对应 train 1,992.060 小时、dev 17.510 小时、test 23.294 小时。索引中修正了源清单 6 位小数舍入带来的约 1 微秒时长差，允许不超过 2 微秒差异；没有改变实际源秒区间。见 [queue/index.json](../experiments/listen_labeling_full_20261023/queue/index.json)。

正式程序约 15:14 UTC 启动，先 8 worker；15:23 UTC 提升为 32 worker，模型副本按实时显存逐步加入。15:23:50 UTC 首个 20 分钟报告记录：已处理 111 条，结构完整 92 条，其中双类有效支持均非零 78 条；这些是启动阶段快照，不是最终数量或稳定吞吐。

- 生产入口：[run_listen_labeling_full.py](../scripts/run_listen_labeling_full.py)。64 个互斥分片，原子写结果、阶段检查点和独立进度游标；崩溃后不会把同一完成记录计数两次。
- 源读取：[full_source.py](../src/listen_labeling/full_source.py)。以 listener 视频 TAR member、listener PCM 采样边界和实际视频 PTS 定位，图片时间原点与音频第一个采样对齐。clip 音频哈希已与测试包同条独立 WAV 送模内容核对一致。
- 生产配置：[production_config.json](../experiments/listen_labeling_full_20261023/production_config.json)。端点列表支持热加入同权重副本；并发控制为 `production/concurrency.json`。模型路径和命令见 `*.vllm.process.json`。
- 结果：`experiments/listen_labeling_full_20261023/production/results/<shard>/<sample_id>.json`，包括开放事件、audit、完整成功时的 canonical_draft、输入证据和失败历史；这些 source-backed 标签尚未绑定最终物理训练媒体，不直接导入 P2。
- 检测：[watch_listen_labeling_full.py](../scripts/watch_listen_labeling_full.py)，每 60 秒检查，每 1,200 秒生成一个报告；[最新状态](../experiments/listen_labeling_full_20261023/watchdog/latest.json)、[报告目录](../experiments/listen_labeling_full_20261023/watchdog/reports/)。只恢复显式注册的本任务进程，一小时最多 3 次；不停止其他任务。

模型部署修复：独立 vLLM 环境需将自身 lib 加入进程的 LD_LIBRARY_PATH，解决 libstdc++ ABI 问题；Qwen-Audio 按自身配置使用 8,192 上下文。Omni 纯音频服务把 image 上限设为 0 时遇到编译期 meta/CUDA 张量冲突，保留 image=1 的模型初始化后正常启动，**实际请求仍不发送图像**。新增服务启动根据实测空闲显存计算预算，计划保留约 12 GiB；其他作业会改变余量，因此不能把某次资源快照当作永久可用量。

worker 在 22:45 UTC 左右停止认领新样本，在途请求可完成写入；八小时交付目标约 22:52 UTC。检查脚本的本轮报告窗口已延长到 22:52 UTC，覆盖计划的 22:50 UTC 最终导出。最终会汇总成功、部分失败、无有效支持及剩余清单，不把未处理数量藏入“完成”。

## 2026-10-23 16:02 UTC：八卡运行及质量复核

生产已提升至 64 worker，GPU 0/1/2/3/4/7 各一套 Joy，GPU 5/6 各一套 Omni 音频服务。GPU 2 已结束实验并重新启动标准逐帧输入服务，确认产生了正式队列的成功结果；没有将实验视频适配器混入生产。16:02:27 UTC 快照：已处理 4,348 条，完整结构通过 4,187 条，失败 161 条，其中双类检查支持均大于零 3,195 条。8 卡均在执行推理，瞬时利用率 71–100%，所有注册服务健康，无报警。其他作业仍保留。

最近一段实测约每小时 7,300 条结构完整结果，120 万条队列无法在本轮八小时内全量完成。已按用户确认执行部分交付和剩余清单方案。这个速度只用于量级判断，不能作为最终数量承诺。

抽查了三个 Joy 点头阳性样本的实际 512 输入帧，发现持续低头、转头及身体移动可能被标为点头；其中一个事件覆盖整段，但抽查只看了前三秒，不能据此把全段定为人工负例。尝试更严格的“俯仰往返”文字要求后，模型仍可能虚构恢复动作描述，不能把加长 prompt 宣称为质量修复。

另用原生视频时间编码复测同一 50 条、100 个请求，保留实际采样帧和 PTS，不降 fps、不截成默认 32 帧。原生输入 discovery 33/50、audit 48/50，完整两阶段 33/50，低于原逐帧方案的 44/50；**不部署此变体**。详见 [原生视频复测汇总](../experiments/listen_labeling_full_20261023/native_video50_summary.json)，原始响应在同目录 `native_video50/`。这些证据只支持当前工程选型，尚不能证明点头准确率达标；全部标签继续保留 `human_verified=false`、`training_admitted=false`。

本轮离线回归 37 项通过，包括 listener 身份、原框裁剪到 512、源区间与音轨对齐、阶段失败重检、结果原子落盘和游标恢复不重复计数。[代码绑定验证记录](../experiments/listen_labeling_code_validation/result.json)与实模型比较分开，mock 测试不冒充实模型成功。

### 读取结果

[导出入口](../scripts/export_listen_labeling_full.py)每 20 分钟生成一个独立快照；[最新导出索引](../experiments/listen_labeling_full_20261023/exports/latest.json)中的 `directory` 指向当前快照目录。导出进程已加入监测和有界重启。结果文件包括：

- `schema_pass.jsonl`：四阶段完整且通过结构校验的标签草稿，附源清单位置、listener 身份、源秒边界、裁剪变换和原始结果路径。
- `supported_both_classes.jsonl`：上述结果中两个目标类均有非零检查支持的子集。仍不是人工确认可训练的子集。
- `open_events.jsonl`：开放发现描述，包括部分失败样本中已经成功的发现阶段，另标 `complete_sample`。
- `failed.jsonl`：失败阶段与原因，完整响应仍在原始结果 JSON。
- `remaining.jsonl`：最终快照列出未完成候选；周期快照暂为空，索引明确标注 `remaining_manifest_written=false`，不得把空文件理解为队列已完成。

22:50 UTC 的最终快照计划生成完整剩余清单。完成数按快照实际捕获的结果文件计算，在途晚到结果不会被假装已经包含。缺框 7,347 条另见 `queue/excluded.jsonl`，不混入待推理队列或成功数量。
