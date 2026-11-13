# P3 / Bernini 固定文本倾听实验计划

日期：2026-10-21。状态：已进入实现与小样本工程验证；正式训练和效果评估尚未完成。实现入口、数据修订和运行记录见 [P3_Bernini_固定文本实现与运行.md](P3_Bernini_固定文本实现与运行.md)。

## 最新数据要求（覆盖下文初始 V4 数据处理设定）

用户确认：解决音量过小/不一致；每个数据子话轮不超过 10 秒，优先在 speaker 停顿或词间隙切分，连续且不重叠，尽量避开 listener 短反馈；找不到合适边界时才硬切。

当前 V5 将 457 个父区间切为 872 个子话轮，总时长仍为 6574.949 秒；100 条原始参与者音轨按整轨统一响度，目标 −18 LUFS，再供所有子话轮共用。训练改用 `data_previews/seamless_subturns_v5/manifest.json`，保持已固定的 10/2/2 连通组划分，窗口数为 train/dev/test=1160/224/228。训练窗口的 8k+1 帧和最大 8.04 秒配置保持不变。归一化后的音轨仍保留静音及轨道内部的动态，不按静音片段单独放大。

下文 V4 的 457 段与 420 个不少于 4 秒区间是初始数据快照；现行数据入口、响度/同步核验与复现命令以实现文档为准。

## 1. 实验问题和范围

检验：在不提供逐样本 GT 回复文本、动作描述和反应时间表的条件下，speaker 音频能否通过 Omni 的目标音视频特征规划，驱动 LTX 生成合理的 listener 面部视频与自身声音。

首轮为离线、给定长度的 listen-only 生成。输入为同一窗口的 speaker 音频、listener 首帧、固定任务文本，以及窗口长度/帧率等几何信息；输出为 listener 音视频。允许利用整个输入窗口，不声称实时、因果或自主预测话轮结束。输出长度由输入窗口确定，不从 listener GT 发声时间推导。

当前不加入 Whisper 分支、GLARE 六类反应监督、逐样本动作 caption 或自主文本计划生成。保留 Gemma 原生文本条件与 Connector 条件。第一轮结果只用于判断可行性和定位瓶颈。

## 2. 当前数据事实

依据 `data_previews/seamless_speaker_span_v4/manifest.json`：

| 项目 | 当前值 |
|---|---:|
| interaction | 50 |
| V4 连续区间 | 457 |
| 总时长 | 6574.949 秒，109.6 分钟 |
| 录制 session | 16 |
| 参与者 ID（当前 V00 范围） | 30 |
| session 与参与者共同关联的连通组 | 14 |
| 片段时长最小 / 中位数 / 最大 | 1.021 / 9.006 / 118.847 秒 |
| 小于 4 秒的区间 | 37 |
| 反馈 / 无转写发声 / 较安静候选 | 163 / 52 / 242 |

候选类型来自自动规则，不是人工确认标签。共享人物造成两个跨 session 组：S0314+S0426、S1469+S1997。仅按 interaction 或 session 单独划分都不足以确保当前人物隔离。

这些素材来自原 naturalistic/test/V00；用于本实验训练后不能再将其称为未见过的官方 test。这里另建 local train/dev/test。

## 3. 数据准备与划分

1. 先按共享 session 或参与者形成的 14 个连通组划分，拟定 10/2/2 组为 train/dev/test；按时长尽量接近 70/15/15，并参考自动声音候选分布平衡。具体 ID 在训练前固定，seed=20261021，不按模型结果挑组。
2. 先划分再生成训练窗口。重叠窗口、同 interaction、同 session、共享人物均不跨集合。保留独立 test，整个调参过程只用 dev；最终锁定配置后才运行 test。
3. 使用 V4 边界和固定裁剪框，从原始视频与独立 WAV 重新提取训练输入，避免把预览 AAC 和重复编码视频作为正式音视频训练源。
4. 首轮训练以原区间不短于 4 秒的 420 段为候选，最终数量取决于质量检查与分组；37 段短区间单独保留为短片鲁棒性检查，不删除。
5. 将长区间采样为最长约 8 秒的训练窗口，不重定义主话轮，不跨 V4 排除区间或拼接不连续片段。视频保持 512×512、25 FPS；LTX 时间结构是 8k+1 帧，最大取 201 帧（8.04 秒）。短窗口取不超过可用范围的合法帧数，所有模态共享实际窗口。记录余帧、未覆盖尾部和事件截断情况，不能悄悄丢掉反馈尾部。
6. 先形成有限的窗口清单，train 可以采用约 4 秒步长和末端补窗；dev/test 使用预先固定的确定性窗口，不按模型表现换窗口。不能把重叠窗口当成独立对话样本报告统计置信度。
7. 首帧使用当前窗口第一帧；不使用后续 GT 图像作为参考。不提供 listener 声纹参考，因此首轮不把复现真实 listener 音色列为验收要求。
8. 音频分别按 Omni 和 LTX 编码器要求重采样，保留共同时间起点、原始静音和相对音量；不各自删除静音，也不把 source speaker 音轨混入目标。

质量核验先抽 train/dev 共约 24 个窗口，覆盖短反馈、较安静、无转写发声、首尾重叠、长短片段及多个分组。听音核对串音、同步和词边界；发现系统性问题先修数据。test 中需要人工裁决的评测标签在最终评分时独立处理，不参与选点。

## 4. 固定文本契约

Omni 提示，所有样本相同：

```text
The reference image shows the listener. The input audio belongs to
the other participant, who holds the conversational floor.

Generate the listener's synchronized facial video and own audio
throughout this listening interval. Model the listener's expressions,
head movements, and vocal feedback in relation to the incoming speech.
Preserve periods without vocal feedback and maintain the listener's
identity. The listener remains in the listening role throughout.
```

Gemma 提示，所有样本相同：

```text
A continuous close-up video of the person in the reference image
listening to another person speaking off-camera. The camera remains
fixed, and the person's identity and background remain consistent.
Facial expressions, head movements, and brief vocal feedback, when
appropriate, form a natural listening response. The person remains
in the listening role throughout. The soundtrack contains the
listener's own vocalizations and periods without vocalization.
```

两套文本分别冻结版本和 SHA，训练/验证/推理一致。Gemma 条件可按固定文本和编码器版本复用，但需检查音视频 connector 输出格式。首轮不加逐样本外观 caption，以减少变量。

不把 GT listener 转写、事件类型、VAD、情绪、是否安静或反应起止加入 prompt、prefix 或推理布局。合法输出长度/特征槽位由 speaker 输入窗口和编码规则确定；不能通过只为发声部分创建音频槽位泄漏活动时间。

## 5. 复用代码与必要改造

参考：`audio-driven-v4-p3-autoselect-20261013/src/omni_av_p3_online/` 的在线 Beta(10,1) P→C0→A 实现；不是早期离线 Oracle/Pred25 混合 hidden 路线，也不启用 Whisper 版本。实施前固定代码提交和配置，不改历史实验目录。

已确认的旧接口问题：

- `omni_av_p3/data.py` 筛选有非空 reply 的样本，并构造 GT plan 和固定回复时间表。
- `omni_av/omni.py::encode_prefix` 序列化 `TextPlan`。
- `omni_av_p3_online/data.py` 检查 text cache 与逐样本 TextPlan 相等，并硬编码 Train91009/Dev48 数量。
- 旧版感知视觉特征默认 2 FPS，temporal patch=2，约每秒一组时间特征。
- `select_contexts` 将 prefix 和各模态未 MASK 目标 hidden 送入生成 Connector；MASK 位置参与特征预测监督。推理则先全 MASK 预测，再重新读取所有已填充槽位。

因此需建立独立 listen manifest/loader：允许无文本、无发声目标；直接构造固定任务 prefix；去除 Reply/Delivery/Behavior/Vocal 计划和回复时间表；更新完整性校验与规模检查；重新提取所有目标特征和 VAE 缓存。

初始化默认复用原始预训练 Omni-3B 与 LTX-2.3，新增模块重新初始化；不直接加载旧 P400/A300。这让首轮避免旧详细计划及 listen+response 适配和未知旧训练样本重叠的干扰。旧 A300 温启动可作为后续单列实验，先核查旧训练数据重叠，不能与干净初始化混为同一协议。

## 6. 缓存与特征时间分辨率

必须生成：

- speaker 音频和 listener 首帧的 Omni 输入特征，以及固定文本对应输入 embedding；缓存输入，不缓存随后可训练 Omni 的最终 hidden。
- listener 目标视频原生感知特征、目标音频原生感知特征；目标覆盖完整窗口，含静音。
- listener 音视频 VAE latent、参考首帧 latent、音视频共同时间布局。
- 固定 Gemma 文本条件及来源、归一化、模型版本、时间轴/形状的元数据。

首轮主配置建议目标视觉特征取 8 FPS（temporal patch=2 时约 0.25 秒一组）。这是对旧 2 FPS 的明确改动，不改变最终输出 25 FPS。先在 16 个窗口比较 2/8 FPS 的时间位置、有效帧数、token 量、显存与动作变化表示；选用配置在主实验前固定。若 8 FPS 成本不可接受，记录原因后选择 4 FPS，不能在不同实验组之间静默切换。

不添加时间 RoPE 新分支或局部 attention 以免同时改变架构。先沿用已有规划布局的时间坐标；注意规划端坐标不等于 LTX 已获得显式条件时间对齐，若后续确认时序是瓶颈，再设计独立改造。

## 7. 执行阶段与初始预算

一切预算以新窗口数 N 为基准，报告 optimizer step、样本曝光次数、有效 epoch 和实际 GPU 时间。下列为首次实验预算，不是预期收敛步数。

| 阶段 | 训练内容 | 冻结内容 | 初始预算 |
|---|---|---|---|
| S0 工程/小样本拟合 | 从 train 取 16 个窗口，检查 P、C0、A 前反向与生成；额外做 Oracle 条件下的小集拟合 | 按相应阶段 | 每阶段先 2 个更新与保存恢复；小集拟合最多 200 更新 |
| P 特征规划 | Omni LoRA、MASK/模态 embedding、音视频特征 Connector 与 FM decoder | Omni 基础参数、感知编码器；不加载 LTX 训练 | 最多 20 epoch 或 2000 更新，先到者为准 |
| C0 生成适配 | 音视频 GenerationConnector | P 阶段选定的规划器、LTX、Gemma、VAE | 最多 10 epoch 或 1000 更新 |
| A 可选联合适配 | GenerationConnector＋DiT text-attention LoRA；暖启动后联合 Omni LoRA/特征头 | 所有基础参数、Gemma/VAE | 最多 5 epoch 或 500 更新；前 min(50,总更新10%) 步冻结规划器 |

训练结构沿用在线 Bernini：

- MASK 比例使用 `Beta(10,1)`，按模态选位置；全静音样本也有完整音频特征槽位，不删除。主配置不另加 GLARE 行为损失。
- P 损失为音频＋视频特征 FM；C0 为音视频生成 FM；A 联合上述损失。音视频特征权重初始各 1，分别记录数值和梯度尺度，不能只看总 loss。
- 起始学习率：Omni LoRA `1e-5`，特征头 `1e-4`，GenerationConnector `3e-5`，DiT LoRA `5e-6`；Omni rank16/alpha32，DiT rank8/alpha8。使用 AdamW、梯度裁剪 1.0、阶段内 5% warmup；这些是小数据实验建议值，不冒充原论文最优值。
- effective batch 初定 8，通过梯度累积保持一致；一台空闲 H200 做压力检查后决定 1/2/4 卡，初轮不默认占用三机 24 卡。显存和吞吐实测后再估算时间。
- 主训练保持自然窗口分布，按时长/形状分桶降低 padding。若出现只学静音，在同一初始化与预算下增加“反馈候选过采样”单独实验，不能把候选类别混入文本条件；dev/test 不改变分布。
- 每个有效 epoch 做一次固定 seed 的数值验证，最长不超过 100 更新一次；C0/A 每 2 epoch 或 200 更新做一次 Dev12 生成，取更频繁者。P 需检查全 MASK 推理预测，而非仅 teacher-forced 部分可见 loss。

P 用 dev 的全 MASK 特征质量与条件利用诊断选点，最多留两个候选进入共同 C0 探针；C0/A 用固定 Dev12 生成评估，不仅按 loss 选择。

S0 要检查实际参数更新、冻结参数不变、GT 排除、全静音有效性、时间坐标、重采样、解码长度、保存恢复和一次完整 Pred25 生成。Oracle 小集拟合不能改善时先定位表示/渲染接口；不直接全量加步数。对候选无转写发声需试听确认后才能把它当作笑声样本。

## 8. 必做对照

固定相同参考图、Gemma 文本、输出长度、采样器、输出 seed。默认规划 Pred25、每轮特征 FM 5 步；LTX 使用此前核验的原生调度、Euler30、video CFG3/audio CFG7，不启用 STG/rescale/upscaler。工程阶段确认当前模型和接口支持后锁定这些配置。

| 对照 | 条件 | 回答的问题 |
|---|---|---|
| B-text | 原始 LTX＋首帧＋固定文本，无 Omni/Connector | 基座仅凭倾听提示会生成什么；这是参考基线，不是严格等参数对照 |
| M-pred | 当前 P3 检查点，speaker 音频，全 MASK Pred25→回填→hidden→Connector | 主结果：是否能自主产生有意义的倾听反应 |
| D-oracle | 与 M-pred 同一检查点，将真实目标特征完整填入后重新读取 hidden | GT 辅助诊断，区分规划误差和特征/渲染问题；不参与自主生成排名，不视为保证达到的数学上界 |
| D-prefix | 同一检查点，仅保留 prefix hidden，移除目标槽位 hidden | 生成是否使用目标特征规划；属于推理消融，有条件分布变化，不能等同重训基线 |
| D-audio-swap | 同一检查点、参考图、长度和各阶段 seed，交换为另一段等长 speaker 音频并重新规划 | 输出是否依赖输入音频；不能沿用原 prefix 或规划结果 |

B-text/M-pred/D-oracle：预先按组和声音候选覆盖选定 Dev12，每条 3 个固定 seed，共 108 个结果。D-prefix/D-audio-swap：固定 Dev8、各 3 seed，共 48 个结果。总计 156 个诊断输出/检查点；普通中途选点仅跑 M-pred Dev12 单 seed，完整矩阵用于阶段终点或少量候选。

若 dev 合格窗口不足相应数量，则使用全部并报告实际分母；不从 train 或 test 补足。seed 固定为 137/251/389。样本、seed 和失败均进入报告，不只展示好例子。

A 阶段只有 C0 显示出条件利用且生成能力仍受限时启动。若需要归因“开放 LoRA 是否有收益”，从同一 C0 分出 A 和继续 C0 两支，匹配额外曝光与推理条件；不能只拿训练更久的 A 与旧 C0 比较后归因。

## 9. 评估内容

无需先为整批数据打六类视觉反应标签，但需要小规模人工审阅和输出测量。

- 可用性：生成成功率、有效时长、音画长度、身份/背景稳定、明显面部变形、杂音、speaker 串音或复制、是否变成长篇正式回答。
- 声音活动：按人工抽查的 dev 参考区间评价静音区间误发声、反馈遗漏、发声次数和时长。已有转写/VAD仅提议候选；没有人工核验时只报告描述统计，不称真值准确率。
- 时机：短反馈起点与持续时间、与 speaker 语音的关系、音画是否协调；对单一 GT 的时间偏差只作为参考一致性，不等同唯一合理行为。
- 视觉：点头/表情是否存在、是否自然、是否机械循环。先用盲审和头部/关键点轨迹辅助测量，不将 GLARE 六类全量标注作为训练前置条件。评测工具失败应单列覆盖率。
- 文本/音质：无指定回复文本，不把相对 GT 的 WER 设为主指标；ASR用于观察是否出现长回复、复制 speaker 内容。对极短反馈/笑声不强行使用 UTMOS 或 SyncNet；仅在工具适用的片段报告并附覆盖率。
- 条件利用：同 seed 换音频后是否改变反应，且对新音频更合理；输出变化本身不能证明利用正确。至少配对盲审，不只比较像素差。
- 多样性：三个 seed 的动作和发声分布，避免固定模式或随机乱动。不要求所有 seed 都复制同一个 GT 事件。

人工按 1–5 分记录反应适宜性、时机、视觉自然度与声音自然度，随机隐藏方法名；不能确定时单列。可先由一位审阅者完成，明确局限；正式结论增加独立审阅者。统计按原始连通组聚合，dev/test 各仅两组时不声称可靠的总体显著性。

锁定主模型后，在 test 固定窗口上运行 B-text 与 M-pred，各 3 seed；Oracle若补充仅列诊断。报告全部覆盖率、每组结果与失败。旧检查点或新模型若曾训练过这些原始样本，必须重新界定 held-out 含义。

## 10. 继续、停止与下一步

工程硬门槛：没有 GT 文本/目标特征进入主推理条件；静音样本可正常训练；窗口及模态同步；保存恢复正确；无 NaN/OOM 未解决。任一失败先修复。

预设首轮继续标准：主模型完整评测生成成功率至少 95%；配对盲审中相对 B-text 的倾听适宜性胜率（平局计半）超过 60%，同时静音误发声/明显面部异常比例不增加超过 5 个百分点；换音频诊断出现可解释的输入相关变化。样本很小，这只是工程推进门槛，不是统计显著性或论文达标判断。

若连续 3 次 Dev 生成评估无改善，或只增加动作/声音却适宜性下降，则暂停加步数，保留检查点并归因：

| 现象 | 优先排查 |
|---|---|
| Oracle也明显失败 | 感知特征是否保留细微动作、Connector、Gemma/LTX条件竞争、渲染器适配 |
| Oracle较好、Pred较差 | 全 MASK 规划、训练/推理分布差异、静音占比、音频条件利用 |
| 动作和发声存在但时机漂移 | 目标特征时间分辨率、时间布局、LTX读取条件的对齐能力 |
| 一直静音或始终泛化点头 | 数据不均衡、目标特征敏感性、隐藏条件被忽略 |
| 换音频几乎无变化 | prefix构造、声学特征、规划器是否有效利用条件 |

后续每次只增加一个因素：候选采样平衡、时间对齐机制、活动辅助监督或详细文本计划。当前不预设所有问题都由增加视觉反应标签解决。

## 11. 实施顺序与产物

先固化分组与窗口 → 24 条数据核验 → 固定文本与 loader 改造 → 16 条特征/工程与 Oracle 拟合 → P → C0 → 完整对照 → 决定是否 A → 锁定模型并运行 test。

产物应包括：split/window manifest、质量检查与排除清单、prompt及配置哈希、缓存索引、梯度/恢复验收记录、各阶段 checkpoints、全 MASK 规划报告、逐样本生成与评分、对照页面和最终结论。只发布一个当前实验页面，不重新开放已停用的历史 HTML。

本计划不对实际训练耗时作未经测量的承诺；由 S0 的显存、每步吞吐、缓存速度和窗口数量给出实施预算。
