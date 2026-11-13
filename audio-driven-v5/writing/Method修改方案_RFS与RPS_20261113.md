# Method 修改方案与模块论证

记录日期：2026-11-13。

本方案针对作者在本轮会话中提供的四个方法小节：Direct Speaker-Audio Conditioning、Response Feature Supervision、Reaction Presence Supervision、Training and Inference。保留现有模块和计算过程，重点补足设计理由、模块之间的承接及必要的信息流。

**状态：修改方案已记录，具体改写待讨论通过。RFS／RPS 名称已确定；本次未修改 LaTeX 正文、训练代码或配置。**

参考：[Method 写作借鉴](Method写作借鉴_GLARE与OmniPersona_20261111.md)、[Feature 与 Event 监督详解及结构图画法](Feature与Event监督详解及结构图画法_20261110.md)、[补充材料待补内容清单](补充材料待补内容清单.md)。返回 [写作总索引](README.md)。

## 1. 修改原则

作者要求方法段落具有连续的设计论证：

> 为了实现什么功能 → 已有路径如何实现、已经提供什么 → 尚未覆盖什么要求 → 怎样的设计更符合任务需要 → 本文改变什么、具体怎样实现 → 这一改变带来什么直接作用 → 后续还需要补充什么。

每个模块先让读者理解为什么需要它，再介绍表示、网络结构和公式。已有路径既可以是有引用依据的先前工作，也可以是本文前面已经建立的生成路径；不必每节都泛泛批评所有已有方法。

- 保留作者原有的换行、缩进和 LaTeX 排版，仅修改讨论通过的内容，并逐项说明改动。
- 保留有实际论证作用的动机句，尤其是 “To make brief and subtle feedback a direct learning target”。
- 描述设计直接带来的作用，例如增加目标特征监督、显式约束类别存在性；实际生成质量或合理性的改善由实验支持。
- RFS 与 RPS 是互补监督，不写成 RFS 引发了某种问题、必须由 RPS 修复。
- 公式服务于已解释清楚的预测过程；不增加未采用的模块，不以改名暗示新的算法机制。

## 2. Direct Speaker-Audio Conditioning

### 当前不足

开头的互补关系方向正确，但 “complementary information” 比较概括。后面的时间处理、归一化、gate 与初始化细节较集中，各自服务于什么目的还可以连接得更清楚。

### 拟采用的论证顺序

> 描述引导的条件表达预期回应 → 当前话语还提供随时间变化的声学线索 → 引入直接声学路径 → Whisper 提取并映射为 speaker context \(S\) → 两路 cross-attention 注入 → 语义条件与直接声学条件共同参与生成。

可用重音、停顿和语调变化说明声学线索的含义，但不将 Whisper 分支写成已显式检测或分别编码这些属性。也不声称 Qwen hidden 不含声学信息。

### 具体修改

- 开头明确生成过程为什么需要访问当前话语的连续声学表征，再介绍 Whisper。
- 保留 cross-attention 公式、注入位置，以及 generation states 提供 query、\(S\) 提供 key/value 的来源。
- 将保留静音与停顿、共同时间原点放在一起解释，说明它们共同保持原始音频时间结构。
- 保留零初始化及其直接作用：新增分支初始不产生 residual update。
- 每头 gate 和详细 Q/K 归一化拟压缩或后移到实现细节；正文保留到何种程度待讨论确定。

## 3. Response Feature Supervision

### 当前不足

从“Connector 需要完成映射”直接跳到“辅助目标提供直接监督”，尚未充分说明希望生成条件具备什么性质。AV 目标本来就通过 LTX 优化 Connector，因此不能把 RFS 的动机写成 Connector 原先没有监督。

### 拟采用的论证顺序

> 理解表征需要转化为生成条件 → 希望条件保留能够预测目标回应的信息 → 用目标视听特征预测约束条件 → 提取目标特征并建立位置对应 → 同一份 \(U_m\) 条件化辅助解码器 → 特征空间流匹配 → 梯度更新 Connector。

RFS 的核心动机是：让送入 LTX 的条件携带与目标倾听者视听回应有关的信息，并通过辅助预测任务直接约束这种信息保留能力。

### 具体修改

- 先解释辅助任务的用途，再介绍辅助解码器及 Bernini 的 AdaLN 结构。
- 保留目标特征来自冻结 Qwen 视觉／音频编码器，以及视觉网格、音频时间分箱建立位置对应的事实。
- 合并重复的两处 “At each valid position”，将加噪、条件输入和速度预测组织成一次连续说明，再接损失公式。
- 保留特征空间 flow velocity 的预测对象；不改写为 \(U_m\) 与目标特征的直接逐元素回归。
- 结尾突出辅助解码器与 LTX 使用同一份 \(U_m\)，说明辅助任务为何能够约束实际生成条件。
- 保留有效位置归一化、安静音频作为有效目标，以及辅助分支仅用于训练的必要说明。

## 4. Reaction Presence Supervision

### 当前不足

动机已有基础，但还可以更明确地承接 AV 与 RFS 的连续监督。当前直接出现 “The pooled LTX representations provide...” ，没有充分交代表征来源、池化和类别预测头。

### 拟采用的论证顺序

> AV 与 RFS 提供连续媒体和特征层面的学习信号 → 短暂、细微的反馈类别需要成为显式学习目标 → 合理反应的时刻、持续长度与重复次数可以变化 → 采用整段类别存在性监督 → 生成表征、池化和类别预测 → 标签与 BCE → 与 AV 目标的互补作用。

“To make brief and subtle feedback a direct learning target” 的动机保留，并与整段监督的设计理由连接。

### 具体修改

- 在标签和损失之前，补齐简短预测路径：LTX 最后一个 block 的视听 hidden → 分模态聚合有效生成位置 → 独立 reaction presence heads → 整段 logits。
- 保留正例、低权重弱负例和未知项屏蔽规则，以及 extraction-validity／evidence-consistency 的现有口径。
- 将类别范围说明拆清楚：先按训练集支持量启用类别，再在当前全局训练批次中保留有效标签数 \(N_c>0\) 的类别。
- 保留按有效标签数而非权重和归一化的定义，以及维持弱负例较小贡献的解释。
- 结尾保留 RPS 与 AV 的分工：前者显式监督类别存在性，后者继续监督配对音视频；不重复展开开头的时序可变性动机。
- RPS 的直接监督对象是生成表征上的类别预测。鼓励最终音视频表现出相应反应是设计目的，不把类别预测正确等同于所有生成行为均正确。

## 5. Training and Inference

### 当前情况与拟采用的顺序

现有顺序基本完整，不做大幅重写。训练部分保持：

> 规划器为什么需要适配 → 文本规划训练 → 生成器使用的描述和 hidden → AV 目标 → 总目标及参数更新范围。

推理部分保持：

> 描述预测 → 条件形成 → 联合采样 → 视频与音频解码。

### 具体修改

- 规划器 LoRA 的目的定位为适配标注中的倾听反馈行为。偏向安静倾听的初步观察可以作为动机，但不把目标写成无条件增加动作或声音。
- 保留规划器输入与最终任务一致：speaker audio 和 listener first frame。
- 保留两阶段训练及其参数归属，不能写成 Qwen 规划器由生成器的总损失联合更新。
- 保留生成训练使用原始 Qwen hidden、推理使用 LoRA-adapted Qwen 的实际区别；缓存操作本身可以压缩。不补写没有实际依据的版本选择理由。
- 保留 LTX 视频／音频 VAE、AV 流匹配、固定首帧和有效位置 mask 的核心定义。
- 总目标统一使用 RFS／RPS 名称；推理不重复各模块动机。

## 6. 名称与符号统一

已确定的小节名称为 **Response Feature Supervision** 和 **Reaction Presence Supervision**。RPS 标题不再加 “Clip-Level”，正文首次定义时明确其监督范围是整段。

| 当前残留写法 | 拟统一写法 |
|---|---|
| `Feature-loss gradients` | `RFS gradients` |
| `clip-level event supervision` | `reaction presence supervision (RPS)`，正文说明整段范围 |
| `feature and event supervision` | `RFS and RPS` |
| `event heads` | `reaction presence heads` |
| \(\mathcal L_{\mathrm{Feature}}^m\) | \(\mathcal L_{\mathrm{RFS}}^m\) |
| \(\mathcal L_{\mathrm{Event}}\) | \(\mathcal L_{\mathrm{RPS}}\) |
| \(\lambda_{\mathrm F},\lambda_{\mathrm E}\) | \(\lambda_{\mathrm{RFS}},\lambda_{\mathrm{RPS}}\) |

内部交叉引用标识如 `\label{sec:feature}`、`\label{sec:event}`、`\label{eq:feature}`、`\label{eq:event}` 保留，避免无必要的引用改动。损失计算及权重数值不因更名而变化。

## 7. 讨论与执行范围

下一步先讨论四个小节的论证顺序，特别是 RFS 的动机，以及 speaker-audio 细节在正文与实现细节之间的分工。通过后再按作者当前原文逐段修改。

执行时检查：每个操作是否回应前文提出的要求；每个公式的预测量和目标是否提前定义；RFS／RPS 是否与实际监督位置一致；动机是否被不必要地删去；新增内容是否超出已确认实现。交付时列出具体修改，保持其余文字及排版。
