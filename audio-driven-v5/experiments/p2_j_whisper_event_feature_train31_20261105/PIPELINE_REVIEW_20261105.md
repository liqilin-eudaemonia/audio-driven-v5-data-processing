# Event＋Feature＋Whisper：当前31卡运行复核

2026-11-05复核结论：在本次检查范围内，未发现阻断当前训练的代码错误。检查没有修改训练源码、缓存或启动参数，没有暂停训练。代码正确性检查不能替代最终生成效果评测。

## 输入与结构

- 当前使用旧500h的285,638条训练样本与原始、未SFT的冻结Qwen H；没有切换到新重标子集。
- H经Connector生成U，U保留原视听生成条件路径；Gemma guided文本和listener首帧条件保留。
- speaker Whisper-large-v3特征经LayerNorm＋Linear，进入独立的视频／音频cross-attention残差分支，不覆盖原Qwen／文本条件。
- Event对U做有效token掩码池化和整段多标签BCE监督（16视觉＋3音频）。Feature通过两路EmbeddingDecoder做冻结listener teacher特征的flow-matching重建监督。
- Event与Feature梯度流向Connector及各自辅助头；Whisper分支与LTX文本LoRA由AV loss更新。冻结teacher没有作为生成条件输入。

## 时间与来源

speaker音频按同一pair的source_start_s/source_end_s读取，保持静音及整段时间。Whisper token以pair起点为零，每320个16kHz采样对应20ms；尾token截到实际时长。缓存读取严格校验时间格、样本身份、模型及来源收据。

LTX视频查询只使用时间坐标，音频查询使用其音频时间坐标，与Whisper时间使用同一时间RoPE尺度。当前是整段双向cross-attention，不是causal mask；这与已有Whisper对照一致。

本次真实联合读取最长11.95秒样本 `improvised_V01_S0168_I00001223_local_03483d951059`：

- speaker为P1308，listener为P1309，未互换。
- 最终裁框revision为`quality-bounded-crop-v1-20261030`，原图坐标x=1197、y=243、width=height=509。
- listener配方为`j-listener-segment-official-union-v1`。
- Whisper形状`[1,598,1280]`，首区间0–0.02秒、尾区间11.94–11.95秒（浮点容差内）。
- teacher视频`[1,3072,2048]`、音频`[1,96,2048]`，几何校验通过；19类Event与冻结H联合读取成功。

本次没有重新扫描全部媒体；全量收据校验沿用启动前已完成的285,638条预检。

## 损失、分布式与续训

目标为 `AV + 0.05 Event + 0.025 video_Feature + 0.025 audio_Feature`。视频／音频Feature各乘0.025；不是两路相加后再取平均。

AV／Feature按全局样本数归一化；Event按全局各类别已知标签数归一化，再对有效类别平均。梯度通信使用SUM，与上述归一化匹配，没有额外重复除以31。

五组参数是Connector、Event头、Feature decoder、LTX文本LoRA、Whisper适配器及残差分支；参数归属校验禁止重复优化、漏注册或冻结主干误解冻。完整检查点保存模型、Adam和每rank RNG，恢复要求来源身份一致。

31卡全局batch31，8＋8＋8＋7的rank布局；node8物理GPU7排除。预取2批、512MiB通信桶与时长分组已生效。相比历史32卡，batch及分组改变，因此不能称优化轨迹完全相同；历史6000步的样本暴露量约对应本轮6194步，当前仍按500步保存，未额外设置6194步保存。

## 本次验证证据

- 重跑74项CPU测试，全部通过（27.80秒）：联合模型、31卡布局、Whisper时间与掩码、梯度路径、梯度重算、CPU预取、完整检查点恢复、CFG两路条件和clip Event。
- 当前运行记录的240个源码身份与工作区逐一相符，无差异。
- 检查截至第136步的全部136条日志：损失有限、加权总损失公式正确、五组梯度均有限且非零。
- 启动验收记录：31rank最长样本两次完整反向及Adam更新通过，随后恢复初始模型／优化器／RNG；最大rank显存预留96.734375GiB。
- 初始检查点验收记录：新31卡与此前32卡测试的2,313个模型状态张量逐值相同。
- 推理回归未调用Event或teacher分支；CFG正负两路均保留Whisper与U，listener参考首帧保持。

本次尚未做已训练正式权重的真实生成验收，也不能从早期loss判断组合一定有收益。训练继续运行。
