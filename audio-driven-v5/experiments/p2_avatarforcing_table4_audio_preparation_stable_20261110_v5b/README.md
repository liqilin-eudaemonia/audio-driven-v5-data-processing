> 后续状态：四进程GPU前端仍未通过严格一致性检查；本轮全量没有成功完成。现转到[CPU人脸前端＋GPU分离版本](../p2_avatarforcing_table4_audio_preparation_cpu_frontend_20261110_v5d/README.md)，旧文件仅保留诊断。

# IIANet音频准备：固定前端计算开关的正式接续

目标仍为复现AF Table4自身8项及核清其他行来源；未完成目标，不通过拟合种子、指标或测试子集追表格数值。

## 已确认的进展

前轮[官方256 GT完整对照](../p2_avatarforcing_table4_canonical_gt_20261110_v4/RESULTS.md)已完成并独立重算，主口径和完整DIM候选均0/8，Exp FD20.00→42.82，不能把GT切换当作修复。

随后按AF论文5.1补目标说话人音源分离。[原版IIANet代码、官方Vox2及作者issue6视觉权重](../p2_avatarforcing_table4_audio_preparation_20261110_v5/README.md)均取得；277个音频状态、137个视觉状态逐项相等。两条公开演示对应片段的4路功能预检通过，重复音频网络前向误差0。speaker分离波形与原音频的相关为0.999956／0.999599，listener输出低31～45dB。分离模型不是空载或随机初始化。

## 为什么单独开本轮

原版独立进程一致性检查失败：相同CNN speaker的嘴部ROI不相同，原始音频最大差0.000147987，大于事先1e-6门槛，未启动批量。关键点库构造器会重新打开全局cuDNN benchmark，导致原先固定的前端执行策略没有持续生效。失败输出和日志保留在[旧版预检](../p2_avatarforcing_table4_audio_preparation_20261110_v5/parallel_parity_check.log)。

本轮在FaceAlignment构造后立即固定benchmarkFalse／deterministicTrue／TF32False。模型、权重、MTCNN／FAN／裁嘴函数及输入数据不变；不复用旧版嘴部ROI或分离输出。新增两个并发独立进程处理同一条speaker，必须满足ROI数组完全相同、音频最大差≤1e-6，才自动启动完整43对／86路，最多4个独立视频worker。没有放宽门槛。

[执行策略变更](CHANGE_FROM_V5.json) · [分离协议](PROTOCOL.json) · [并行协议](PARALLEL_PROTOCOL.json) · [启动回执](LAUNCH.json) · [实际控制器状态](PARALLEL_RUNNER_STATUS.json) · [严格预检结果](PARALLEL_PARITY.json) · [全量进度](PARALLEL_STATUS.json)。后两类文件按实际阶段产生。

只使用node11物理GPU7，每进程allocator5%，计入上下文并保留16GiB余量；不停止其他任务。每个进程继续执行同一份分离函数，仅隔离输出目录和任务清单，不做网络内部batch改写。全量完成后逐个读取86路FLOAT WAV和原始数组，核对实际长度／有限性／SHA并独立重算音频统计。

## 下一轮AF的音频条件必须明确区分

公开代码对Wav2Vec2输入做归一化。旧版两条功能预检中，即便listener输出很安静，进入预处理后的RMS仍为0.465／0.994；全零输入则严格为0。因此“听起来接近静音”不能替代“模型看到全零”。[实际前端测量](../p2_avatarforcing_table4_audio_preparation_20261110_v5/WAV2VEC_AUDIO_CONDITION_AUDIT.json)。

后续预先固定两套完整输入条件，全部43条：①分离speaker＋自身音轨全零；②分离speaker＋分离listener。其余模型、参考图、seed20、NFE10、CFG2／1、帧率和导出不变。第二套使用目标listener音轨，不是第一套相同的信息条件，不能混报成共同listener协议。AF未公开Table4实际采用哪一套；两套依据是论文的音源准备和公开模型的双路音频接口，不按表格接近程度逐格拼接。[事先固定的后续条件](NEXT_AF_AUDIO_CONDITIONS.json)。

目前没有本轮分离后的AF新43条评分，不能把音频准备或严格一致性预检称为Table4已复现。
