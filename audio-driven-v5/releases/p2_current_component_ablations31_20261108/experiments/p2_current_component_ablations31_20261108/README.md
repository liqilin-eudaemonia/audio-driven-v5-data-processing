# 最新500h：完整模型分别去掉一个组件

本轮只准备代码、独立配置和验收，不启动GPU训练或加入自动接续队列。用户已确认采用完整模型分别移除Whisper、Feature、Event的三组设计，复用既有组件并替换到最终500h和对应最新标注。各组从原始预训练权重及seed20261023独立初始化；不继承完整模型或其他消融的已训练权重。

| 目录／arm | 实际条件与目标 | 不执行的分支 |
|---|---|---|
| [no_whisper](no_whisper/config.json) | 原始Qwen条件；AV＋0.025视频Feature＋0.025音频Feature＋0.05 LTX整段Event | 不创建Whisper adapter／cross-attention，不打开speaker Whisper缓存 |
| [no_feature](no_feature/config.json) | 原始Qwen＋speaker Whisper；AV＋0.05 LTX整段Event | 不创建EmbeddingDecoder，不打开teacher缓存 |
| [no_event](no_event/config.json) | 原始Qwen＋speaker Whisper；AV＋0.025视频Feature＋0.025音频Feature | 无Event头／损失／优化器组，不打开Event标签或数值缓存 |

原有Qwen条件已经包含speaker音频信息；“去掉Whisper”移除的是额外语音直连分支。Feature沿用原EmbeddingDecoder的flow-matching重建，通过条件U更新Connector及Decoder；冻结Qwen不更新。Event沿用当前LTX最后block、输出投影前的实际音视频hidden，对有效生成位置整段聚合；19类配置及有效类别mask、未知／冲突屏蔽和0.1弱负例权重均复用当前完整模型，无时间区间监督。

当前Event配置19类中实际启用18类；`visual:mouth closing`有3032个正例，按原完整模型固定的5000门槛屏蔽。没有为本轮改变类别或降低门槛。

## 数据与复用

- 最终284,218段／500.001259小时：train281,060＋dev3,158，test为0。历史`training_admitted`不参与重新筛选。
- 清单及成员顺序：`experiments/listen_j_upright_av500h_train_dev_v3_20261105/TRAINING_CACHE_INDEX.json`和`experiments/p2_ltx_event_current_sources_v2_20261106/sample_list.csv`。
- 复用20261105版原始未SFT Qwen hidden、Gemma完整caption、AV VAE，以及各组启用的speaker Whisper／listener teacher混合缓存；不新增缓存或重新打标。
- 含Event两组复用`p2_clip_guarded_current500h_20261106`与`p2_clip_numeric_guarded_current500h_20261106`；逐样本source digest绑定当前描述及补充证据。训练不解析caption或LLM原文。
- 三组使用与当前AV-only／AV＋Feature／AV＋Whisper完全相同的31卡sample/noise计划：9169步覆盖一轮，尾部21条padding；比较6000步时传`--stop-after 6000`。历史完整模型曾变更batch，正式对比仍需报告样本展示预算。

## 代码与配置

统一入口：[scripts/p2_current500h_component_ablations.py](../../scripts/p2_current500h_component_ablations.py)。独立模块：[src/p2_current_component_ablation](../../src/p2_current_component_ablation)。

通过独立包装复用MatrixModel／QwenWhisperAV、EmbeddingDecoder、LTX hidden readout、整段Event头和原规范化目标。没有修改已冻结的训练包。每组checkpoint包含本组schema／配置／数据／源码／世界大小身份、完整模型／Adam及各rank RNG，严格拒绝跨组或跨输出目录恢复。[inference.py](../../src/p2_current_component_ablation/inference.py)提供对应loader和native CFG采样；推理不加载teacher或Event标签。

四机31卡：node12＋node8＋node4各0–7卡，node11仅0–6卡；node11 GPU7留推理。三个独立master端口29861／29862／29863。沿用全重计算、CPU预取2批、512MiB streaming FP64 SUM、GPU Adam、BF16前后向／FP32可训练参数、134GiB进程上限，每500步完整保存。三组共享启动锁，显式launch仍须先通过所选31卡空闲检查，不终止任何已有任务。

```bash
PY=/private/liqi_workspace/project_code/Omni-LTX-stage2/.venv/bin/python
# 示例；另两组改成no_feature或no_event。
"$PY" scripts/p2_current500h_component_ablations.py preflight --arm no_whisper
"$PY" scripts/p2_current500h_component_ablations.py commands --arm no_whisper --stop-after 6000
# 用户分配资源并明确安排训练后再执行：
"$PY" scripts/p2_current500h_component_ablations.py launch --arm no_whisper \
  --stop-after 6000 --logs experiments/p2_current_component_ablations31_20261108/no_whisper/launch
```

`STOP`写在对应arm目录：当前完整更新结束后保存退出。恢复用同组完整checkpoint的`--resume`，不得将其他组权重作为resume。`--stop-after`经31卡node入口环境变量传到每个rank；6000步退出保存由训练循环执行。

## 验收范围

见[PREPARATION_ACCEPTED.json](PREPARATION_ACCEPTED.json)。测试覆盖实际native tiny LTX更新、所有活动参数梯度、Feature／Event独立梯度路径、共享初始化、严格下一步恢复及跨组拒绝、两进程真实Gloo＋FP64 SUM＋Adam与参考逐值一致、禁用缓存读取、31rank映射与停止参数传递。每组32条真实缓存回读覆盖train／dev及多种缓存来源，完整caption逐字核对；全量选集／缓存身份绑定到已有完整审计。

尚未进行这三组完整规模的GPU显存、NCCL与速度实测。正式启动时，每组先在实际rank上执行最长真实样本的两次完整反向及Adam更新，并恢复原始模型／优化器／RNG，再进入正式训练；不能把CPU验收当成已开训或GPU验收。
