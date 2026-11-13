**去Event组已四机31卡正式开训（2026-11-08）：** AV＋speaker Whisper＋双路各0.025 Feature，Event关闭，使用原始未SFT Qwen H、最终284,218段／500.001259h和最新完整caption。从预训练独立初始化；沿用时长分组496、CPU预取2、512MiB streaming FP64 SUM、GPU Adam及全重计算，每500步保存，6000步完整保存后停止。node12／node8／node4用GPU0–7，node11按用户确认改为GPU1–7，GPU0现有vLLM保持。31rank最长样本两次完整更新及恢复通过，显存峰值96.36GiB；首批正式更新loss公式／四组有效梯度／Event为0检查通过。其他自动训练队列保持取消。[启动验收](no_event/LAUNCH_ACCEPTED.json) · [训练步数](no_event/status.json) · [卡号包装入口](../../scripts/p2_current_noevent_node11shift_20261108.py)。

**当前启动：去Event＝AV＋Whisper＋Feature（2026-11-08）：** 用户单独授权运行本组；保持31卡、原始未SFT Qwen H、最终500h／最新caption、AV＋双路各0.025 Feature、每500步保存，6000步完整保存后停止。node12／node8／node4用GPU0–7；node11改用GPU1–7，GPU0留给现有vLLM。原训练源码／配置／sample-noise计划不变，仅独立卡号包装入口变化；其余自动队列保持取消。入口`scripts/p2_current_noevent_node11shift_20261108.py`，日志`no_event/controller_gpu1_7_20261108.log`，启动备份`no_event/launch_gpu1_7_snapshot/`。本组GPU最长样本与正式更新验收正在执行，以`no_event/LONGEST_SAMPLE_CHECK.json`、`no_event/status.json`及启动验收记录为准。

**最新指令：当前AV＋Whisper到6000完整保存后停止，先统一测评。** 本目录QUEUE_STOP已取消所有后续自动接续，sequence_v2控制器已退出。独立[stop_current6000.py](stop_current6000.py)只负责当前6000保存／校验／退出，不启动任何新组。[挂载验收](STOP6000_ARMED_ACCEPTED.json) · [实时状态](STOP6000_STATE.json)。下方全部接续顺序均为历史记录，恢复排队须用户另行安排。

**下一组已优先插入AV＋Event（2026-11-08）：** 新队列为当前AV＋Whisper6000→AV＋Event6000→去Whisper6000→去Feature6000→去Event6000；均完整保存、校验、退出后接下一组。AV＋Event关闭Whisper和Feature，保留0.05 LTX整段Event，最新500h及最新标签，预训练独立初始化。[AV＋Event配置与验收](av_event/README.md) · [新队列入口](sequence_v2.py) · [授权](SEQUENCE_V2_AUTHORIZED.json) · [实时状态](SEQUENCE_STATE.json)。旧sequence.py控制器已被新控制器替换，后文原三组定义不变。

# 最新500h：完整模型分别去掉一个组件

用户已授权自动接续：当前AV＋Whisper在step6000完整保存、校验并退出后，按去Whisper→去Feature→去Event顺序运行；每组到6000完整保存、校验并退出，再启动下一组。队列入口[sequence.py](sequence.py)，[授权与固定顺序](SEQUENCE_AUTHORIZED.json)、[实时队列状态](SEQUENCE_STATE.json)、[控制器](SEQUENCE_CONTROLLER.json)。各组从原始预训练权重及seed20261023独立初始化；不继承完整模型或其他消融的已训练权重。

队列保持四机31卡、每500步完整保存、node11 GPU7预留。当前AV＋Whisper通过外部控制器在6000状态发布后、该步定期完整保存期间写入STOP；后三组使用已测试的原生`--stop-after 6000`。只有准确6000、完整检查点SHA及有限值／Adam步号／31份RNG核验、原31rank进程退出、所选GPU连续两次空闲检查均通过才允许接续。任何失败保留现场并阻止后续组启动，不跳组。新增队列8项测试通过，不改已冻结训练源码。`QUEUE_STOP`写在本目录可取消后续接续；不立即终止正在运行的训练。若意外中断某组，保留检查点，需按本组严格resume接续，不能换用上一组权重。请勿重复手动launch。

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

2026-11-08 再次复核通过：[REVIEW_ACCEPTED.json](REVIEW_ACCEPTED.json)。本轮共126项CPU测试通过；补测三组全重计算的loss与全部梯度逐值一致，以及实际CFG推理条件和参考帧保留。每组重跑32条canary回读，并另抽16条（train12＋dev4，排除canary）走真实后台预取路径；三组使用相同16个随机ID。对284,218条／5,400,142个Event数值目标全量核对0/1、known、weight及source digest，与冻结来源标签全部一致；不是重新人工验收标注语义。见[数据复核](REVIEW_DATA_ACCEPTED.json)和[可复跑审计脚本](review_data.py)。未发现需修改训练代码的问题，既有训练源码与冻结身份未改变；三组仍未开训，GPU实测边界如下。

见[PREPARATION_ACCEPTED.json](PREPARATION_ACCEPTED.json)。测试覆盖实际native tiny LTX更新、所有活动参数梯度、Feature／Event独立梯度路径、共享初始化、严格下一步恢复及跨组拒绝、两进程真实Gloo＋FP64 SUM＋Adam与参考逐值一致、禁用缓存读取、31rank映射与停止参数传递。每组32条真实缓存回读覆盖train／dev及多种缓存来源，完整caption逐字核对；全量选集／缓存身份绑定到已有完整审计。

尚未进行这三组完整规模的GPU显存、NCCL与速度实测。正式启动时，每组先在实际rank上执行最长真实样本的两次完整反向及Adam更新，并恢复原始模型／优化器／RNG，再进入正式训练；不能把CPU验收当成已开训或GPU验收。
