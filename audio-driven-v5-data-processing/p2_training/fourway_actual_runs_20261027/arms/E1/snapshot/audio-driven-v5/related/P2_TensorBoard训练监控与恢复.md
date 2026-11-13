# P2 TensorBoard 训练监控与恢复

2026-10-26。已为当前 AV / AV＋Event＋text-attention LoRA 训练接入 TensorBoard，保留逐 rank JSONL。每 100 个 optimizer step 保存和严格检查点恢复逻辑仍在；原生 LTX 冻结、无 feature decoder/teacher 的训练目标不变。

## 1. 配置和查看方式

[普通 LoRA 配置](../configs/p2_listen_av_event_text_lora_v1.json) 和 [H200 配置](../configs/p2_listen_av_event_text_lora_h200_v1.json) 已默认开启：

```json
"tensorboard": {
  "enabled": true,
  "log_every": 1,
  "flush_secs": 30
}
```

配置位于 `training` 下。`log_every` 是 optimizer step 间隔；首次更新、周期检查点、预算末步及显式 stop-after 步也会记录。旧配置未填写时默认关闭。只由 rank 0 写事件文件：

```text
<output>/runs/<seed>/<E0或E1>/tensorboard/events.out.tfevents.*
<output>/runs/<seed>/<E0或E1>/metrics.jsonl
<output>/runs/<seed>/<E0或E1>/train_rank<rank>.jsonl
```

S0 的 seed 目录带 `s0_` 前缀。`metrics.jsonl` 是与 TensorBoard 对应的全局数值；`train_rank*.jsonl` 保留各 rank 原有训练记录，AV loss 仍为局部均值，不能把 rank 0 的局部 AV 当作全局曲线。

可直接查看本次已验证的真实 GPU E0/E1 曲线：

```bash
/private/liqi_workspace/project_code/Omni-LTX-stage2/.venv/bin/tensorboard \
  --logdir /private/liqi_workspace/project_code/audio-driven-v5/experiments/p2_tensorboard_20261026/single/runs \
  --host 127.0.0.1 --port 6006
```

浏览器访问 `http://127.0.0.1:6006`；若浏览器位于另一台机器，转发该端口。正式训练时将 `--logdir` 改为正式配置 output 下的 `runs`。一个 logdir 可以同时比较不同 seed、E0/E1 的曲线。本次没有留下常驻 TensorBoard 服务。

训练环境已经安装 TensorBoard 2.21.0。重建环境时使用 [依赖文件](../requirements-p2-monitoring.txt)：

```bash
uv pip install \
  --python /private/liqi_workspace/project_code/Omni-LTX-stage2/.venv/bin/python \
  -r requirements-p2-monitoring.txt
```

启用但缺少依赖时，所有 rank 都在加载大模型之前报错，不会静默丢失日志。

## 2. 指标口径

| 标签 | 含义 |
|---|---|
| `loss/av` | 所有 rank、全局 batch 的 AV loss 均值 |
| `loss/av_video`、`loss/av_audio` | 采用实际训练 mask 的视频/音频 flow-matching loss，各自先按样本有效位置归一，再按样本平均；两项之和为 AV |
| `loss/base` | 基础损失，当前无 feature 版本等于 AV |
| `loss/event`、`loss/event_weighted` | 全局 Event loss，以及乘事件权重后的贡献 |
| `loss/optimization_objective` | base＋加权 Event，包括 E0 的独立 probe 训练项 |
| `loss/generator_objective` | E0 为 base；E1 为 base＋加权 Event。E0 probe 不反馈到生成参数 |
| `event_loss/<类别>` | 按该类全局有效标签数归一的 BCE，包含训练所用正例权重 |
| `event_contribution/video`、`audio` | 两模态对总 Event loss 的贡献；分母是全局活跃类别数，两项相加等于 Event，**不是各模态单独类别均值** |
| `supervision/known_bins/<类别>`、`active_classes` | 每类全局有效监督数和活跃类别数 |
| `learning_rate/connector`、`events`、`text_lora` | 当前更新实际使用的分组学习率，包含 warmup |
| `grad_norm/main_preclip`、`events_preclip` | Connector＋LoRA、事件头两组裁剪前范数 |
| `time/*_max_rank` | 步、数据准备、计算、周期检查点耗时，取最慢 rank |
| `throughput/samples_per_second` | 全局样本数 / 最慢 rank 步耗时 |
| `throughput/media_seconds_per_second` | 全局样本实际请求时长之和 / 最慢 rank 步耗时 |
| `memory/*_gib_max_rank` | 当前/本次记录步峰值的 allocated、reserved 显存，取最大 rank，单位 GiB |
| `progress/*` | global batch、累计样本位置数、按数据集大小折算的 epoch 进度 |

事件 loss 在训练中已按全局计数聚合，监控不会再对它重复 SUM。AV 用各 rank 样本数加权汇总，不把局部损失误作全局值。某类别本步完全无监督时，记录 `known_bins=0`，但不伪造 `event_loss=0`。当前样本无摇头正例，所以实际验证为 42 条 scalar 标签；完整七类数据可能更多。

步耗时从取样之前计时，包含缓存读取/H2D、计算、原有 JSONL 写入及本步周期检查点；不含初始化、监控自身聚合/写入和训练退出时的额外末步补存。因此保存步的吞吐曲线会自然下降。CUDA 计时在记录步同步，显存峰值也在该步重新开始统计。

日志只使用 detached 数值；没有增加训练 loss、梯度路径或随机采样。没有把训练 loss 包装成生成质量/F1；实际回应能力仍由独立生成评测判断。

## 3. 保存、恢复与曲线衔接

- 正式配置 `save_every=100` 保留。周期检查点保存后 flush 日志，正常结束及异常清理时关闭 writer。
- 显式 `--resume step_XXXXXX.pt` 恢复到第 N 步后，下一步从 N＋1 开始。
- TensorBoard 使用 `purge_step=N+1` 隐藏旧事件文件中超出检查点的记录。原事件文件保留，不覆盖；读取时不会显示重复的后续步。
- 逐 rank JSONL 和全局 `metrics.jsonl` 在恢复时同步裁去超出检查点的尾部；支持最后一行因中断而不完整。中间损坏或已提交步骤重复会明确报错，不静默吞掉。
- writer 创建/写入失败会同步到其他 rank，避免一个 rank 失败、其他 rank 继续等待下一次训练通信。

仍要求配置、数据、源码、权重和 rank 数等满足严格恢复身份；没有加入自动重启或自动挑选最新检查点。本次源码变更前的 GPU 短测检查点属于旧版本，不能绕过身份校验直接恢复到新代码；可使用其对应源码备份复现。尚未开始正式效果训练。

## 4. 验证记录

证据目录：[p2_tensorboard_20261026](../experiments/p2_tensorboard_20261026/)。

- **146 项 CPU 测试通过**，包含实际 TensorBoard event 文件读取、无监督类别、两种 Event 目标口径、配置校验、损坏日志保护、恢复清理和不消耗 Torch RNG。
- 两 rank CPU/Gloo 使用不同样本数量和不同 AV 值，验证按样本加权的全局 AV、Event 不重复归约、单 writer 以及 writer 异常传播；原训练 CPU DDP E0/E1 参数差仍为 0。
- 真实 H200：单卡 E0、单卡 E1、双卡 E1 完成各 5 步；单/双卡 E1 在第 3 步停止再恢复至第 5 步。
- 恢复测试主动加入旧第 4 步曲线（AV=999）和 JSONL 尾部。恢复后实际事件文件只显示步骤 1–5，旧值被清除，无重复步骤；逐 rank JSONL 也一致。
- 额外运行 TensorBoard 关闭的单卡 E1。与开启＋恢复的运行相比，模型、优化器、各 rank RNG **逐位一致**；开启的单/双卡模型与优化器也逐位一致。
- 核对实际事件文件中的 42 条指标，包括全局 AV、视频/音频分项、Event、学习率 warmup、梯度、吞吐和显存。

[GPU 验收](../experiments/p2_tensorboard_20261026/gpu_verification.json)、[CPU 多卡指标验收](../experiments/p2_tensorboard_20261026/cpu_metrics_ddp/summary.json)、[CPU 测试日志](../experiments/p2_tensorboard_20261026/cpu_tests.log)。所有测试 GPU 进程已结束；本次仅做有限 S0，没有启动正式效果训练。
