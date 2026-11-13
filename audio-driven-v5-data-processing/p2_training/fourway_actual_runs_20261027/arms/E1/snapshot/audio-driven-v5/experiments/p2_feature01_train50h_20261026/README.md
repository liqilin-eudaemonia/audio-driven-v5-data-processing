# AV + 0.1 Video Feature + 0.1 Audio Feature，50h 配对实验

用户确定：本轮开启两个 Feature decoder，生成分支不加 Event loss；两路 Feature loss 权重各 0.1。与已完成 E0 使用相同数据、seed、预算和共享初始化，从预训练初始化重新训练，不加载 E0 的训练后权重。

## 训练口径

- 数据：30,415 段，约 50 小时；沿用 E0/E1 的原始目标音频与标签、缓存。未切换到其他并行任务的音频静音版本。
- 资源：3 台 × 8 张 H200；global batch=24，每卡每步 1 段；1,268 步，共 30,432 个样本位置，至少一轮。
- seed=20261023；Connector / Feature decoder 学习率均 3e-5，text-attention LoRA rank=8、alpha=8、学习率=1e-5；warmup=50，沿用原调度。
- 生成目标：`L_AV + 0.1*L_feature_video + 0.1*L_feature_audio`。
- 沿用 E0 的独立 Event probe：输入 detach，只有探针自己的参数被 0.05 Event loss 更新，对 Connector/LoRA 的 Event 梯度为零。目录中的 `E0` 表示代码的 `probe_only` 模式，本实验由新目录与 Feature 配置区分，旧 E0 结果没有覆盖。
- 每 100 步及最终/主动停止时保存完整 checkpoint，包括两个 decoder、Connector、LoRA、probe、优化器与各 rank RNG；支持严格断点恢复。
- 训练读取 input、teacher、VAE、text、events。原有 teacher 缓存直接复用，不重算。

## 版本与损失实现

入口：[版本化 CLI](../../releases/p2_feature_v1/scripts/p2_listen.py)，配置：[production/config.json](production/config.json)。

实际反向权重使用 `training.feature_loss_weights={"video":0.1,"audio":0.1}`。不修改 `extractor` 中的历史缓存字段，避免无意义地使既有 input/teacher 缓存失效。两路系数不仅影响日志，确实乘入 backward；裁剪前梯度缩放有独立 CPU 验证。

先构建 Connector、Event probe、LoRA，再在保留 CPU RNG 的上下文中构建两个 decoder，保证共享参数与 E0 初始哈希完全一致。decoder 为原有条件流匹配网络，其输出层与条件调制零初始化，最初几步 Feature 对 Connector 梯度为零是预期行为；真实 GPU 测试需确认之后两路梯度均出现。

版本目录隔离了本轮修改及旧版 CLI/training_data；其余复用依赖通过源码 SHA 固定在 `release.json` 中，启动和恢复都会校验。不得在正式运行期间修改这些已记录文件。独立音频后处理工作的新版 root CLI 不作为本轮入口。

## 验证与状态

2026-10-26 验收完成：158 项 CPU 测试及 18 项子测试通过；真实预训练模型单卡短/最长片段各 5 步通过，最长片段峰值分配显存约 124.5 GiB。三机 24 卡连续 5 步与第 3 步停止后恢复到第 5 步，1,121 个模型张量、3,360 个优化器张量和 48 个 RNG 张量逐位一致，所有 rank 的样本顺序及损失也精确一致。共享初始化哈希与 E0 完全一致，两路 Feature 梯度在初始化后的更新中已验证非零，Event 对生成分支为零。

30,415 段含 teacher 的全量审计于约 13 分 19 秒完成，三节点正常退出，全局仅执行一次；数据、标签、类别统计精确匹配 E0。正式后台启动程序已于 **2026-10-26 13:40 UTC** 发起，于 **13:44 UTC** 确认 24 卡均已实际更新至少 14 步，52 项 TensorBoard 标量写入正常，正式配对核验通过；[启动验收](training_start_validation.json)。后续实际进度见下方实时状态。源码和启动输入已另存 `frozen_sources/`，哈希见 `release.json`；此为本地固定快照。

- [CPU 回归](regression.log)、[补充监控测试](monitoring_test.log)
- [真实 GPU 测试状态](gpu_suite_status.json)：单卡短片段、单卡最长片段、三机 24 卡连续运行与中断恢复。
- [全量预检日志](audit_cluster.log)：30,415 份 teacher 与其余缓存的完整审计；通过后启动使用已验证 receipt。每次实际取样仍验证缓存。
- [GPU 精确恢复对比](gpu_comparison.json)、[总验收](verification.json)
- [正式实时状态](status.json)、[与 E0 配对核验](pair_validation.json)

正式产物位于 `production/runs/20261023/E0/`，实验含义是 **AV+Feature，无生成 Event 监督**。`single/longest/continuous/resume` 均为隔离测试产物，不能当正式结果。

## TensorBoard 与恢复

```bash
cd /private/liqi_workspace/project_code/audio-driven-v5
../Omni-LTX-stage2/.venv/bin/tensorboard --logdir experiments/p2_feature01_train50h_20261026/production/runs/20261023/E0/tensorboard --port 6006
```

记录 AV 总/分模态 loss、Feature 原始 loss、加权 Feature loss、加权合计及 Feature/AV 比值、生成分支目标、独立 probe 指标、学习率、梯度、显存、速度及进度。`loss/optimization_objective` 包含独立 probe，比较生成目标请看 `loss/generator_objective`。

三机管理器与各节点 supervisor 均脱离终端持续运行。首次启动由验收完成后执行：

```bash
../Omni-LTX-stage2/.venv/bin/python experiments/p2_feature01_train50h_20261026/controller.py --detach
```

停止后从已完成标记对应的 checkpoint 恢复（替换实际步数；先确认本轮三个节点已退出；若存在本轮 STOP 文件需移除）：

```bash
../Omni-LTX-stage2/.venv/bin/python experiments/p2_feature01_train50h_20261026/controller.py --detach \
  --resume /private/liqi_workspace/project_code/audio-driven-v5/experiments/p2_feature01_train50h_20261026/production/runs/20261023/E0/step_000100.pt
```

恢复不能改变数据、source、seed、world、batch、预算或配置。独立推理也使用本版 CLI，并传入本轮配置和 checkpoint；不需要 teacher 或标签来生成。
