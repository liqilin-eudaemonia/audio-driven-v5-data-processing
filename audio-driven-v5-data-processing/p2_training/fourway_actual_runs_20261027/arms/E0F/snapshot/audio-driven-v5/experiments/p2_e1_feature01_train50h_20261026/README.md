# E1＋Feature：已部署自动接续，等待 E0＋Feature 完成

与 E0＋Feature 做配对比较：同一数据、缓存、初始化、训练预算及 Feature 系数，仅将事件监督从 `probe_only` 切换为 `joint_event`。用户随后授权自动接续：三个独立后台守护已部署，等待 E0＋Feature 完整结束后自动启动 E1＋Feature。当前阶段与重试原因见 [队列实时状态](queue/state.json)。

## 损失与参数更新

生成目标：

```text
L = L_AV + 0.1 × L_feature_video + 0.1 × L_feature_audio + 0.05 × L_event
```

Event 系数 0.05 沿用既有 E1 配置；AV 与两路 Feature 设置与当前 E0＋Feature 完全一致。Event 沿用七类定义、有效标签 mask、全局有效数量归一化及正例权重，不把 unknown 当负例。

| 参数 | 直接收到的损失梯度 |
|---|---|
| Connector | AV＋两路加权 Feature＋加权 Event |
| 视频 Feature decoder | 0.1 × 视频 Feature |
| 音频 Feature decoder | 0.1 × 音频 Feature |
| Event heads | 0.05 × Event |
| text-attention LoRA | AV |
| 预训练 LTX 主干、特征提取模型 | 冻结 |

参数更新仍受原优化器、学习率调度及梯度裁剪影响。各损失共享 Connector，并不表示 Event/Feature 会直接回传到下游 LoRA。

## 配对与目录

- 30,415 段约 50h，seed=20261023，3 机 24 卡，global batch=24，1,268 步。
- 从相同预训练初始化重新开始；**不从 E0 已训练权重继续训练**。
- 两个 decoder、text-attention LoRA、梯度检查点策略及学习率完全复用 E0＋Feature。
- 代码：[p2_feature_v1](../../releases/p2_feature_v1/scripts/p2_listen.py)，已有 `--arm E1` 联合路径，本次未修改其源码。
- 配置：[config.json](config.json)，内容与 E0＋Feature 的 production/config.json 完全一致。arm 由启动命令明确指定，配置中没有另一个会冲突的 arm 开关。
- 新管理目录：本目录，保存 controller/node 日志、状态、资源检查及配对验收。
- 训练结果：`../p2_feature01_train50h_20261026/production/runs/20261023/E1/`。
- E0 仍保留在相邻 `E0/`。训练器原生 `pair.json` 要求两组初始化、数据、配置、源码、world 等完全一致；外部配对验证只允许 run.json 的 `arm` 不同。

共用 production 根目录是为了复用同一份缓存与预检 receipt，并启用原生配对校验。两组 checkpoint、TensorBoard、训练日志和 STOP 分别在 E0/E1 子目录；三机管理器另有独立日志与端口 29960。每次启动仍检查文件变化与实际取样缓存，未绕过校验。

## 验证范围

[测试记录](tests.log)：8 项针对性 CPU 测试通过，覆盖：

- AV、Feature、Event 都能回传两路条件；Event 回传 Connector，不直接进入 decoder/LoRA；E0 探针断开。
- 两路 Feature 系数保持 0.1，E1 的生成目标日志包含 0.05 Event。
- 全部事件 unknown 时，E1 与 E0 的参数及优化器更新完全一致。
- E1＋Feature 保存恢复后下一步模型、优化器和损失精确一致；错误 arm/错误目录恢复被拒绝。
- 启动命令为三机 24 卡、E1、同配置；配对只允许 arm 变化；正在运行 E0 的固定文件 SHA 全部保持一致。

[只读启动检查](launch_check.json)、[准备验收](verification.json)、[源码与入口哈希](release.json)。**本次未进行 E1＋Feature 的真实 GPU 验证**；已完成的 E0＋Feature 真实模型/最长片段/24 卡恢复验证仅作为共享实现的既有证据，不冒充 E1＋Feature 实测。

## 自动接续与故障恢复

- 三机各一个 CPU 守护，单一共享锁主控，备用机自动接管；不依赖聊天或终端持续连接。
- E0 必须完成 1,268 步、节点退出并验证最终 checkpoint SHA，才启动 E1。E1 从共同初始化新训，恢复仅使用本阶段自己的 checkpoint。
- SSH/部分节点启动失败会重试；训练意外中断或持续无进度时，只清理白名单中的本任务进程，保留失败产物，从最近 SHA 正确的完整 checkpoint 恢复。无完整 checkpoint 则保留失败目录后重新初始化。
- 队列、当前任务或 run 目录的 STOP 会暂停接续，不自动清除；不会杀其他工作负载。
- 22 项 CPU 测试通过；三机实际共享锁互斥通过。真实主控终止演练中，备用机约 5.3 秒完成接管，E0 继续前进，[验收证据](queue_failover_validation.json)。
- 已固定三机主机公钥供互访；过期启动 attempt 会被拒绝，防止迟到 SSH 重复启动。

[部署记录](queue_deployment.json) · [三机互斥](queue_lock_validation.json) · [测试](queue_tests.log) · [队列代码哈希](queue-release.json)。全量音频后处理的后续接续正在另行适配，不能将它视为已完成。

## 手动检查、监控与恢复

只读检查，不启动进程：

```bash
cd /private/liqi_workspace/project_code/audio-driven-v5
../Omni-LTX-stage2/.venv/bin/python experiments/p2_e1_feature01_train50h_20261026/controller.py --check
```

下列是脱离自动队列时的手动启动入口。当前自动队列已接管，**不要再重复手动启动**：

```bash
../Omni-LTX-stage2/.venv/bin/python experiments/p2_e1_feature01_train50h_20261026/controller.py --detach
```

状态在本目录 `status.json`；每 100 步、最终或主动停止时保存。TensorBoard：

```bash
../Omni-LTX-stage2/.venv/bin/tensorboard \
  --logdir experiments/p2_feature01_train50h_20261026/production/runs/20261023/E1/tensorboard --port 6007
```

看 `loss/generator_objective`、AV 分模态、Feature 原始/加权、Event 总体/分类型、梯度、学习率、吞吐和显存。E1 中生成目标包含 Event，与 E0 的独立 probe 口径不同。

停止后恢复（确保旧 E1 的三个节点已退出，清理本 E1 的主动 STOP 文件；替换实际有效步数）：

```bash
../Omni-LTX-stage2/.venv/bin/python experiments/p2_e1_feature01_train50h_20261026/controller.py --detach \
  --resume /private/liqi_workspace/project_code/audio-driven-v5/experiments/p2_feature01_train50h_20261026/production/runs/20261023/E1/step_000100.pt
```

仅接受本 E1 结果目录的 checkpoint；内部仍严格核对数据、配置、source、arm、seed、world、预算及优化器/RNG 状态。原始 E1（无 Feature）和 E0＋Feature 的 checkpoint 均不能拿来当作本轮恢复点。


## E1 完成后的全量处理

[722,706 段全量音频处理守护已部署](../p2_full_audio_postprocess_crop384_20261026/README.md)。CPU 准备提前执行；本训练队列明确完成 E1、最终 checkpoint 校验及 24 卡空闲后，全量 GPU 缓存自动接手。两套守护独立固定版本，原训练数据与配置不因全量处理改变。全量缓存完成后独立验收，不自动启动全量训练。
