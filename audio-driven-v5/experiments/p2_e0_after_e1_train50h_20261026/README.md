# E0：在 E1 完成后自动接续的严格配对实验

**已按用户要求停止此旧任务：E0 当时尚未训练更新。旧控制器 STOP 保留。预检查修复、验证及后续 E0 请见 [新版入口](../p2_preflight_fix_20261026/README.md)。以下为历史启动记录，不要重新启动本目录控制器。**

2026-10-26，用户明确要求：当前 E1 结束后，用相同数据、相同 seed 开展“关闭两个 feature decoder，E0 的 Connector 仅由 AV loss 更新”的实验。

**接续已实际触发（09:40 UTC）：E1 全部 1,268 步完成，三节点正常退出，最终检查点 SHA256 通过；E0 三机 24 个 worker 已启动，当前全量数据/缓存预检查中。** [实际切换验证](transition_validation.json)。

**接续控制器已部署。** 实际阶段见 [queue_status.json](queue_status.json)。`waiting_e1` 表示尚未启动 E0；`waiting_resources` 表示 E1 已成功完成，正在等全部 GPU 空闲；`training_precheck` 表示三机 E0 进程已启动、原生全量数据/缓存校验中；`training` 表示已加载模型并通过配对身份检查；最终 `complete` 才表示 E0 全预算训练完成。控制器独立于本次对话和 SSH 会话运行。

## 配对设置

| 项目 | E0 设置 |
| --- | --- |
| 数据 | 与 E1 完全相同的 30,415 段、50 小时零 4 秒 |
| 训练 seed | 20261023 |
| 机器 | 172.29.252.11、172.29.15.91、172.29.225.51，各 GPU 0–7 |
| global batch / steps | 24 / 1,268，共 30,432 个样本位置 |
| 初始化 | 从与 E1 相同的预训练权重和随机初始化重新训练；不加载 E1 已训练检查点 |
| Connector | 仅由 AV loss 更新 |
| text-attention LoRA | 与 E1 相同的 rank=8 / alpha=8，仅接收 AV 梯度 |
| Event head | 保留 `probe_only`：输入 U detach；Event loss 仅更新事件头，不回传 Connector / LoRA |
| Feature decoder / teacher | 两个 decoder 均不构建；不读 teacher、不计算 feature loss、不建对应参数组 |
| 保存与监控 | 每 100 步保存、最终 step_001268.pt、逐步 TensorBoard；保留完整断点恢复 |

E0/E1 **使用同一份未修改的 [config.json](../p2_e1_train50h_lora_tb_20261026/config.json)** 和相同数据/标签/缓存。原配置目录名中保留 `e1`，是为了复用训练器已有的严格 `pair.json` 校验，而非表示 E0 接续 E1 权重。只有训练 CLI 的 `--arm` 从 E1 变为 E0；新通信端口为 29747，不改变训练数值配置。

原生训练器会重新核对配置、seed、世界大小、数据顺序与标签、类别统计、初始参数哈希、源码和预训练资产身份。除 arm 外任何配对身份不一致都会停止；控制器还会再次比较两个 `run.json` 并发布 `pair_validation.json`。

E0 的事件 loss 仍可出现在 TensorBoard 中，它属于独立探针。衡量生成分支优化目标时看 `loss/generator_objective`（E0 为 AV），不要把探针训练项误解为 Event loss 回传了 Connector。

## 自动接续条件

[controller.py](controller.py) 每 20 秒检查一次，依次要求：

1. E1 三个节点全部 `finished` 且 returncode=0。
2. E1 最终 status 为 step=budget=1,268、stopped=false。
3. E1 最终检查点及 complete marker 完整，文件 SHA256 匹配。
4. 冻结的 82 个启动/训练相关文件哈希不变。
5. 三台指定机器各 8 张卡显存占用均 <1 GiB。

满足后在三机并行启动 [run_node.py](run_node.py)。每个节点和协调器都有独占文件锁；节点持久 supervisor 记录 torchrun PID、进程出生时间、命令和退出码。断开 SSH 不终止作业。

E1 异常或未完成预算时不会自动开始 E0。E0 某节点失败时只终止本控制器拥有、身份验证通过的其他 E0 进程，不干预 E1 或其他用户进程；保留错误状态，不无限重试。

原生每 rank 全量缓存校验仍保留。E1 本轮该阶段约 28 分钟，所以 E0 接续进程启动后 GPU 可能暂时为空，不能据此判断作业没有启动。以节点日志和 queue 状态为准。

## 路径

- [授权及冻结文件清单](release.json)、[三机预检查](preflight.json)、[CPU 测试记录](tests.log)。
- [控制器状态](queue_status.json)、[控制器日志](queue.log)。
- 本目录 `node0/1/2.json`、对应 `.log`、`worker_logs/`：E0 启动状态和进程日志。
- E0 训练输出：`../p2_e1_train50h_lora_tb_20261026/runs/20261023/E0/`。
- E0 TensorBoard：上述 E0 目录下的 `tensorboard/`；与 E1 独立存放。
- 启动后自动生成：`predecessor_verified.json`、`launch_results.json`、`pair_validation.json`。

已部署的启动命令（勿重复执行）：

```bash
/usr/bin/python3 /private/liqi_workspace/project_code/audio-driven-v5/experiments/p2_e0_after_e1_train50h_20261026/controller.py --detach
```

如需取消尚未开始的 E0，在本控制目录创建 `STOP`。若 E0 已经运行，同一个 STOP 会由控制器转成 E0 原生 step-boundary STOP，使其保存并正常退出。不要在 E1 目录创建 STOP。

如需恢复已暂停的 E0，先确认所有原 E0 进程已退出，移走控制目录和 E0 训练目录的 STOP，并在三机分别调用 `run_node.py --node-rank N --resume /绝对路径/E0/step_XXXXXX.pt --detach`。原生恢复严格要求该 E0 run 的检查点；不能拿 E1 检查点恢复 E0。现有控制器不自动发起恢复或回滚。

## 验证

44 项 CPU 测试、18 个子测试通过，涵盖接续条件、失败阻止启动、完整检查点 SHA、同 config/seed 且不 resume、初始化匹配，以及原生无 feature、E0 事件梯度隔离、LoRA 和性能配置测试。测试中的 CPU bf16 RMSNorm 非融合路径有一条 PyTorch 提示，不影响测试通过。

已实际核对三台机器连接、主机名、8 卡状态与冻结源码。未改 P2/v4 训练源码、E1 配置、原始数据、标签、缓存或既有 E1 启动器。
