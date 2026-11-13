**后续更新：** node12＋node8＋node4的24卡真实训练已通过，见[测试记录](../p2_ltx_clip_test24_20261106/README.md)。本轮源码已更新，四机正式训练必须通过本文launch入口重新执行preflight；旧preflight源码身份不能继续沿用。node11尚未参加GPU联合实测。

# 当前 LTX 整段 Event 四机 32 卡配置

配置：`configs/p2_ltx_hidden_clip_32gpu_node12_8_4_11_20261106.json`。使用已验收的 `configs/p2_ltx_hidden_clip_events_current500h_20261106.json`，包含修正后的标签及数值缓存，不含 node17。

| node rank | 节点 | 服务名称 | 内网地址 | GPU | global rank |
|---|---|---|---|---|---|
| 0 | node12 | liq-dev-1 | 172.29.252.37 | 0–7 | 0–7 |
| 1 | node8 | zz-dev-1 | 172.29.104.237 | 0–7 | 8–15 |
| 2 | node4 | liqi-dev-3 | 172.29.225.51 | 0–7 | 16–23 |
| 3 | node11 | js-dev | 172.29.189.119 | 0–7 | 24–31 |

node12 作为本轮 torchrun 协调节点，地址 `172.29.252.37:29826`。四机内部 SSH 均使用 root:22；20356、20491、20359、20476 作为外部端口元数据记录，不用于机间 SSH 或 NCCL。

每机 8 张 H200 已确认，eth0、mlx5_0:1 和 GID index 3 可用。沿用现有 RDMA、streaming FP64 梯度归并、512 MB bucket、预取深度 2、时长分组窗口 512 和完整 gradient checkpointing。GPU 分配上限沿用 134 GiB。global batch 32、每卡每步 1 个样本、8,882 步，训练目标仍为 AV + 0.05 整段 Event + 两路 0.025 Feature。

## 查看与启动

只查看已生成的四机命令：`COMMANDS.json`。

准备好开始训练后，在有集群 SSH 凭据的控制节点执行：

```bash
bash /private/liqi_workspace/project_code/audio-driven-v5/experiments/p2_ltx_clip_cluster_node12_8_4_11_20261106/launch.sh
```

该入口调用现有 launcher，先做全量 preflight，再启动四机；每次创建独立日志目录。只有执行此命令才会启动训练。本轮仅写配置、生成命令及做远端只读检查，没有执行 launch，没有运行 NCCL collective，也没有初始化训练模型。

已有该实验的 checkpoint 时须通过 `--resume /绝对路径/step_XXXXXX.pt` 明确接续，launcher 保留原来的严格身份校验。

`VALIDATION.json` 记录配置/命令/启动脚本哈希。`NODE_PROBE.json` 记录四机 SSH、H200 数量、网络设备、共享路径和训练配置一致性。端口空闲是检查时状态，未预占该端口。
