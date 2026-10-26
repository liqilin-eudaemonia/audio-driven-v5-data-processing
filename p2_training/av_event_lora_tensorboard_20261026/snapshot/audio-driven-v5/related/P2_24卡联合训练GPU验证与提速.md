# P2：24 卡联合训练 GPU 验证与提速

> 后续已补齐 [TensorBoard 实时训练监控与恢复](P2_TensorBoard训练监控与恢复.md)：当前 LoRA/H200 配置默认开启，包含全局指标聚合和恢复后的曲线清理，146 项 CPU 测试及真实 GPU 验证通过。

日期：2026-10-26。测试对象是 **AV / AV＋Event＋匹配 text-attention LoRA**，无 feature decoder、无 teacher 读取。真实 22B LTX 权重、512×512、25 FPS、LoRA rank/alpha=8；保留 BF16 前向、FP32 可训练参数、FP64 梯度累加和确定性训练策略。

本次完成正确性及性能验证，没有启动正式效果实验，也不能由这些短跑判断事件监督改善了生成。

## 1. 机器、数据与验收

| 节点 | 地址 | GPU |
|---|---|---|
| node12 | 172.29.252.11 | 8×H200 |
| node17 | 172.29.15.91 | 8×H200 |
| node4 | 172.29.225.51 | 8×H200 |

开始时确认 24 卡空闲。主要测试复用此前通过验收的 50 段、约 213 秒缓存，来源为 `p2_listen_ddp_fix_v2_20261025`，使用既有弱背景派生标签，仍标记 debug-only，禁止用于正式效果结论。没有改写原始标签、覆盖范围或缓存；缺少摇头正例，该类在这批 S0 中不参与事件损失。另从全量已完成缓存取真实 11.948 秒样本 `naturalistic_V03_S0178_I00000382_local_9fdf4dcfe3ba`，只用于最长序列的显存和性能压力测试。

已经通过：

- 原执行方式和加速方式分别运行 E0/E1 单卡、双卡各 7 步；各自单/双卡最终参数最大误差 **0**。
- 加速前后分别比较 E0/E1：841 个模型 tensor、2520 个优化器 tensor、2 个 RNG tensor **逐位一致**。初始化、数据顺序、监督统计和训练预算匹配。
- 加速版 E1 在第 6 步停止，通过真实 `--resume` 续到第 7 步；E0/E1 都再次恢复并重算下一步，参数、优化器和 loss 逐位一致。
- 加速版正式训练入口在 **3 机 24 卡完成 5 步 E1 S0**，global batch=24；冻结权重检查正常。
- 单卡梯度审计：AV 到 U_video/U_audio、LoRA 均有梯度；E0 Event 不回传 U，E1 有效事件类别回传对应 U；Event 不直接回传下游 LoRA。
- 新版本检查点独立完成 `normal` 和 `remove_u` 音视频生成，后者保留训练后的 LoRA；生成不读取 target/标签。
- **140 项 CPU 测试通过**；两进程 CPU/Gloo 检查覆盖无标签 rank、全部 mask 的稀有类，E0/E1 参数误差均为 0。

证据：[实验目录](../experiments/p2_lora_gpu24_20261026/)、[加速前后等价性](../experiments/p2_lora_gpu24_20261026/optimization_equivalence.json)、[加速版单/双卡](../experiments/p2_lora_gpu24_20261026/optimized_pair_comparison.json)、[E0 恢复](../experiments/p2_lora_gpu24_20261026/optimized_replay_E0.json)、[E1 恢复](../experiments/p2_lora_gpu24_20261026/optimized_replay_E1.json)。

## 2. 实测速度

每个性能组合运行 6 次真实 forward/backward/optimizer update，第一步作为预热，报告后 5 步均值。表中吞吐不含模型加载、全量缓存预检、保存检查点及生成评测；含每步 CPU→GPU 输入搬运。性能脚本使用真实训练目标和优化器，固定样本顺序及噪声；学习率固定，正式训练入口另用 S0 验证 warmup 和恢复。

| 卡数 | global batch | 执行方式 | 秒/步 | 样本/秒 |
|---:|---:|---|---:|---:|
| 1 | 24 | 原方式：全部层检查点、逐参数通信 | 51.00 | 0.47 |
| 8，同机 | 24 | 原方式 | 8.88 | 2.70 |
| 8，同机 | 24 | 加速版正式实现 | 7.98 | 3.01 |
| 24，跨机 | 24 | 原方式＋修正后的网络配置 | 4.16 | 5.76 |
| 24，跨机 | 24 | 隔层检查点＋32 MiB 通信桶 | **3.60** | **6.66** |
| 4，同机 | 8 | 隔层检查点＋32 MiB 通信桶 | 5.15 | 1.55 |
| 8，同机 | 8 | 隔层检查点＋32 MiB 通信桶 | **3.22** | **2.48** |

24 卡的加速配置相对同样 24 卡原方式，吞吐提升约 **15.5%**。相同 batch=24，8 卡与 24 卡加速版的最终模型 SHA256 完全一致；24 卡各 rank 的参数也一致。其训练时间窗口内 `nvidia-smi` GPU utilization 均值约 **96.5%**。这个指标包含通信内核的等待，不能等同于有效算力效率，因此决策主要看样本吞吐。

50 段混合样本的加速配置显存峰值约 89–90 GiB/卡。最长样本正式加速实现为 **6.27 秒/步、114.2 GiB**，连续 6 步完成。对照：全部层检查点约 6.90 秒/步、45.0 GiB；四分之三层检查点约 6.59 秒/步、78.7 GiB。全关检查点虽然更快，但仅 7.872 秒样本就达到约 136 GiB，不能作为最长约 12 秒数据的默认配置。

完整分项与显存、采样功率、利用率口径：[benchmark_summary.json](../experiments/p2_lora_gpu24_20261026/benchmark_summary.json)。这只是短时性能测量，不是全量数据长时间吞吐保证。正式数据的长度、输入 token 数和缓存存储性能会影响结果；当前逐样本全量启动预检未计入表格。另测 722,706 样本规模时，现有 sampler 每步每 rank 生成排列约 0.20 秒（8 卡、batch=8），也不应把小数据集吞吐直接当作全量速度。

## 3. 已落地的配置与实现

新配置：[H200 训练配置](../configs/p2_listen_av_event_text_lora_h200_v1.json)、[两 seed 配对协议](../configs/p2_listen_av_event_text_lora_h200_pair_v1.json)。保留 global batch=8、1000 步及原学习率；只增加：

```json
"performance": {
  "gradient_checkpointing": "alternating",
  "gradient_bucket_mb": 32
}
```

`gradient_checkpointing` 支持 `all`、`alternating`、`three_quarters`、`none`。`gradient_bucket_mb=0` 采用原逐参数通信；正整数采用相应 MiB 的桶。旧配置不填这两个选项时维持原行为。

实现位于 [performance.py](../src/p2_listen/performance.py)：只替换当前 renderer 实例的 block 检查点调度，不修改共享 LTX 包，不改变 block、LoRA 的参数路径；训练、恢复和生成都经过统一 `build_model`。生成阶段 eval 不做检查点重算。通信继续在 FP64 累加并 SUM，最后一次转换到参数 dtype；单独同步参数是否使用，保留 `grad=None`，不改变全局损失归一化和梯度裁剪。

冻结特征提取代码及缓存身份代码未改，旧缓存无需因本次执行优化重算。执行选项和源码指纹进入新检查点身份；不要把旧源码产生的检查点直接塞给已改变的源码恢复。原版本在本次实验的 `code_before/`，当前版本在 `code_after/`，各自对应已保存的验证记录；此前 GitHub 无 LoRA 固定标签未被覆盖。

## 4. 跨机网络问题及已验证解法

默认网络首次在建链时出现 `ibv_modify_qp: No such device`。显式 GID=3 后，GPU Direct 路径仍出现 `invalid request local work queue error`。这些失败没有进入有效训练更新，相关日志保留。

最终通过的配置使用 NCCL 内置 IB、RoCE v2 的 GID=3、8 条 400 Gbit/s 网卡、关闭本环境出错的 GPU Direct 网络路径，使用主机内存中转，同时关闭 NVLS。**仍使用 RoCE，不是退回 eth0 的 10 Gbit/s 数据传输**；eth0 只负责 bootstrap。

完整配置：[p2_listen_h200_multinode.env](../configs/p2_listen_h200_multinode.env)。只对启动 shell 生效，没有修改机器全局网络设置。24 rank 的 4 / 32 / 416 MiB SUM 通信逐元素校验通过；8 网卡中转下 416 MiB all-reduce 约 6.95 ms，随后真实 24 卡训练通过。

在三台机器分别运行相同的 torchrun 命令，只有 node-rank 为 0/1/2。必须使用同一份、global batch 能被 24 整除的配置：

```bash
source configs/p2_listen_h200_multinode.env
/private/liqi_workspace/project_code/Omni-LTX-stage2/.venv/bin/python \
  -m torch.distributed.run --nnodes=3 --nproc-per-node=8 \
  --node-rank=0 --master-addr=172.29.252.11 --master-port=29632 \
  scripts/p2_listen.py train \
  --config experiments/p2_lora_gpu24_20261026/optimized_24gpu/config.json \
  --arm E1 --train-seed 20261027 --s0 --steps 5
```

上述为可重跑的有限 S0 示例，换了 seed 避免覆盖本次证据；正式训练不要使用 debug 数据或 `--s0`。

## 5. 当前成对实验如何分配 24 卡

当前计划是两 training seeds × E0/E1，共四个实验，global batch=8。推荐维持这套优化预算，按下表并行：

| 机器 | GPU | 作业 |
|---|---|---|
| node12 | 0–7 | seed 20261023 / E0，8 ranks |
| node17 | 0–7 | seed 20261023 / E1，8 ranks |
| node4 | 0–3 | seed 20261024 / E0，4 ranks |
| node4 | 4–7 | seed 20261024 / E1，4 ranks |

每对 E0/E1 的卡数、初始化、数据顺序、预算都匹配；8 rank 每卡每步 1 个样本，4 rank 每卡累加 2 个样本，总 batch 都是 8。单机用 `torchrun --standalone --nproc-per-node=8` 或 `4`，通过 `CUDA_VISIBLE_DEVICES` 指定上述 GPU；两份四卡进程各自 standalone，避免端口冲突。单机无需跨机网络环境，设置 OMP/OPENBLAS threads=2、CUBLAS 工作区及 expandable_segments 即可。

按小样本稳态速度粗算，1000 步单个八卡作业约 54 分钟、四卡作业约 86 分钟；这是外推，未计启动预检、存盘、评测，也未实测两份四卡作业同机长期并发的影响。相比三台各跑一份、第四份再排队，该分配能让四份实验同时开始。

若目标改为最快完成单个训练作业，可以用 24 卡、batch=24；但这会改变优化批量。仍训练 1000 步会从 8,000 变为 24,000 个样本位置，不是原预算的直接提速。正式切换应让 E0/E1 同时使用新批量，并重新明确步数预算。本次仅用于压测，未擅自更改正式协议。

正式数据就绪后，应从其已准入且缓存身份匹配的配置接入上述训练选项；新模板的 output 不是自动选定的正式数据目录。此文交付的是已验证执行方式和性能范围，不代替数据筛选、标签准入或生成效果评测。
