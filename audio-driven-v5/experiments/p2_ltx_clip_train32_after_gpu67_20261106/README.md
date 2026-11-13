**此32卡运行已在3982步完整保存后接续到[31卡新运行](../p2_ltx_clip_train31_cadence_20261107/README.md)。旧STOP保留；当前保存频率以新运行的1000→500分段规则为准。**

**当前正式运行：固定all全重计算，已从305步完整状态恢复并持续越过329步故障批次。四机32卡、batch32、每300步完整保存不变；四分之三候选已停用。实时步数以status.json为准。**

[实际状态](status.json) · [实时性能](PERFORMANCE_STATUS.json) · [恢复验收](RESTORE_ALL_VERIFIED.json) · [9小时守护](../p2_train32_guard9h_20261106/README.md) · [8小时验收](../p2_train32_guard9h_20261106/EIGHT_HOUR_AUDIT.json) · [TensorBoard](http://172.29.252.37:6006)。

每15秒持续记录GPU和步时，PERFORMANCE_STATUS.json为实时统计，performance.jsonl为历史。当前策略固定all；候选停滞与回退证据见[ROLLED_BACK.json](../p2_ltx_clip_recompute32_20261106/ROLLED_BACK.json)，不再自动尝试three_quarters。

**保存频率已按最新要求改为每300次参数更新。当前双卡配置仍为50，切换时只修改此保存频率，其他优化器与训练参数不变。预约时间不变。**

# 双卡→四机32卡预约接续

预约时间：2026-11-06 14:50:41 UTC 起检查资源并执行切换。时间到之前，当前node11 GPU6/7双卡继续训练。

四机：node12 172.29.252.37（node rank0，global rank0–7）、node8 172.29.104.237（1，8–15）、node4 172.29.225.51（2，16–23）、node11 172.29.189.119（3，24–31）。每机GPU0–7全部参与；协调端口29836，SSH使用root:22。

global batch32、每卡1条、每300步保存完整checkpoint，沿用当前优化器、学习率、warmup进度、样本列表、标签及采样计划。只改运行目录和并行world；不会从头训练，也不沿用早期测试权重。

后台协调器coordinator.py已独立启动，PID见COORDINATOR.json。QUEUE_STATE.json为实时状态，coordinator.log为日志。

## 自动切换步骤

1. 在等待期间完成新配置的全量CPU preflight。
2. 到预约时间后，检查四机32卡。除了当前训练占用的node11 GPU6/7，其余卡必须无其他计算任务，且显存/利用率满足空闲条件；资源不满足则继续等待，当前双卡继续运行，不结束其他任务。
3. 符合条件后，向双卡训练创建STOP，等待它完成当前更新、保存完整checkpoint并正常退出。
4. 使用届时最新的完整checkpoint迁移world2→32。模型和Adam状态保留；配置除output目录外完全一致。每个新rank的RNG来自旧rank r%2，逐样本仍按global ordinal重新设种子。校验未来样本分片与原顺序一致；不宣称不同world未来浮点轨迹逐位相同。
5. 迁移后再检查所有32卡连续两次空闲，启动四机torchrun。worker继续执行严格checkpoint/源码/数据身份校验。若代码或数据在等待期间变化，切换前会拒绝启动。

MIGRATION.json和实际COMMANDS.json在切换时生成；COMMANDS_TEMPLATE.json仅为静态检查示例，其中EXAMPLE_CHECKPOINT.pt不是可运行的resume路径。

## 取消与停止

创建本目录CANCEL可以取消尚未启动的预约。若已经进入双卡正常停止阶段，取消会保留已保存的状态，不擅自重启旧双卡。

四机训练启动后，创建本目录STOP可在当前更新完成后保存退出。

## 已完成检查

四机SSH资源探测成功；现有双卡PID已被正确识别为本任务。资源判定的空闲/本任务占用/其他任务占用规则已检查；四机命令的GPU、world、rank、端口、绝对路径以及2→32的样本分片已验证。READY_CHECK.json记录控制器和配置哈希。正式GPU迁移将在预约条件满足后执行。


## TensorBoard（本轮）

- node12内网入口：[http://172.29.252.37:6006](http://172.29.252.37:6006)。页面服务已实际读取并对齐训练JSON中的9项loss；包含16项scalar（9项loss、6项梯度、1项步时）。
- 日志目录：`tensorboard/`；服务身份与启动命令：[TENSORBOARD_SERVER.json](TENSORBOARD_SERVER.json)；HTTP与数据核验：[TENSORBOARD_VERIFIED.json](TENSORBOARD_VERIFIED.json)。
- 服务设置`CUDA_VISIBLE_DEVICES=''`及OMP/BLAS线程限制，不占训练GPU；每10秒重载。scalar保留10000点，覆盖本轮8882步预算。
- 浏览器在外部时，沿用平时连接node12的SSH主机、端口和密钥参数，追加`-L 16006:127.0.0.1:6006`，随后打开`http://127.0.0.1:16006`。若可直接SSH到内网节点，可用：

```bash
ssh -N -L 16006:127.0.0.1:6006 root@172.29.252.37
```
