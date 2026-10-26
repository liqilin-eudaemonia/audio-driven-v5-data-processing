# P2 AV / AV＋Event：text-attention LoRA 实现与验证

> 后续已补齐 [TensorBoard 实时训练监控与恢复](P2_TensorBoard训练监控与恢复.md)：当前 LoRA/H200 配置默认开启，包含全局指标聚合和恢复后的曲线清理，146 项 CPU 测试及真实 GPU 验证通过。

> 后续 GPU 验证已完成：3 机 24 卡、E0/E1 单/双卡一致性、实际续训及独立生成通过；新增 H200 执行配置和 140 项 CPU 测试，见 [24 卡联合训练 GPU 验证与提速](P2_24卡联合训练GPU验证与提速.md)。下文 136 项测试和无空卡状态是实现当时的记录。

2026-10-26。按用户确认，在已备份的无 feature 版本上，为 E0/E1 同时加入原生 LTX text cross-attention LoRA。固定基线仍在 [GitHub c72c10a](https://github.com/liqilin-eudaemonia/audio-driven-v5-data-processing/tree/p2-av-event-no-feature-20261026/p2_training/av_event_20261026)；该标签不包含本次 LoRA 修改。

## 1. 要解决的问题和对照

Connector 将 Qwen hidden 转成 U，并与原生文本条件拼接。允许 LTX 的文本注意力投影适配这些新条件，降低“事件信息存在于 U，但冻结的生成接口不能利用它”的风险。仍通过 E0/E1 实际生成比较检验事件监督，不以分类头拟合替代效果结论。

| 模块 | E0 / probe_only | E1 / joint_event |
|---|---|---|
| Connector | AV loss | AV＋0.05×Event loss |
| 原生 text-attention LoRA | AV loss | AV loss |
| 事件头 | detach(U)，训练自身 | U，事件梯度回到 Connector |
| feature decoder / teacher | 关闭／不读取 | 关闭／不读取 |
| Omni、Gemma、VAE、LTX 原生参数 | 冻结 | 冻结 |

Event loss 不直接回传到下游 LoRA。它改变 U 后，会间接影响之后的 AV 训练。Connector 和 LoRA 的生成参数共同裁剪，事件头单独裁剪，避免 E0 的 probe 梯度改变生成参数的裁剪系数。两组从相同初始化独立联合训练；没有新增前置对齐阶段。数据、seed、预算及采样协议保持匹配。

## 2. 配置与注入范围

- [新基础配置](../configs/p2_listen_av_event_text_lora_v1.json)
- [新配对协议](../configs/p2_listen_av_event_text_lora_pair_v1.json)
- 原无 LoRA 配置 `p2_listen_av_event_v1.json` 保留，旧配置默认不启用 LoRA；CLI 默认配置也没有静默更换。

新增字段位于 `training`，不进入缓存提取器配置：

```json
"feature_loss_enabled": false,
"text_lora": {
  "enabled": true,
  "rank": 8,
  "alpha": 8.0,
  "learning_rate": 0.00001
}
```

复用 [v4 P2.1 注入器](../../audio-driven-v4/src/omni_av_p21/lora.py)，不修改其源码。当前真实权重的模型配置为 48 个 Transformer blocks：各 block 的 `attn2` 和 `audio_attn2`，分别在 `to_q`、`to_k`、`to_v`、`to_out.0` 注入，共 **384 个线性层**。这不包括 self-attention、音视频交叉注意力、FFN 或原生归一化层。共享投影同时处理文本和 U，因此 LoRA 并非只作用于 U token。

rank/alpha/LR 是本轮初始配置，不是经过效果搜索的最优值。B 矩阵零初始化，初始前向保持原生输出；adapter 主参数为 FP32，训练继续使用 bf16 autocast、FP64 梯度累加与跨卡求和。LoRA 参数 **18,874,368**，总可训练参数 **53,336,199**（Connector＋事件头＋LoRA）。这些计数已用真实 22B 权重的配置在 meta device 构建核验，未加载预训练张量。

## 3. 代码职责

| 文件 | 修改 |
|---|---|
| [config.py](../src/p2_listen/config.py) | 严格验证可选 LoRA 字段，统一优化器组 LR；检查点源码身份纳入复用注入器 |
| [model.py](../src/p2_listen/model.py) | 仅注册 Connector、事件头和 LoRA；保持 renderer 内 adapter 与 DDP 根中 adapter 为同一对象；检查参数所有权 |
| [training.py](../src/p2_listen/training.py) | renderer 先加载，随后按固定 seed 构建模型；AV 计算放入 DDP forward；LoRA 独立 AdamW 组与相同 warmup，纳入同步、裁剪、保存和恢复 |
| [generation.py](../src/p2_listen/generation.py) | 输入特征和首帧 VAE 准备后加载 LTX；注入 adapter 后严格加载完整训练状态；丢弃 Connector 后 renderer 继续保有同一套已加载 adapter |
| [GPU 重放](../scripts/check_p2_listen_gpu_replay.py) | 先构建 renderer/adapter 再恢复，包含 LoRA 优化器状态与学习率 |
| [CPU 审计](../src/p2_listen/audit.py)、[两进程检查](../scripts/check_p2_listen_cpu_ddp.py) | LoRA 模式使用小尺寸原生 LTX Attention，不加载预训练权重；检查 E0/E1、恢复、无标签 rank |

检查点只保存可训练模块，不复制冻结的 22B 权重。config、实际目标层列表、源码和基座身份写入 run/pair；LoRA 与无 LoRA 架构不能直接混用恢复，改变 rank/alpha/LR 也不能绕过严格恢复检查。推理缺失 adapter 权重或层列表不一致会报错，不静默退回原生模型。

## 4. 推理和缓存边界

- `normal`：加载 Connector 和 LoRA，共享 U 进入 CFG 正负两分支。
- `audio_swap`：新 speaker 音频重新提取输入，使用同一检查点的 Connector 和 LoRA。
- `remove_u`：移除 U，但保留训练后的 LoRA，用于诊断该模型对 U 的依赖。
- `text_only`：原生 LTX 文本/首帧参照，不加载训练 LoRA。

仍只读取 input、VAE、text、events，已有 teacher 留作后续消融。冻结提取器、缓存身份源码、v4 源码、全量特征抽取配置保持原样。50 段验证目录通过软链接复用四类原缓存，不建立 teacher 目录；其 debug 身份保留，只能用于 S0。

## 5. 运行入口

先按原数据导入与弱标签契约准备独立 output，显式选新配置。已有身份匹配的缓存可复用；不需要重新提取 input/VAE/text/events。

```bash
P2_PY=/private/liqi_workspace/project_code/Omni-LTX-stage2/.venv/bin/python
P2_CFG=/path/to/formal_av_event_text_lora_config.json

# 仅在已有数据和缓存、且指定 GPU 空闲时执行；以下不是已运行记录。
CUDA_VISIBLE_DEVICES=0 "$P2_PY" scripts/p2_listen.py train --config "$P2_CFG" --arm E0 --train-seed 20261026 --s0 --steps 5
CUDA_VISIBLE_DEVICES=0 "$P2_PY" scripts/p2_listen.py train --config "$P2_CFG" --arm E1 --train-seed 20261026 --s0 --steps 5

# S0 后，正式数据上的两组使用相同配置预算，独立于 S0 run。
CUDA_VISIBLE_DEVICES=0 "$P2_PY" scripts/p2_listen.py train --config "$P2_CFG" --arm E0 --train-seed 20261026
CUDA_VISIBLE_DEVICES=0 "$P2_PY" scripts/p2_listen.py train --config "$P2_CFG" --arm E1 --train-seed 20261026
```

上述为单卡启动模板，实际启动前应核对资源占用。后续已在 node12/node17/node4 完成真实 GPU 验证；正式效果训练尚未启动，全量特征缓存未因 LoRA 或执行优化重算。

## 6. CPU 验证与后续 GPU 验收

[验证目录](../experiments/p2_listen_text_lora_validation_20261026/README.md)、[汇总](../experiments/p2_listen_text_lora_validation_20261026/summary.json)。

- **136 项 CPU 测试通过**：含 17 项新增 LoRA 测试，覆盖目标层、零初始化、所有权、冻结基座、两种事件梯度路径、梯度裁剪、精确恢复、推理严格加载、normal/remove_u/text_only 调用路径和 bf16 autocast。
- **原生 Attention 的两进程 CPU/Gloo 对照通过**：E0/E1 各 3 步，含无事件监督 rank、不同类别有效计数和累积；LoRA 与整体参数的单/双进程最大差均为 0。无 LoRA 模式回归也为 0。
- **CPU 5 步审计和下一步恢复通过**：AV 梯度到两路 U 和 LoRA；E0 事件梯度不进 U，E1 事件梯度进 U；事件项到 LoRA 为 0。
- **真实架构 meta 检查及 50 段真实缓存检查通过**：48 blocks / 384 目标层；无 teacher 访问，四类缓存身份、哈希和有限值通过。

**后续已运行本版完整预训练 LTX GPU S0、GPU 恢复和独立音视频生成。** 在用户释放的 3 机 24 张 H200 上完成，包含加速前后逐位一致性验证和新的 H200 配置，详见 [24 卡联合训练 GPU 验证与提速](P2_24卡联合训练GPU验证与提速.md)。正式生成收益仍待成对效果实验。
