# P2 首轮 AV / AV＋Event 对照：关闭 feature decoder

2026-10-26。用户已选择先验证“事件监督是否改善生成”，当前实现加入显式 AV-only 基础损失模式。完整 feature 版本的独立备份为 [157 文件源码快照](../experiments/p2_training_with_feature_backup_20261026T032623Z/README.md)。本次没有把历史完整模型检查点转换成简化模型，也没有启动正式效果训练。

## 1. 本轮对照

| 项目 | E0 / probe_only | E1 / joint_event |
|---|---|---|
| Connector 更新 | AV loss | AV loss＋0.05 × Event loss |
| 事件头输入 | detach(U) | U |
| 事件头自身更新 | 是 | 是 |
| feature decoder / loss | 不构建／不计算 | 不构建／不计算 |
| teacher 缓存读取 | 无 | 无 |

两组共享初始化、全局样本顺序、噪声 seed、训练预算、七类弱背景标签与 mask。Omni、LTX、VAE、Gemma 继续冻结；LTX 反向计算继续向 U 和 Connector 传递 AV 梯度。事件头不作为生成效果的最终裁判。

## 2. 配置入口

新基础配置：[p2_listen_av_event_v1.json](../configs/p2_listen_av_event_v1.json)。配对声明：[p2_listen_av_event_pair_v1.json](../configs/p2_listen_av_event_pair_v1.json)，不自动启动任务。

```json
"training": {
  "feature_loss_enabled": false,
  "connector_lr": 0.00003,
  "event_lr": 0.0001
}
```

上面仅展示有关字段。完整模板保持 global batch 8、1000 步、warmup 50、事件权重 0.05、clip norm 1、weight decay 0。弱背景策略为 `event-interval-weak-background-v1`，几何上限采用与当前全量抽取一致的 305 对齐帧、512×512、25 FPS。

**必须明确传入新配置。** CLI 的历史默认配置仍是 `p2_listen_event_v2.json`，未指定该布尔字段的旧配置仍启用 feature 分支。恢复完整分支使用 `feature_loss_enabled=true` 并提供 `feature_lr`，开始新的独立实验。模式开关严格要求 boolean，字符串 `"false"` 或整数 0 会被拒绝。

`extractor.feature_video_weight` / `feature_audio_weight` 保留原值 1，以保持冻结提取器及缓存身份兼容；关闭分支由新的显式开关控制。这些旧权重不会在关闭模式中产生 feature 计算。简化模板不需要 `feature_lr`。

## 3. 实际修改

| 文件 | 行为 |
|---|---|
| [config.py](../src/p2_listen/config.py) | 解析模式、条件验证 feature_lr，统一选择训练所需缓存部分 |
| [model.py](../src/p2_listen/model.py) | 关闭时构建 `Extractor(auxiliary=False)`；没有两路 decoder、没有对应 state_dict 参数、没有 features 优化器组；意外传入 teacher 会报错 |
| [training_data.py](../src/p2_listen/training_data.py) | 新的训练缓存读取适配；关闭时 teacher 返回 None，完全不访问 teacher 路径；完整模式沿用原 loader |
| [training.py](../src/p2_listen/training.py) | 只累加实际存在的 feature 项；关闭时 base 就是 AV；跳过 feature 梯度审计；记录 `feature_loss_enabled=false`、`feature_losses=null`、独立 AV loss |
| [generation.py](../src/p2_listen/generation.py) | 使用共同的 `build_model(c)`，按检查点配置重建无 decoder 架构 |
| [p2_listen.py](../scripts/p2_listen.py) | 关闭模式的 `cache --part all` 只选择 input/text/vae/events；显式 `--part teacher` 仍可为后续消融生成缓存 |
| [恢复重放脚本](../scripts/check_p2_listen_gpu_replay.py) | 按相同配置构建模型、读取训练数据，无 teacher 依赖 |
| [CPU 审计](../src/p2_listen/audit.py)、[CPU 双进程检查](../scripts/check_p2_listen_cpu_ddp.py) | 按所选模式检查 E0/E1 和恢复；双进程脚本新增 --config |

关闭后只有 `connector`、`events` 两个 AdamW 参数组，可训练参数为 **34,461,831**；完整模式约 223,975,559。保留已有确定性计算、FP32 事件头和 FP64 梯度累加/跨卡 SUM。

模式写入 config、run、pair、status 与 checkpoint identity。两种模式的权重键和 optimizer 参数组也不同，严格恢复会拒绝混用。源码变化仍受原严格 source identity 约束，历史检查点不能绕过检查直接续训。

## 4. 缓存和正式数据接入

这组训练只读取 input、VAE、text、events 四类缓存。teacher 文件可以继续保留供后续消融。

本次没有修改 `features.py`、`layout.py`、`annotations.py`、P3 冻结提取代码或复用的 v4 源码，也没有修改正在运行的全量抽取配置。新增开关放在 `training` 中，不进入冻结提取器配置。验证确认两种模式的已有缓存身份一致，避免新训练模式使全量特征缓存失效。

正式数据使用独立实验 output；导入同一批正式 observation/target 和弱标签后，将身份匹配的 input/vae/text/events 缓存关联到新 output。不要直接在旧 run 上更换标签、预算、模式或数据清单后续训。

```bash
P2_PY=/private/liqi_workspace/project_code/Omni-LTX-stage2/.venv/bin/python
P2_CFG=/path/to/new_formal_av_event_config.json

# 新数据尚无缓存时：关闭模式不会生成 teacher。
"$P2_PY" scripts/p2_listen.py cache --config "$P2_CFG" --part all --split train --device cuda:0

# 正式数据先做短 S0；这两个命令需在已导入数据并缓存后执行。
"$P2_PY" scripts/p2_listen.py train --config "$P2_CFG" --arm E0 --train-seed 20261026 --s0 --steps 5
"$P2_PY" scripts/p2_listen.py train --config "$P2_CFG" --arm E1 --train-seed 20261026 --s0 --steps 5

# 正式配对训练使用非 S0 run；预算来自配置，E0/E1 完全匹配。
"$P2_PY" scripts/p2_listen.py train --config "$P2_CFG" --arm E0 --train-seed 20261026
"$P2_PY" scripts/p2_listen.py train --config "$P2_CFG" --arm E1 --train-seed 20261026
```

以上为接入模板，未启动正式训练。已有的 50 段带 debug 身份，仍不允许进入正式训练；弱背景标签已有正负支持，普通 S0 不需要旧的 `--debug-positive-events` 开关。

## 5. 本轮验证与边界

证据目录：[p2_listen_av_event_validation_20261026](../experiments/p2_listen_av_event_validation_20261026/README.md)，机器可读汇总：[summary.json](../experiments/p2_listen_av_event_validation_20261026/summary.json)。

- **119 项 CPU 测试全部通过**，其中新增 14 项覆盖布尔配置、缓存兼容、decoder 不构建、teacher 不读取、优化器所有权、E0/E1 梯度、全 unknown 时 AV 仍更新、恢复与推理架构。
- **两进程 CPU/Gloo 对照通过**：关闭与保留 feature 两种模式，E0/E1 各连续 3 步，单/双进程模型参数最大差均为 0；包含某 rank 没有事件监督及逐类计数不等。
- **无 feature CPU 审计通过**：E0 事件梯度为 0，E1 事件梯度到视听 U 非零；5 步后保存恢复，下一步参数完全一致。
- **50 段真实缓存通过**：新验证目录根本没有 teacher 目录，并启用 teacher 文件打开拦截；四类缓存身份、内容哈希和有限值均通过。完整模式在同目录会因缺 teacher 被正常拒绝。
- **备份和缓存保护通过**：核验修改前备份及冻结缓存指纹文件，保留旧 teacher、原数据和历史验证结果。

本轮没有重新运行完整 LTX GPU S0：检查时 `172.29.104.220` 的所有 GPU 均被现有任务使用，授权范围内没有空卡。旧完整 feature 版本的真实 GPU 验收不充作本次简化结构的 GPU 验收；简化结构的 GPU S0 和生成效果比较仍需在空卡上执行。本次未停止、调低或重启全量抽取任务。
