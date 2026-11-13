# AV＋Event＋Feature＋Whisper：四机32卡实测与执行优化

本轮已完成固定样本测速和18步真实更新验收，通过后停在完整第18步检查点，没有启动完整8927步。结果见 [test32/COMPLETE.json](test32/COMPLETE.json)。原500h、原始冻结Qwen H、最终裁框／净化listener音频缓存、speaker Whisper和teacher不变；损失仍为AV＋0.05 Event＋0.025视频Feature＋0.025音频Feature。

## 真实资源与预检

- node17 `172.29.15.91`、node12 `172.29.252.37`、node4 `172.29.225.51`、node8 `172.29.104.237`，各8张H200。
- 未停止其他任务。新500h增量VAE缓存自动开始后，先等它的GPU编码正常结束；连续两次确认32卡空闲再启动本轮。
- 285638条联合缓存收据全量预检通过，8个CPU进程，286.83秒。H、Gemma、AV、Event、Feature teacher和speaker Whisper严格按来源关联，未重新编码媒体。
- 73项CPU回归通过；另以3条真实数据验证后台线程独立SQLite连接、CPU缓存读取、样本顺序和训练RNG不变。
- 32rank最长11.95秒片段，两次完整联合forward/backward/Adam通过，最大rank显存预留峰值96.734375GiB；随后恢复初始参数、Adam和CPU／CUDA RNG。

## 三组同样本测速

固定实际调度第3–18步的512条样本及噪声ordinal。全部使用同一初始权重、global batch32、全层重算、BF16前后向、FP32可训练参数、GPU驻留Adam、FP64梯度SUM。学习率置零以保持权重，排除预热、梯度哈希和模型哈希开销；不累计正式训练步数。

| 执行方案 | 平均秒／步 | 输入等待秒／步 | 最大rank预留显存GiB |
|---|---:|---:|---:|
| 原有256MiB桶，串行读取 | 5.734607 | 0.143664 | 107.355469 |
| 256MiB桶，后台预取2批 | 5.497998 | 0.000127 | 106.951172 |
| **512MiB桶，后台预取2批** | **5.471520** | **0.000148** | **106.675781** |

三组全局AV、Event及两路Feature损失逐值一致；完整裁剪后梯度SHA256一致；模型所有状态在零学习率测速前后逐值不变。全部实际测得峰值低于预设128GiB门槛。

采用最后一组：步时减少约4.59%，吞吐提高约4.81%。**主要收益来自输入预取**；512MiB相对256MiB预取方案仅快约0.48%，这一次小样本实测不足以证明该差异显著，也不称为全局最优配置。

完整逐步时间、损失、梯度与来源：[benchmark/RESULT.json](test32/benchmark/RESULT.json)。

## 已落地的代码

- `src/p2_whisper_event_feature/execution.py`：有界CPU输入预取；只读取冻结缓存，一条后台线程，每rank预取2批，GPU仍然每次一条。预取不接触CUDA、噪声生成器或训练随机数。
- `data.py`：预取线程使用自己的只读SQLite连接，主线程先完成数据库完整SHA核验；不跨线程共用SQLite连接，不跳过缓存payload的SHA校验。
- `training.py`：按原时长计划提交和按序消费，保留每条样本的ordinal；新增输入读取／等待时间日志；退出时关闭并等待有界预取。联合损失归一化和优化器不变。
- 配置新增 `combined_execution: {"prefetch_batches":2,"gradient_bucket_mb":512}`。原纯Whisper对照的配置和源码未修改；新配置中的原有 `whisper_training_execution` 保留对照来源，新实验以 `combined_execution` 中两个实测覆盖项为实际设置。
- [32卡测速入口](../../scripts/p2_whisper_event_feature_benchmark.py)与[测试控制器](../../scripts/p2_whisper_event_feature_test32.py)：只操作自己的进程，监控等待与运行阶段；固定三组测速后自动验证18步真实更新并停下，不进入全量训练。

## 18步真实训练验收

2026-11-05 11:41:20 UTC 验收结束：[test32/state.json](test32/state.json)、[metrics.jsonl](metrics.jsonl)、[test32/COMPLETE.json](test32/COMPLETE.json)。全部通过：

1. 32rank真实完成第1–18步，五组参数均存在有限、非零梯度，全部损失有限，联合权重公式正确。
2. 初始权重和第18步完整模型／Adam／32rank RNG检查点成功写入，SHA256和检查点身份通过。
3. 训练按测试范围停止；最终保存步为18，预算仍是8927。正常全量继续时每500步保存，不改变训练预算或学习率计划。

后续严格从 `step_000018.pt` 恢复本实验即可保留这18步进度；不能用旧纯AV Whisper的加载器恢复。`test32`控制器用于一次性新运行验收，已有run时会拒绝覆盖。

真实更新第3–18步平均 **5.821668秒／步**，约5.50条／秒。它包含实际参数更新与正式训练日志开销，与固定权重测速的5.471520秒／步分别报告。外推剩余8909步约14.41小时，建议按**14–16小时**理解；仅16个计时batch，完整数据的描述长度、时长分布、保存与I/O变化仍会影响耗时。

检查点：`step_000018.pt`，SHA256 `300f904a8df3a68ea6136dc232f6ee63a03c87aeeb03ad9049b3f0b83e46fe36`。测试停止时额外保存第0／18步，正常继续后仍每500步保存。尚不能由18步loss判断本实验的生成质量收益或收敛趋势。

## 继续训练

确认四机资源可用后，在项目根目录执行：

```bash
../Omni-LTX-stage2/.venv/bin/python scripts/p2_j_whisper_event_feature.py launch \
  --resume experiments/p2_j_whisper_event_feature_20261105/step_000018.pt \
  --logs experiments/p2_j_whisper_event_feature_20261105/continue18
```

命令会重新进行全量收据预检，再按原身份恢复模型、Adam和各rank RNG。本次没有执行该全量继续命令。
