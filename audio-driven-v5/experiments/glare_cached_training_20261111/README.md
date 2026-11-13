# GLARE 最终500h缓存训练接入

2026-11-11。准备代码，不自动启动正式训练。最终缓存汇总与训练准备并行；生产入口未发布时训练必须失败退出，不能拿未完成清单开始全量训练。

## 代码与既定方案

- `scripts/train_glare_cached.py`：单卡／torchrun DDP、每pair前向＋梯度累积、dev固定子集验证、最新与最佳checkpoint、恢复。
- `src/baseline_training/glare_cached.py`：验证sealed train/dev清单，按字节偏移读取；加载LIA、标准化音频、Wav2Vec2投影前特征、SER、动作统计、reaction及mask、官方intensity。
- `scripts/run_glare_cached_training.sh`：固定现有可用Python和依赖overlay，明确分配GPU才启动。

遵循用户已定官方优先方案：复用官方FLOAT.forward、原生FMT、flow采样、L1动作／速度损失、SmoothL1反应损失、官方多卡脚本的可训练参数组、AdamW lr/weight_decay均1e-5、裁剪1、10%warmup余弦调度。不采用旧4步诊断的1e-4学习率。步数由启动命令显式指定，不能把默认100 epoch当作已获批准的计算预算。

必要适配：每个完整pair取第0帧参考、其余帧作目标、无历史，与已提取缓存一致；不随机裁子窗。复用冻结Wav2Vec2/SER，audio_projection保留梯度；六类监督采用本项目生产标签和有效mask。官方NULL／尾部默认零继续保留。不同长度逐pair前向并累积，不添加视频padding。每rank默认8pair/update，有效batch=8×world；显式`--accumulate`可调整。多卡每epoch仅丢弃不能均分的最后至多world-1条，下一epoch重新洗牌。

音频端点：缓存严格用(N-1)×640采样点。官方浮点切片在201/203目标帧时可能少1点；缓存模式直接使用整型窗口已提取特征，明确修正该端点，不宣称与少1点的官方波形重新编码逐位等价。统计只用于训练，不授权用测试listener GT统计推理。

## 启动及恢复（准备命令，不表示已启动）

GPU资源和训练步数实际确定后执行：

```bash
CUDA_VISIBLE_DEVICES=<已分配卡号> NPROC_PER_NODE=<本机卡数> \
  bash scripts/run_glare_cached_training.sh \
  --output experiments/glare_500h_training \
  --steps <明确的优化器更新步数> --accumulate 8
```

多机设置NNODES、NODE_RANK、MASTER_ADDR、MASTER_PORT，所有节点共享相同output与registry。恢复增加`--resume experiments/glare_500h_training/latest.pt`，保持原steps/world/accumulate/源码/数据身份；不重复消耗已经训练的游标。保存训练参数、AdamW、调度位置、每rank RNG及数据游标。权重初始化及官方代码指纹记录在protocol.json。

CPU数据检查：使用同一Python/overlay运行`train_glare_cached.py --output /tmp/glare_check --steps 1 --check-data 8`。只有sealed生产清单发布后可运行。`--check-data`不初始化GPU。

## 验证边界

5项CPU测试通过：官方零强度及未知反应mask读取、回执变化拒绝、payload损坏拒绝、窗口契约检查、原子checkpoint。3条真实缓存加载与全payload SHA核对通过。单GPU真实前向／反向及checkpoint重载结果见SMOKE.json（仅文件存在且通过才算完成）；多rank同步和跨进程冷恢复尚需短训验收，不能把代码准备视为多机正式训练已验收。dev loss是固定最多64条均匀子集的原生随机flow损失，固定验证RNG，不是完整dev评测或最终视频指标。

全量发布后补充验收：生产sealed train清单的8条均匀分散样本加载通过，见`production_read_check.log`。最终缓存清单完整性核验见全量运行目录的`COMPLETION_AUDIT.json`。
