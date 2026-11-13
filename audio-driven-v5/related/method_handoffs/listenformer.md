# ListenFormer：运行与交接

**Seamless完整test6624已接续（2026-11-12）：** 历史双200已完成；全量沿用同模型和指标，由GPU4/GPU7常驻Monitor按容量接续。[全量进度/数值](../../experiments/p2_reference_methods_seamless6624_20261112/RESULTS.md) · [运行与Monitor](../../experiments/p2_reference_methods_seamless6624_20261112/README.md)。

**2026-11-12 双200补测进行中：** 本轮仅历史固定ResponseNet200＋Seamless200。[实时数值表](../../experiments/p2_reference_methods_dualtest_20261112/RESULTS.md) · [本轮模型、输入适配与恢复](../../experiments/p2_reference_methods_dualtest_20261112/README.md)。历史ViCo结果继续保留；进行中不代表双200已完成。

更新：2026-11-11。状态：**本地完整200轮ViCo重训；43条动作、视频与统一指标完成**。本页汇总已验证的本地证据；历史资产核查不代表今天重新验证了外部下载链接。

## 当前训练、权重位置和断点恢复

当前正式ViCo检查点：[best.pt](<../../experiments/p2_vico_additional_baselines_20261107/listenformer/best.pt>)；训练恢复用[last.pt](<../../experiments/p2_vico_additional_baselines_20261107/listenformer/last.pt>)，协议和完成记录在同目录。它们不是作者公开checkpoint；此前分享取消后未找到匹配的官方完整权重。`vico_more_train.py listenformer train` 会恢复模型、优化器、调度、epoch游标与随机状态；200轮已完成，不需要重开。另有[真实数据短训及恢复验收](<../../experiments/baseline_training_realdata_20261109/README.md>)，每阶段仅4次更新，用于接通训练和冷恢复，不作为论文成绩。本地诊断检查点：[checkpoint.pt](<../../experiments/baseline_training_realdata_20261109/runs/listenformer/checkpoint.pt>)。

## 官方推理命令和输入参数

[官方README](<../../compare/ListenFormer/README.md>) 的eval.sh对应命令（在官方仓库内，Epoch_200.bin为官方示例名，并非当前本地best.pt格式）：
```bash
CUDA_VISIBLE_DEVICES=7 python -u eval.py --batch_size 1 \
  --output_path saved/listenformer_E200 \
  --resume saved/listenformer/checkpoints/Epoch_200.bin --task listener
```


## 数据处理、缓存及训练／推理适配

使用BFM73、音频45维、仅训练集统计；speaker序列＋listener首帧，不读未来listener。[训练/测试特征与缓存](<../../experiments/p2_vico_additional_baselines_20261107/features>)。PIRender权重：[renderer checkpoint](<../../experiments/p2_vico_additional_baselines_20261107/assets/renderer/face/epoch_00190_iteration_000400000_checkpoint.pt>)。[统一推理加速入口](<../../compare/deployment/vico_inference_fast.py>)的listenformer generate/render在node11/GPU7运行，逐条复用已验证回执并只加载一次模型。
```bash
CUDA_VISIBLE_DEVICES=7 compare/.envs/vico_more/bin/python compare/deployment/vico_inference_fast.py listenformer generate
CUDA_VISIBLE_DEVICES=7 compare/.envs/vico_more/bin/python compare/deployment/vico_inference_fast.py listenformer render
```


## 评测结果与下一步

[已完成43条评测及比较表](<../../experiments/p2_vico_full_eval_20261111/README.md>)。与paper数字只能说明差距，不能把本地重训权重等同作者原模型。

[返回方法总索引](README.md)。已有结果按各自协议保留，不把短训、辅助权重或仅通过导入检查记为正式方法复现。

**后续ResponseNet完整586已入队：** Seamless完整6624完成后自动接续，使用GPU4/GPU7空闲资源。[完整586进度与结果](../../experiments/p2_reference_methods_responsenet586_20261112/RESULTS.md) · [运行与队列](../../experiments/p2_reference_methods_responsenet586_20261112/README.md)。

**2026-11-13最新资源限制：** Seamless6624及后续ResponseNet586仅使用node11 GPU7；GPU4不可用。历史双200及已完成产物保留。
