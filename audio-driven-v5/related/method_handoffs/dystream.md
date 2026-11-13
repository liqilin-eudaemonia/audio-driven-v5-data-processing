# DyStream：运行与交接

**Seamless完整test6624已接续（2026-11-12）：** 历史双200已完成；全量沿用同模型和指标，由GPU4/GPU7常驻Monitor按容量接续。[全量进度/数值](../../experiments/p2_reference_methods_seamless6624_20261112/RESULTS.md) · [运行与Monitor](../../experiments/p2_reference_methods_seamless6624_20261112/README.md)。

**2026-11-12 双200补测进行中：** 本轮仅历史固定ResponseNet200＋Seamless200。[实时数值表](../../experiments/p2_reference_methods_dualtest_20261112/RESULTS.md) · [本轮模型、输入适配与恢复](../../experiments/p2_reference_methods_dualtest_20261112/README.md)。历史ViCo结果继续保留；进行中不代表双200已完成。

更新：2026-11-11。状态：**官方权重43条视频与统一指标完成**。本页汇总已验证的本地证据；历史资产核查不代表今天重新验证了外部下载链接。

## 当前训练、权重位置和断点恢复

主权重：[last.ckpt](<../../compare/DyStream/checkpoints/last.ckpt>)；配套tools和音频编码器按部署清单加载。另有[真实数据短训及恢复验收](<../../experiments/baseline_training_realdata_20261109/README.md>)，每阶段仅4次更新，用于接通训练和冷恢复，不作为论文成绩。本地诊断检查点：[finetune/checkpoint.pt](<../../experiments/baseline_training_realdata_20261109/runs/dystream/finetune/checkpoint.pt>)。 正式ViCo推理复用官方权重，无新增训练。

## 官方推理命令和输入参数

[官方README](<../../compare/DyStream/README.md>)，仓库内UI命令：
```bash
CUDA_VISIBLE_DEVICES=7 python -u app.py
```
批量输入JSON包含 `resampled_video_path, audio_path, audio_self_path, audio_other_path, motion_self_path, motion_other_path, mode, dataset_type, video_id`；本地完整调用见[部署命令](<../../compare/deployment/README.md>)。

## 数据处理、缓存及训练／推理适配

listener self音轨全零，speaker进入other分支；首帧已裁好，crop=False；seed137，官方5步和CFG，512/25生成后映射256/30评分。[统一推理加速入口](<../../compare/deployment/vico_inference_fast.py>)复用逐片段输出；缺失片段的原生加速入口[vico_dystream_fast.py](<../../compare/deployment/vico_dystream_fast.py>)保留模型与精度，仍有node8/GPU7约束。不要把UI端口当作SSH端口。

## 评测结果与下一步

[已完成43条评测及比较表](<../../experiments/p2_vico_full_eval_20261111/README.md>)。只生成视频，不声称生成listener语音。

[返回方法总索引](README.md)。已有结果按各自协议保留，不把短训、辅助权重或仅通过导入检查记为正式方法复现。

**后续ResponseNet完整586已入队：** Seamless完整6624完成后自动接续，使用GPU4/GPU7空闲资源。[完整586进度与结果](../../experiments/p2_reference_methods_responsenet586_20261112/RESULTS.md) · [运行与队列](../../experiments/p2_reference_methods_responsenet586_20261112/README.md)。

**2026-11-13最新资源限制：** Seamless6624及后续ResponseNet586仅使用node11 GPU7；GPU4不可用。历史双200及已完成产物保留。
