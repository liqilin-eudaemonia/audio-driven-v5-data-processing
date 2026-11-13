# DIM（Dyadic Interaction Modeling）：运行与交接

更新：2026-11-11。状态：**核心生成权重未就绪；正式重训暂缓**。本页汇总已验证的本地证据；历史资产核查不代表今天重新验证了外部下载链接。

## 当前训练、权重位置和断点恢复

code/test_s2s.py预期saved_checkpoints_vico/best_model30fps_ftvq.pth，当前未验证取得完整可用组合。需要配对speaker VQ、listener VQ、predictor和统计。另有[真实数据短训及恢复验收](<../../experiments/baseline_training_realdata_20261109/README.md>)，每阶段仅4次更新，用于接通训练和冷恢复，不作为论文成绩。本地诊断检查点：[listener_codec/checkpoint.pt](<../../experiments/baseline_training_realdata_20261109/runs/dim/listener_codec/checkpoint.pt>)、[predictor/checkpoint.pt](<../../experiments/baseline_training_realdata_20261109/runs/dim/predictor/checkpoint.pt>)、[speaker_codec/checkpoint.pt](<../../experiments/baseline_training_realdata_20261109/runs/dim/speaker_codec/checkpoint.pt>)。

## 官方推理命令和输入参数

[官方README](<../../compare/Dyadic-Interaction-Modeling/README.md>)，在code目录：
```bash
python vico_preprocessing.py
python test_s2s.py
```
前者是预处理、后者是ViCo推理；先检查脚本内checkpoint和数据路径。官方训练分别为train_vq.py、train_s2s_pretrain.py、finetune_s2s_pretrain.py，不可拿随机初始化替代正式权重。

## 数据处理、缓存及训练／推理适配

EMOCA pose6＋expression50及HuBERT768；不能以DECA系数直接冒充。两阶段短训与codec缓存见[真实数据短训及恢复验收](<../../experiments/baseline_training_realdata_20261109/README.md>)。保持官方模型和损失，仅适配本项目数据。

## 评测结果与下一步

本地缺正式生成checkpoint；DIM指标函数已经用于统一评分，不等于DIM生成模型已经评测。

[返回方法总索引](README.md)。已有结果按各自协议保留，不把短训、辅助权重或仅通过导入检查记为正式方法复现。
