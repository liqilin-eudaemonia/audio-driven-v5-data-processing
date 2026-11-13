# ViCo baseline / RLHG：运行与交接

更新：2026-11-11。状态：**仅诊断训练；正式重训在飞书主表暂缓**。本页汇总已验证的本地证据；历史资产核查不代表今天重新验证了外部下载链接。

## 当前训练、权重位置和断点恢复

RLHG与ViCo baseline为同一条方法。官方示例Epoch_500.bin未确认为本地可用正式权重。另有[真实数据短训及恢复验收](<../../experiments/baseline_training_realdata_20261109/README.md>)，每阶段仅4次更新，用于接通训练和冷恢复，不作为论文成绩。本地诊断检查点：[checkpoint.pt](<../../experiments/baseline_training_realdata_20261109/runs/vico/checkpoint.pt>)。

## 官方推理命令和输入参数

[官方README](<../../compare/vico_challenge_baseline/README.md>)，仓库内listener入口：
```bash
python eval.py --batch_size 4 --output_path saved/baseline_listener_E500 \
  --resume saved/baseline_listener/checkpoints/Epoch_500.bin --task listener
```
以上checkpoint为官方预期路径，不能据此宣称文件存在。

## 数据处理、缓存及训练／推理适配

BFM73动作、45维音频、训练集mean/std，PIRender为独立渲染阶段。[部署与缺权重核查](<../../compare/deployment/README.md>)。数据和短训接入沿用官方网络，不把辅助重建器当成driver。缓存组件优化不等于完整推理已加速验收。

## 评测结果与下一步

暂无正式43条模型成绩；按现有暂缓配置保留。获得匹配核心权重或另行完成正式训练后再评测。

[返回方法总索引](README.md)。已有结果按各自协议保留，不把短训、辅助权重或仅通过导入检查记为正式方法复现。
