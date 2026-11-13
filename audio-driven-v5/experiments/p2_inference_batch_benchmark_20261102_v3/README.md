# 推理合批实测：保留当前正式配置

2026-11-02，在node17 GPU0、GPU1分别暂停本队列对应的单个推理进程，使用相同GPU完成对照；其他22卡继续正式推理。测试结束后两个进程均已恢复，远端状态均为R，记录见 [PRODUCTION_RESUMED.json](PRODUCTION_RESUMED.json)。未修改正式协议、既有生成结果或检查点。

四条真实测试输入：两条seamless短片段（25fps，原生81帧）、两条ResponseNet长片段（29.97fps，原生241帧），均512×512。AV＋Event step6,000，同一缓存、seed137、30步Euler、video CFG3／audio CFG7。每次测试使用同一对样本，报告两条平均耗时；这是采样阶段计时，不包含缓存、权重加载、VAE解码及导出。

| 配置 | seamless短片段 秒/条 | ResponseNet长片段 秒/条 |
|---|---:|---:|
| 当前单样本／分开CFG | 18.07 | 53.07 |
| 单样本／合并CFG | 16.62 | 51.83 |
| 双样本／分开CFG | 16.68 | 51.85 |
| 双样本／合并CFG | 15.73 | 51.41 |

两种合批同时开启时，短片段耗时降低约13.0%，长片段降低约3.1%；测试进程峰值分配显存分别为38.61GiB、43.33GiB。该数值不包含暂停的正式进程持有的显存。不能将四条小样本的加速比例直接当作全量平均收益。

## 一致性及采用决定

原生基线跨两轮重复的视频和音频latent均逐值一致。合并CFG／双样本采样遵循相同Euler方程、每样本随机种子、条件mask与首帧固定规则；非连续张量的随机数排列也已保持，CPU回归检查通过。但真实BF16计算改变batch后，30步最终输出不再逐值一致。最佳组合的视频latent相对L2差异：短片段约5.54%，长片段约15.05%；音频约1.79%和1.67%。首帧固定部分均逐值一致。

这些差异不等于感知质量下降，也不是模型质量评分；尚未通过生成视频的完整指标证明质量保持。考虑目前长片段仅约3%采样提速、正式评测已经开始生成，本轮保留原生单样本、分开CFG的正式配置，避免在同一对照实验中混入未经质量验证的新计算路径。没有测到成倍提速，也没有据此断言其他优化都无效。

## 结果与复现

- [汇总](SUMMARY.json)、[短片段完整结果](seamless_short/results.json)、[长片段完整结果](responsenet_long/results.json)。对应目录保留各方法latent及输入ID。
- 单样本合并CFG数据取自[v2短片段](../p2_inference_batch_benchmark_20261102_v2/seamless_short/results.json)和[v2长片段](../p2_inference_batch_benchmark_20261102_v2/responsenet_long/results.json)；其输入和基线与v3逐值相同。v2的跨样本两项曾使用不正确的随机数内存排列，已废弃，表中跨样本数据全部取修正后的v3。
- 实现：`src/p2_testsets_eval/batch_sampling.py`；受控测试入口：`scripts/p2_eval_batch_benchmark.py`，命令见 [launch.json](launch.json)。这些候选未接入正式运行时。
- 检查：`CUDA_VISIBLE_DEVICES='' ../Omni-LTX-stage2/.venv/bin/python -m unittest discover -s tests -p test_eval_batch_sampling.py`。
