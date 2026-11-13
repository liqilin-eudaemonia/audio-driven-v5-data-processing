# 梯度统计与FP64通信流水化：31卡对照

在原始H＋AV＋Event＋Whisper第1000步完整保存后暂停，8＋8＋8＋7卡，node8 GPU7继续预留。保留原训练源码，候选实现放在独立模块。两rank CPU跨桶／缺梯度／Adam逐值对照2项通过。

每组使用同一保存的模型和Adam、同一248条样本及噪声序列，1步预热＋8步计时。前面使用零学习率，在最后一批做一次真实学习率更新；均不计入正式训练。GPU模型加载、hash验证不在计时内，梯度统计在计时内。

| 配置 | 秒／步 | 相对首次原配置 |
|---|---:|---:|
| baseline | 5.990082 | +0.00% |
| grouped_stats | 5.935310 | -0.91% |
| grouped_stats_pipeline | 6.131575 | +2.36% |
| baseline_repeat | 6.605771 | +10.28% |

四组的全局损失、完整裁剪梯度、模型实际更新、完整Adam状态和梯度统计均逐值一致；显存均低于128GiB。原始输出见`RESULT.json`，采用判断见`DECISION.json`。

**结论：本轮没有足够证据支持切换。** 统计合并仅约0.9%的表观改善，原配置首尾复测相差约10.3%；不能将这个小样本差异作为稳定全程收益。通信流水化相对首次原配置更慢。候选代码保留供以后独立测试，生产继续使用原配置。

测速结束自动移除本次拥有的临时STOP并重启原顺序队列，从原1000步完整模型／Adam／31份RNG恢复；Feature＋Whisper仍在后续队列。正式训练checkpoint没有采用任何测速权重。未恢复Goal。

代码：`src/p2_whisper_execution_trial.py`、`scripts/p2_whisper_ablation_benchmark.py`、`scripts/run_p2_whisper_execution_trial.py`；回归`tests/test_whisper_execution_trial.py`。源码及SHA备份见`code_snapshot/MANIFEST.json`。正式恢复以`RESUME_VERIFIED.json`为准（生成后才表示已确认）。
