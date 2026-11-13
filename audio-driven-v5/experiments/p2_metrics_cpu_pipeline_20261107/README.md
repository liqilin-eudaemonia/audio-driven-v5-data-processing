# 评测CPU读取与像素流水化验证

[完整实现、配置、使用与约束](../../related/P2_评测CPU预取与像素计算流水化提速_20261107.md)。

同样本实测：Seamless四条59.20→35.65秒，ResponseNet四条78.03→55.45秒；所有逐条指标与FID/FVD特征／汇总完全一致，GT缓存身份与命中保持。7项单元测试通过。单进程微基准，不直接代表8路或全量倍数。

最终报告为RESULTS.json，完成标记STATUS.json；基线／流水线原始指标分别在seamless/、responsenet/，8条固定样本清单在本目录。代码快照、SHA和unit_tests.log一并保存。已完成step6500正式400条结果未被改动或重算。
