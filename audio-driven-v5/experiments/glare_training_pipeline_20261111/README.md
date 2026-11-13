# GLARE microbatch、传输预取和连续分片试验

2026-11-11，node11 GPU7。采用256条真实缓存，两个长度175/299帧，固定有效batch256、FP16、reaction_loss_weight0.05。GLARE主训练在试验前完成当前更新并保存，试验结束后恢复。

## 结果

- microbatch32的暖机后计算时间约2.6–3.1秒，64约2.0秒。128触到试验进程约23.9GiB的显存额度。此额度给同卡任务预留余量，测试结果只适用于本次资源条件。
- 两组有界CUDA传输预取完成，原生loss对照通过。microbatch64开启预取约3.38秒，这轮未显示性能收益。
- 64条缓存约463MB的连续分片读取0.17–0.31秒，逐条读取约0.70秒，CPU张量相等。这是热缓存和临时目录的小规模测试，临时文件已清理，共享Weka全量冷读取收益尚待验证。
- microbatch64接入生产后36个观测更新约2.65秒/更新，未优于此前32。GPU7同时运行其他视频生成和vLLM，负载变化使测试结果存在干扰。已恢复microbatch32。

记录见RESULT.json、probe.log、PRODUCTION_RESULT.json。较早并发测量因可用显存下降触到约9.9GiB额度，记录在RESULT_short.json，不用于比较性能。

## 当前运行

生产入口更新为scripts/train_glare_paper_settings_v6.py及run_glare_paper_settings_v6.sh。准确PID与启动参数在原生产目录launch.json。有效batch256、FP16、reaction权重0.05，保存和验证每2000步。

64测试及恢复32均接续模型、优化器、scaler和RNG。检查点保存当轮epoch_order，切换时保留已经消耗的前缀，只重新组织未消耗的长度桶。样本顺序随microbatch分桶调整。额外检查点用于安全切换。
