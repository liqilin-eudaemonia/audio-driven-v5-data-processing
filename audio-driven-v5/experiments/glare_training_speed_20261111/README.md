# GLARE 训练执行优化：2026-11-11

仅改执行：按长度复用位置编码及attention mask（LRU上限256）；8次累积的loss合并回传CPU，在optimizer.step之前检查全部loss有限性。模型、数据顺序、参考帧／音频窗口、loss、优化器及有效batch不变。没有开启混合精度、修改batch分组或移除同卡vLLM。

独立模块`src/baseline_training/glare_execution_fast.py`，独立入口`scripts/train_glare_cached_fast.py`及`run_glare_cached_training_fast.sh`，旧入口保留。首次接旧checkpoint显式`--allow-original-resume`：逐项检查旧源代码指纹及除执行源码外所有协议字段，记录execution_migration.json；新checkpoint后续恢复仍严格检查身份。支持SIGTERM后完成当前更新并保存，避免今后调整时丢失进度。

## 实测

256条真实pair，原版／优化／优化／原版，各40个更新、每更新8条，统计各轮去掉前4更新的墙钟。原版0.5073秒/更新，优化0.3669秒/更新：耗时降低27.7%，吞吐提高38.3%。同卡正式训练同时运行，有资源竞争；这是交替短测，不能当作独占GPU或全程训练的速度保证。

40步后全部训练参数allclose(atol=1e-6, rtol=1e-5)；优化对原版最大差7.45e-9，原版重复间最大差1.12e-8。不是逐位相同；原生浮点随机执行未改。

详见SUMMARY.json、RESULT.json、benchmark_long.log。接旧300步checkpoint后训练到302并保存，再独立进程恢复到304成功，见resume_smoke.log和resume_cold.log。测试产物不用于正式训练。

正式切换等待原版2000步checkpoint提交，保留该原版权重及protocol；切换实际状态以生产launch.json为准。保存和验证继续每2000步。

已实际切换：从生产2000步checkpoint恢复，旧版权重保留，生产训练继续。PRODUCTION_RESULT.json记录上线前后吞吐及重放loss差异；受同卡负载及样本变化影响，不把上线前后比例当作严格因果测试。
