# GLARE 500h：28卡全量运行

**已完成（UTC 2026-11-11 00:58／北京时间 11-11 08:58）。** 284,218 pair全缓存汇总成功：train 281,060、dev 3,158，缺失0；946,304韵律块完成。两阶段各224任务全部验收，四机supervisor均已结束。最终成员及split与原500h选集完全一致，未启动正式训练。

[最终缓存入口](TRAINING_CACHES_LATEST.json) · [完成核验](COMPLETION_AUDIT.json) · [训练接入与短测](../../glare_cached_training_20261111/README.md)。汇总已逐条检查payload存在、回执及身份；此次最终验收复核清单与seal哈希，没有再次全量读取所有张量字节。8条分散真实样本通过新训练读取器。summary中的production_training_reader_ready=false是汇总器固定保守标记，不能据此忽略独立训练接入记录；多卡正式训练仍需短测验收。

源帧缺口的1条恢复已被正常task122回执验收，详见文末。韵律仍沿用官方NULL零强度及原质量限制，不宣称人工标签质量已通过。


2026-11-10用户明确授权启动。run_id：ce67701c79aa55337118d81f294afdb02352cf42918ef59f1a6d41148256636d。

范围284,218 pair（原train/dev划分）、946,304原音轨韵律块。node8 172.29.104.237、node12、node4各0–7，node11仅0–3。初始8卡韵律、20卡LIA组合缓存，阶段完成后自动接续另一阶段。每卡LIA主进程1、CPU检测服务6、解码进程4、在途组16；MTCNN GPU批32 FP32禁用TF32，LIA原fused2 FP32，韵律batch64完整官方生成。

旧缓存按协议复用，补齐LIA、标准化speaker音频、六类反应及mask、动作统计、Wav2Vec2投影前特征／SER、韵律条件。官方NULL零强度保留，不额外剔除原答质量不合格pair。GPU执行数值差异见[验收](../gpu_fer_fixed_640/README.md)。

## 状态与输出

- [配置](../production_gpu_fer_config.json)、[全量状态](status.json)：finalize.py每60秒更新。
- [完成汇总状态](finalize_status.json)：watching→assembling→complete；异常单列，未完成不发布最终入口。
- 各机`nodes/<node>/status.json`、各卡`gpuN/workers/gpuN.json`（组合缓存）／`gpuN/status.json`（韵律）。大任务尚未完成时总队列done可能为0，卡内状态显示实际新增及复用。
- 非韵律：`../../glare_current500h_shared_cache_20261110/cache/`；冻结音频`../../glare_current500h_shared_cache_20261110/frozen_audio/cache/`，旧父LIA有效缓存仍可引用。
- 韵律原答：`../../glare_current500h_prosody_subset_20261110/cache/`；汇总条件`../../glare_current500h_prosody_subset_20261110/official_conditions/`。
- 最终全缓存train/dev入口：本目录`TRAINING_CACHES_LATEST.json`（仅汇总成功后生成）。

四机supervisor脱离SSH后台运行，子进程失败按现有上限恢复，任务断点可复用；finalize.py仅监控本run并在全部完成后调用assemble，不会启动训练。它遇到停止或终止性失败就退出并记录状态，不伪造完成。

## 命令

在项目根目录执行：

```bash
python scripts/run_glare_cache_cluster.py --config experiments/glare_cache_28gpu_20261110/production_gpu_fer_config.json status
python scripts/run_glare_cache_cluster.py --config experiments/glare_cache_28gpu_20261110/production_gpu_fer_config.json stop
python scripts/run_glare_cache_cluster.py --config experiments/glare_cache_28gpu_20261110/production_gpu_fer_config.json start
```

需要错误恢复时先修复具体错误，再执行相同入口retry-errors及start。若finalize_status已停止／失败，重启本目录finalize.py；它有独占锁防止重复汇总。原早期GPU7任务不恢复。

## 半小时运行巡检

用户已授权持续监控与异常修复。独立后台`monitor.py`每1800秒核对四机主进程实际存活、GPU使用、队列及新增/复用吞吐，结果为`monitoring/PERIODIC_LATEST.json`和逐次`.periodic.json`；`periodic.log`记录摘要。该进程只观察，实际异常由持续goal排查修复；观察超时不等于任务停止。吞吐下降提示需结合阶段切换、缓存复用和收尾判断，不能仅凭GPU瞬时0%重启。

机器重启后，在本目录用后台方式启动`monitor.py`即可恢复周期巡检；它有独占锁防止重复实例。完成全量及最终汇总后监控自动退出；停止请求存在时也退出。

## 单条源帧缺口恢复

运行中task122包含一条源视频100.1ms帧间隔，导致目标310.59秒最近实际帧相距46.43ms，超过本项目自加约33ms检查。采用仅该sample/target/source三者精确匹配的读取例外，仍使用原有最近真实帧，不插值；此为本项目适配，不声称官方提供同一恢复策略。补提1条、复用1257条成功，原源码和其余缓存不变，任务已重新开放正常验收。[记录](monitoring/incidents/task0122_source_gap.json) · [真实补提结果](monitoring/incidents/repair0122/RESULT.json)。
