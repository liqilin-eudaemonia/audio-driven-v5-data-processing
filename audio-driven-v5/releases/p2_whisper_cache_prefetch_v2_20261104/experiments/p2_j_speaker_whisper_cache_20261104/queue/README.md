# 打标结束后自动提取 speaker Whisper 缓存

2026-11-04 已挂后台守护。范围：当前仅AV＋Whisper实验的原500h训练集，**285,638条**，四机32卡。当前只冻结来源／模型协议并等待，未抢占正在打标的GPU。

前置任务：`experiments/listen_j_reaction_vocal_omni_20261103/`，1,107,323条固定清单的检测与打标。

顺序：

1. 等待producer EOF、检测完成、打标队列排空。
2. 核对固定清单验收：总数一致、身份／队列版本一致，无重复、遗漏、未处理或处理失败。
3. 原打标的 `generation_failed` 复核标记和 `insufficient_evidence` 分流仍保留，不提升为通过标签；整体 `completed_with_failures` 状态如实记录。
4. 关闭已完成的原资源巡检控制器，等待四机GPU实际释放；不杀其他任务。
5. `prepare → 32卡extract → finalize`。只提取speaker Whisper，不启动训练。

输入固定为每条样本的配对speaker归一化音轨及源起止；16kHz、冻结Whisper-large-v3、50Hz、FP32，保留前导静音。旧Qwen／Gemma／AV缓存不改。**新版输出为 `experiments/p2_j_speaker_whisper_cache_20261104_prefetch_v2/cache/<sample_id>/`**，全量验收成功后在该新版目录发布 `COMPLETE.json`，不新增WAV。本队列位置保持不变，旧v1协议保留。

当前每卡batch16、4个读取线程、2个CPU计算线程、双缓冲跨批次预取及1个后台写盘线程，终检使用16个CPU读取线程；显式FP32并关闭TF32，每卡启动先做短／跨窗合批一致性检查。独立torchrun端口29844。四机模型权重、入口及SSH连接均已实查。[提速实现与验证](../../../related/P2_Whisper缓存流水线提速与接续_20261104.md)。使用已有已知主机公钥生成本队列的 `known_hosts`，未修改原训练集群配置。

入口：[p2_speaker_whisper_after_annotation.py](../../../scripts/p2_speaker_whisper_after_annotation.py)。守护每30秒检查，等待阶段每20秒轮询；缓存阶段读取各rank进度，启动或处理停滞20分钟会取消本队列自己的缓存worker并重试；完整缓存结果复用，最多5轮。`STOP` 会停止本队列，不自动重启训练。

- [当前状态](state.json)：`waiting_predecessor → waiting_gpu_release → extracting → auditing_cache → complete`。
- [冻结队列配置](queue.json)、[四机实际配置](cluster.json)、[已挂接记录](armed.json)。
- `guard.json`／`controller.json`：PID、进程起始标识和心跳；`controller.log`／各次 `launch_*` 保存日志。
- `PREDECESSOR_AUDITED.json` 保存前置验收摘要和原状态，`progress.json` 汇总32rank缓存进度，`resources.json` 保存空卡确认。
- 接续相关13项CPU检查通过，含缺失／陈旧验收拒绝、复核标记保留、完整mock接续及缓存专用CLI。

```bash
# 从audio-driven-v5执行，只查看状态：
/private/liqi_workspace/project_code/Omni-LTX-stage2/.venv/bin/python \
  scripts/p2_speaker_whisper_after_annotation.py status \
  --work experiments/p2_j_speaker_whisper_cache_20261104/queue
```

Whisper全量缓存预计约460.8GB净张量，队列预留至少512GiB可用空间；当前共享盘空间已确认足够。
