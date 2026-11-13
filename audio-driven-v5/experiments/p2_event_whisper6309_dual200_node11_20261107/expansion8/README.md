> 八卡真实推理验收通过：8个rank均已提交有效输出；原107条latent及metadata SHA逐条复核未变，固定测试清单、权重、seed与采样参数一致。见 `LIVE_ACCEPTANCE.json`。

# node11八卡扩容

用户授权node11全部8卡评测同一6309步权重：固定ResponseNet分层随机200＋Seamless固定200，共400条；原有结果保留，seed137／Euler30／CFG3、7及条件／GT缓存不变。

原GPU7 renderer进程2012453及其在途样本原样保留；仅终止原单卡controller PID，避免其提前进入下游阶段。原controller、worker、协议、cluster已在本目录归档。新controller接管整体阶段顺序，不触碰其他任务或训练。

rank0继续在物理GPU7；新增rank1–7对应GPU0–6。共享逐片段锁＋complete标记，不按固定分片等待，完整结果不重算。全部render进程退出且400条latent均完整后，才进入8卡共享解码；全部视频完成后接原生bind、冒烟、视频／音频评分及全200特征汇总。

评分集群允许GPU0–7，每卡最多4路；按实际可用卡领取，不驱逐GPU7上的另一任务。FID/FVD按各测试集200条全量特征计算，不平均分片分数。

入口：`scripts/p2_event_whisper6309_expand8_20261107.py`。当前controller状态在本目录 `CONTROLLER.json`／`state.json`；各进程在 `processes/{render,decode}`，实际CUDA映射在 `devices/`。`PREPARED.json`验证新旧科学字段完全一致，并记录原renderer未被停止。源码快照在 `code_snapshot/`。

最终结果仍写主目录 `RESULTS.json`／`RESULTS.md`；完成以 `FINAL_AUDIT.json` 与 `HANDOFF_COMPLETE.json` 为准。单卡旧入口不要再次启动。STOP仍使用主目录STOP，在样本边界停止领取；用户停止标记不能自动移除。
