> 已自动接续：最新SFT2752的400条规划／H／Gemma缓存完成，step6000原生模型在两套测试集均产生有限latent，描述／H／Gemma及adapter身份校验一致；见 `GPU_VALIDATION.json`。生成、解码和最终评分仍以实时状态及最终验收为准。

# 接续评测：LTX主干整段Event＋Feature＋Whisper6000＋最新SFT2752

本轮已完成CPU验收并排队，**等待当前6309步模型的两套200条生成、解码及评分全部完成**后，自动接node11（172.29.189.119）GPU0–7，不中断当前评测，也不恢复训练。队列见 `status.json`、`CONTROLLER.json`、`queue.log`、`QUEUE_LAUNCH.json`。

## 明确使用的权重

- LTX：`experiments/p2_ltx_clip_train31_cadence_20261107/step_006000.pt`，原完整权重SHA `f702a031bd127b7993b46c872fdc1f8f02a37cdfb7badece9def0ae9cf7f6bc0`。使用6000步，不是6026步。架构为AV＋0.05 **LTX最后block hidden整段Event**＋两路0.025 Feature＋speaker Whisper。
- 最新规划器：`experiments/p2_planning_sft_upright_3b_32gpu_resume_20261106/training/final_adapter/`，原始Qwen2.5-Omni-3B＋r32/alpha64语言LoRA，已完成2752步；训练与控制器COMPLETE均通过。不是旧2697步adapter。
- SFT模型既生成描述，也对自身生成描述提取H；Gemma根据新描述重新编码。LTX训练用的是原始H，本轮明确记录为“训练base H、推理SFT2752 H”，不修改checkpoint声明。

两套测试集保持当前固定ResponseNet分层随机200＋Seamless固定200，总400条。seed137、Euler30、video CFG3／audio CFG7、512画面与帧数网格保持。只提供同一speaker音轨和listener首帧，不读未来listener画面、真值音轨或GT描述作输入。

## 来源校验与缓存

完整6000步权重在CPU做SHA、模型有限性、推理版逐张量一致性检查；不带Adam的推理版在 `checkpoints/step_006000.pt`，原完整checkpoint保留。两份真实样本分别覆盖ResponseNet、Seamless，speaker波形与既有Whisper缓存逐字节一致，提示词与最新SFT训练run.json完全一致，见 `CPU_ACCEPTANCE.json`。实际GPU加载、两套样例生成及新条件对应关系已通过检查，见GPU_VALIDATION.json；最终指标仍待完成。

31卡保存后，三个共享源码文件因24卡训练接续而更新。独立bootstrap `src/p2_ltx6000_sft_eval_20261107.py` 在当前进程内执行SHA匹配的31卡归档config／resume31／schedule，保留工作区和正在运行的24卡训练。所有其他绑定文件按原checkpoint摘要核验，不放宽来源校验。实际执行归档与workspace摘要见 `FROZEN_EXECUTION.json`。

`conditions/`是本轮独立目录：仅复用首帧PNG及输入审计；**不复用原生Qwen的qwen/或text/正条件**。`whisper/`和空Gemma负条件复用。ResponseNet沿用原生pair speaker WAV读取器，Seamless沿用官方speaker切片读取器；运行时逐条验证重采样后waveform SHA，保留时长、静音与pair原点，不混入listener音轨。

阶段：新SFT规划＋H＋Gemma缓存 → 八卡共享领取生成 → 八卡共享解码 → 原生视频／音频评分 → 各测试集全200条FID/FVD汇总。沿用常驻模型／VAE、无损GT RGB与特征缓存、指标batch64、每卡最多4路评分。缓存阶段只向有充分空闲显存的卡分配八个固定shard；若GPU7仍有其他任务，该shard可由先空闲的卡接续，避免其他7卡等待或缓存OOM。不停止其他任务。

## 入口与结果

- 入口：`scripts/p2_ltx6000_latest_sft_dual200_20261107.py`；CPU `prepare`，排队 `queue`，各卡 `worker`。
- `protocol.json`冻结模型、adapter两文件SHA、测试清单和执行代码；`adapter_session/protocol.json`为独立推理session，不打开旧训练H或teacher。
- `CPU_PRECHECK.json`／`CPU_ACCEPTANCE.json`：来源、权重导出、latest-SFT、真实输入、条件隔离。
- `progress/{cache,render,decode}`／`devices/`：逐卡实际进度；`metrics_status.json`：评分。
- 媒体：`generated/ltx_hidden_event_feature_whisper_6000_sft2752/{responsenet,seamless}/<id>/generated.mp4`和`listener.wav`。
- 汇总：`RESULTS.json`／`RESULTS.md`，逐条在`metrics/`，`COMPARISON.json`对照当前6309组；全部成功以`FINAL_AUDIT.json`及`HANDOFF_COMPLETE.json`为准。
- 创建本目录STOP可停止领取及后续阶段；训练STOP和当前6309评测STOP彼此独立。不要重复启动活跃queue；保持已授权暂停的Goal不变。

本组同时改变训练数据、Event读出位置、Feature分支与规划器，不属于只改一个因素的严格消融。其余历史对照与覆盖率/N/A保留；不将静音／过短音频当普通语音填零平均。
