**2026-11-09解码环境问题已修复并实际接续：** 本轮始终只测一个全LoRA完整pipeline的step6000权重；32表示并行卡数。此前启动器把仅评分需要的vendor PYTHONPATH带入原生解码，导致PyAV导入失败、协调器退出；200条latent均已完成且保持原样。现将生成／解码PYTHONPATH清空，评分才使用vendor路径；5项回归和真实297帧512×512解码通过。已重新启动四机后处理，实际解码超过100条且继续推进。错误、旧代码、修复及实测均保存在`recovery_decode_environment/`。

**已四机32卡实际接续（2026-11-09）：** 原训练6428步完整保存并释放后，评测协调器自动扩到32卡；全部32个原生全LoRA严格加载通过，每卡1632处目标，四个实际主机核对通过。已有生成保留，固定Seamless200／SFT2752／seed137；后接就绪解码、分片评分和全200汇总。GPU实际验收见`RUNNING32_ACCEPTED.json`；下方资源冲突说明为已解决的历史阶段。

**实际资源状态：** 初次探测其余24卡空闲，准备期间完整全LoRA训练从6208恢复并占用其余31卡，每卡约99GB；因此本评测目前仅node11 GPU0实际运行，已通过1632目标严格加载和真实seed137产出检查。其他卡保持80GiB空闲显存准入，释放后自动接入。已向用户询问是否保存暂停31卡训练、评测后恢复；未停止或kill任何训练／vLLM。详见`RESOURCE_CONFLICT.json`、`GPU_VALIDATION.json`及实时`status.json`。

# 全attention／FFN LoRA完整pipeline6000：Seamless固定200，四机32卡

用户2026-11-09明确授权四机32卡推理测评。训练权重：

`/private/liqi_workspace/project_code/audio-driven-v5/experiments/p2_ltx_all_attention_ffn_lora31_20261108/step_006000.pt`

原完整权重SHA256：`3c2c78e8b1c81e3cf84efdc565b16a28565c0e923cacf8e902c3adcd6232eea6`。48层原生attention／FFN的1632处rank8 LoRA，双路Whisper全参数；训练目标AV＋0.05整段Event＋两路各0.025 Feature。本次不是无Event版本，也不是旧text-only LoRA。

## 统一评测条件

- 固定Seamless200；清单SHA256 `d4bda633eefebfd0b0a1b055684a749d86a6b28ec7c9d1a032e4a151b0468b07`。
- seed137，Euler30，video CFG3／audio CFG7，512×512、原FPS及原帧数。
- 最新Qwen2.5-Omni-3B规划SFT2752；直接复用已完成的同200条规划文本／H／Gemma、listener首帧及speaker Whisper缓存，不重新生成规划或抽缓存。
- 空Gemma负条件rank00与原8份缓存逐值一致后共用；训练H为原始未SFT，推理H为SFT2752，与原主干联合6000的评测条件相同。
- 不读取未来listener画面／GT音频／GT caption作为推理输入。GT仅用于评分。

## 来源及加速

CPU核对完整6000模型／Adam／31份RNG、282源码、LoRA范围后，导出不含Adam的模型推理版，逐张量与原权重一致。GPU worker只读mmap，前后核对文件size／mtime／ctime，严格载入原生全LoRA模型和1632处目标；不让32进程重复扫描37GB优化器文件。

四机node12／node8／node4／node11各GPU0–7；node11既有vLLM不停止，推理按剩余显存准入（80GiB）。跨卡共享样本锁与最长片段优先；每节点每轮最多启动2个新进程。GPU结束生成即可领取已就绪latent的解码（最多4路，OOM降2），哪个固定逻辑分片媒体齐全就评分（最多2路／卡，OOM降1）；慢节点资源探测异步，不阻塞其他三机。

评分沿用CPU预取2、4线程像素计算、batch64、无损GT RGB／特征缓存。提前评分清单由已验收GT绑定替换生成路径；最后运行原生全量bind逐行对照，再从完整200条分片特征汇总FID/FVD，不平均分片分数。音频技术错误和有效覆盖率按原口径报告，UTMOS无有效窗口保留N/A。

## 代码、状态与输出

- 入口：`scripts/p2_all_lora6000_seamless_eval32_20261109.py prepare|queue|worker`。
- 原生模型适配／导出：`src/p2_all_lora6000_eval_20261109.py`；源码快照`code_snapshot/`。
- `export/RECEIPT.json`：完整来源、推理文件SHA、逐张量一致性和共享盘文件绑定。
- `CPU_ACCEPTANCE.json`及`tests.log`：固定输入／负条件、32卡映射、来源绑定、缺失样本拒绝检查。4项回归通过。
- `status.json`：生成／解码／已评分分片；`RESOURCES.json`、`devices/`、`model_acceptance/`、`processes/`与`logs/`为实际执行证据。CPU准备完成不等于GPU已验收。
- 媒体：`generated/ltx_all_lora_event_feature_whisper_6000_sft2752/seamless/<id>/generated.mp4`与`listener.wav`。
- 结果：`RESULTS.json`、`metrics/`；`COMPARISON.json`对照原text LoRA联合6000同seed137结果。
- 完成以`PROGRESSIVE_BIND_AUDIT.json`、`FINAL_AUDIT.json`和`HANDOFF_COMPLETE.json`为准。根目录STOP取消本评测；守护只拥有本评测子进程，已提交输出保留，不停止其他任务。没有创建Goal或聊天定时汇报。
