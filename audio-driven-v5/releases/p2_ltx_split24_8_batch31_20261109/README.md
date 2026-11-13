**24＋8卡实际接续验收通过（2026-11-09）：** 无Event全LoRA＋Whisper＋Feature从单卡step110完整状态接续node12／node8／node4共24卡，验收到180步，最近20步14.15秒／步；完整Event组精确6000完整保存后接续node11全部8卡，验收到6004，首4步53.39–103.72秒、均值76.11秒，尚不作为长期稳定均速。两组batch31、样本/noise顺序、模型／Adam／step、最终500h及最新caption保持；各完成两次最长11.95秒真实更新并恢复验收前状态，峰值分别129.55／94.62GiB，实际卡位及loss／梯度通过。node11的SSH握手失败已单独重试恢复，24卡组未重启，vLLM保留；只读巡检每5分钟。

# 全LoRA两组训练拆分：无Event24卡＋完整组8卡

用户已确认：两组保留global batch31，完整组step6000完整保存后再切换。node8／node12／node4各8卡运行AV＋Whisper＋双路Feature；node11 GPU0–7运行AV＋Whisper＋双路Feature＋整段Event。vLLM不停止。两组继续各自模型、Adam和step，不重新初始化。

## 执行方式

保持最终284218段／500.001259小时train＋dev、最新完整caption、原始Qwen H、损失权重、所有attention／FFN rank8 LoRA、双路Whisper全参数、all重计算、学习率、每500步保存及原9169步预算。

每步仍处理原31个(sample index,noise ordinal)配对，连续分区为24卡的7×2＋17×1及8卡的7×4＋1×3。AV和Feature每条除31，Event按全局有效标签计数，指标也按样本计权。每条使用原ordinal独立设种子。浮点执行分组改变，不承诺完整后续轨迹逐bit一致。

node11使用CPU FP64累积缓冲、512MiB有界GPU通信缓冲和GPU Adam；其余3机沿用原生GPU FP64累积／流式归约。旧282／315源码保持不变，训练循环最小差异记录在TRAINING_DIFF.json，新运行身份加入runtime／worker／两份循环的哈希。

## 验证

13项CPU测试通过；真实8／24进程Gloo多桶归约、梯度／Adam精确对照通过；node11实际8GPU NCCL小规模FP64累积及Adam对照通过。两组各32条真实缓存回读、全部9169步样本／noise计划逐条核对通过。以上不替代完整LTX的最长样本GPU检查。

目标启动先各做2次最长11.95秒样本、合计batch31的真实反传／Adam更新，随后从迁移检查点严格恢复模型、Adam和RNG，再继续正式训练。正式3步以上、loss公式／有效梯度／实际卡位／速度均通过后发布SPLIT_ACCEPTED.json。发布前只代表已准备／已排队。

## 控制和恢复

control.py run控制器等待完整组5960步，先请求旧单卡无Event在当前更新边界保存退出并迁移；完整组由原独立gate精确6000保存停止。确认旧训练全部退出及其他任务符合现有vLLM并存约定后，先启动无Event24卡，同时迁移完整组6000，随后启动完整组8卡。

状态：STATE.json、controller.log、CONTROLLER.json；CPU准备CODE_READY.json，排队ARMED_ACCEPTED.json，实际完成SPLIT_ACCEPTED.json。任一组失败时不杀另一组或vLLM；异常写ERROR_<pid>.json。发生部分迁移后禁止盲目重跑整个控制器，应查看目标MIGRATION.json和launcher/rank日志后独立恢复。WORK/STOP取消等待，但不停止已启动训练。原完整checkpoint与原STOP均保留。

- 无Event目标：../p2_ltx_all_lora_noevent24_after6000_20261109/
- 完整组目标：../p2_ltx_all_lora_full8_after6000_20261109/
- 源码备份：../../releases/p2_ltx_split24_8_batch31_20261109/

用户要求降低巡检频率后，watch5m.py每300秒只读检查进度、最近损失／速度和实际卡位，记录WATCH5M_LATEST.json／watch5m.jsonl；不重启、停止或修改任何训练。6000精确保存门与切换阶段的同步等待不降频，以免越过边界。

## 实际接续结果

SPLIT_ACCEPTED.json及FINAL_ACCEPTED.json已发布。无Event24卡从110继续、完整组8卡从6000继续；模型与Adam逐张量一致。ERROR_927994.json保留为首次SSH握手失败记录，SSH_FAILURE_RECOVERED.json标明已恢复。原控制器中止后使用resume_acceptance.py独立重试完整组，未重迁移或重启24卡组；验收控制器已正常退出，两个独立launcher继续持有训练，watch5m.py仍运行。
