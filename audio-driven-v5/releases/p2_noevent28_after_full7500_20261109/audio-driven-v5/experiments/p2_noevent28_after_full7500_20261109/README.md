# 完整组7500保存后，去Event组28卡续训

用户授权完整组精确7500完整保存、校验、退出后，开启去Event全LoRA组的四机28卡训练。保持node12/node8/node4各GPU0–7，node11仅GPU0–3，GPU4–7预留。实际切换阶段见 `HANDOFF_STATE.json`，错误则记入 `HANDOFF_ERROR.json`并阻止自动接续。

去Event组从旧24卡输出的完整step660恢复模型／Adam，而非从完整组7500初始化。仍是LTX全部attention/FFN rank8 LoRA＋两路全参数speaker Whisper＋AV＋双路各0.025 Feature；无Event头、标签读取、优化器组和损失。原始Qwen H、最新完整caption、最终train＋dev500h及全部缓存保持。

## Batch与提速证据

Global batch28，每卡每步1条。此前同一无Event全LoRA、同一660初始化及相同744条pairs的24卡实测：batch31用286.6836秒，batch24用247.7927秒，样本吞吐2.5952→3.0025条／秒（+15.69%），同样本耗时−13.57%。这是每卡一条比不均匀31/24分配更快的直接证据，不证明任意batch中全局最优，也不是28卡同样本AB实测。记录见 `BATCH_SPEED_EVIDENCE.json` 和 `../p2_noevent_batch24_speed_20261109/matched744/RESULTS.json`。

已消费20460个原pairs保持；余下263779个pairs按28连续切分，所有sample与noise ordinal原序保留，不新补样，最后19条按19归一化、9个空rank仍参与FP64 SUM。原一轮总样本量不增加，结束步数调整为10081。若要对齐旧batch31的6000预算，应按样本量比较，不能直接比较新6000更新。

## 保存与切换

完整组每500步保存且先发布step7500指标。只读门槛控制器等待该步，在原生checkpoint保存期间写本任务所属STOP；必须完整7500 SHA／模型／Adam／28RNG核验、旧launcher与28个worker全部退出，才可以启动新组。暂停旧小时观察器，node11预留4–7的任务不受影响。所选28卡连续两次空闲检查；遇其他任务不终止它，等待后仍占用则阻止接续。

新组step660状态逐值迁移、28份rank RNG从父rank0初始化；原24份在父checkpoint保留，每样本ordinal种子保持。独立代码／配置／checkpoint身份，不修改父源码和旧STOP。启动先做最长11.95秒两次真实更新，再严格恢复660；至少8次正式更新检查损失、梯度、样本计数、卡位和速度，通过后切只读每小时监控。新组每500步完整保存，首个定期保存点1000。没有安排下一组训练。

GPU优化继续采用已验证的all重计算、GPU Adam、CPU prefetch2、512MiB streaming FP64和四机单mlx5_0；node11当前108GiB进程上限、其他134GiB。最长样本实机结果与正式步速需等启动验证，不把CPU检查当GPU验收。

## 文件

`manager.py`只用于本次660迁移与显式launch；已有正式进度后不可重跑660迁移，须使用本输出最新完整checkpoint继续。`runtime28.py/training.py/worker.py/canary.py`为独立执行版本；`test_plan.py`验证完整剩余计划、尾批与7500停止门槛；`tail_check.py`用3个真实Gloo进程检查空rank、AV/Feature均值、梯度和Adam。`handoff.py`是实际切换控制器；`accept.py/watch_hourly.py`为后续验收和小时监控。
