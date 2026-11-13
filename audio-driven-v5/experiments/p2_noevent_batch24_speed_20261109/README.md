**用户已中止切换（2026-11-09）：** batch24同744样本测试通过、660状态迁移完成，但没有正式batch24更新；用户改为将node11当前完整组接回四机31卡。两份660权重保留。

# 无Event三机24卡：batch31→batch24

用户授权以提速为首要目标，允许改变batch。当前完整保存边界为旧step660；原checkpoint在 `../p2_ltx_all_lora_noevent24_after6000_20261109/step_000660.pt`。最终采用与否以 `ADOPTION_ACCEPTED.json` 为准，试验更新不进入正式训练。

## 固定项与改变项

保留最终train＋dev500h及最新caption、原始Qwen hidden、全LTX attention/FFN rank8 LoRA、两路Whisper全参数、AV＋双路Feature（无Event）、学习率、Adam、八网卡、all重计算和每500更新完整保存。node11完整组不改。

仅将剩余全局batch改为24，每卡每步1条。原已经消费的20460个(sample index, noise ordinal)不动；剩余原计划pairs按原顺序连续切24条，所有样本和noise序号保持。原计划包含的尾部padding也原样保留；不额外补齐新batch。最后19条按19归一化，5个空rank仍参加FP64 SUM，避免重复数据或尾批权重偏低。

因此，保留原一轮计划的终点从9169变为11651；达到原6000步对应样本量约在新step7558。步数不可直接与batch31消融比较；总样本量可比，但梯度更新轨迹必然改变。

## 实测与验证

`benchmark.py` 从同一660检查点开始，分别以batch31更新24次、batch24更新31次，处理同一744个连续pairs。两种设置先做最长样本真实更新和形状预热，再恢复660开始计时。比较744条的总耗时及samples/s，不把单步缩短直接等同于整体提速。记录在 `matched744/RESULTS.json`；两种batch参数不应要求相等。

CPU测试检查完整剩余序列、恢复偏移、尾批与AV/Feature归一化；3rank Gloo检查包含空rank时的FP64归约、梯度及Adam对照。正式采用前迁移只改变身份，逐值验证模型、Adam及24RNG；启动再执行两次最长样本真实更新并恢复。

## 代码与恢复

- `batch_runtime.py`、`training.py`、`canary.py`、`worker.py`：独立派生执行代码，原native与split源码不改。配置明确标注实际batch24。模型结构用规范化旧配置构建，运行spec为新配置。
- `PLAN.json`、`READY.json`：固定边界、样本预算及源码/配置绑定。
- 采用后的输出：`../p2_ltx_all_lora_noevent_batch24_20261109/`。
- `adopt.py migrate` 核验实测吞吐改善后迁移；`adopt.py launch` 从迁移完整660恢复。正式训练推进后，应使用新输出最新完整检查点恢复，不能重新执行660迁移。
- 若试验失败且尚未产生新正式更新，确认试验进程退出后，`fallback.py` 从旧完整660恢复八网卡batch31，不得误用更早224。
- 采用后的 `watch_hourly.py` 每3600秒检查新无Event输出和原node11完整组。只有在新训练实机验收后才替换旧观察进程，不修改node11训练。
