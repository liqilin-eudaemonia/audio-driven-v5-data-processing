# Qwen＋Connector＋全LoRA LTX：28卡短程收益试验

**当前采用最新版评测v3（用户2026-11-12明确要求）。** `LATEST_EVALUATION_ACTIVE.json`与`latest_scoring_20261112_0726/MANIFEST.json`绑定当前工作区最新版源码、配置及Seamless读取代码；`LATEST_WORKSPACE_MATCH.json`核对12个文件均与最新工作区一致。基线和100步统一重算所有指标与事件／规划标签，128条生成视频原样复用，旧评分归档在各阶段`previous_scoring_*`。`latest_rescore_then_trial.py`完成两组最新评分后再接续原100→300短训；不重复训练或生成。状态看`TRIAL_STATUS.json`、日志`latest_rescore_controller.log`。

2026-11-12，用户要求实际验证。node12／node8／node4各GPU0–7，node11仅GPU0–3，保留其GPU4–7。原完整全LoRA6000＋规划SFT2752初始化，最终500h及最新caption；AV=1、Event=0.05、视觉和音频Feature各0.025，规划CE=0.0001，Qwen LR1e-6，生成端原LR的1/3，预热20步。

## 实测与状态

- `run/CALIBRATION.json`：16条训练输入全局向量SUM校准通过；规划／生成梯度范数比0.15021，余弦−0.00153。此为起点诊断，不是效果提升或最优系数证据。
- `run/LONGEST_SAMPLE_CHECK.json`：11.95秒样本两次真实更新后完整回滚；峰值114.924 GiB，在134 GiB进程上限内。
- `run/step_000000.pt`完整保存，目前尚无短训收益结论。当前状态看`TRIAL_STATUS.json`、`run/status.json`、`trial.log`。

## 固定比较

固定独立64dev、seed137、30采样步、video CFG3／audio CFG7。与生成训练及规划训练排除样本和媒体交集。每阶段重新用当前Qwen生成规划、hidden和Gemma；禁止跨已更新Qwen复用这些条件。

`trial.py`自动接续step0评测 → 完整恢复到100 → 同集评测安全门槛 → 通过后恢复到300 → 同集评测。基线只评测，不重训；不自动进入1000步。原生策略在`run/policy.json`，是工程筛查阈值，不是统计显著性。

`runtime28.py`仅适配v2的batch／rank及原样本-noise前缀调度，原模型和损失不改。`eval28.py`复用原生v2推理／解码、OmniResponse画质和音频评分以及既有28卡事件检测；逐样本锁让快卡领取剩余推理任务，模型单独导出避免各卡反复读取Adam，指标按原生公式汇总。实际GPU绑定逐卡核验，旧事件配置中的GPU7仅作历史协议标识，不占用node11 GPU7。

`evaluate_overlap.py`将独立规划文本judge与事件CPU准备重叠；GPU阶段切换等待本轮模型显存释放，避免两个judge短暂重叠。规划judge将“未提及”错误写成absent且没有原文引用时，按已有unknown原则屏蔽该类别，保留完整raw输出，不阻断视频评分或改造为负例；审计见各阶段`PLAN_EVIDENCE_AUDIT.json`。基线12条已有raw输出已按同一规则直接复用。适配修订及未改写的已完成64条规划／媒体／画质证据保存在`EVAL_CODE_HISTORY.json`。

100步已实际完成并完整保存。后续评测入口采用原生完整检查点的硬链接＋mmap读取，复用保存程序刚计算的全文件SHA及完整标记，避免再次扫描36GB及另复制12GB；mmap不访问Adam张量。真实100步检查点与原模型导出的全部5313个模型张量逐值相等，见`ZERO_COPY_ACCEPTED.json`；不改变模型、损失、采样或指标。测试硬链接已清理，后续实际评测硬链接不额外复制检查点存储。

早期接入的Python3.13／3.10 PyAV路径混用已修复。曾使用原始九文件评分版本保护旧身份；用户要求使用最新评测后，该旧评分已归档，不作为当前结论。训练检查点仍保留原训练源码身份（这些事件评分文件不参与模型训练），评分实际加载最新v3私有快照、最新策略及最新Seamless reader，两阶段使用同一最新版本。`isolate_events.py`保留训练身份兼容，`latest_scoring.py`负责实际新版评分；不回写其他任务文件，不修改已有模型、Adam、RNG或校准报告。每轮实际评分源码在`EVAL_CODE.json`与原生`events/detector_contract.json`另行绑定。

## 产物与恢复

- 训练：`run/metrics.jsonl`、`run/tensorboard`；完整检查点含模型、Adam及28份RNG。
- 生成与评分：`run/evaluation/step_000000`、`step_000100`、`step_000300`。
- 完成后：`FINAL_COMPARISON.json`；若100步停止，则保留相应决策和检查点。
- 控制器：`TRIAL_LAUNCHER.json`。重新执行`trial.py`前确认旧PID已退出、锁已释放；已完成阶段按身份校验复用。
- `run/STOP`阻止新训练，并让正在训练在更新边界完整保存退出。

效果需综合画质、误发声、规划能力、GT类别一致性及计划执行诊断。单seed、64条和300步只用于短程适配判断，不能证明全数据最优或长期收益。
