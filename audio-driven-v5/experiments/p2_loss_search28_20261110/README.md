**已安排28卡空闲后自动接续（2026-11-10）：** 每30秒检查固定28卡，全部连续空闲超过5分钟后，仅恢复下一组feature00025的step484至500并完整保存停止；node11 GPU7事件评分继续，其他任务不停止。新写STOP优先，查询失败重置计时，不重训已完成组合。[轮询规则、状态与停止方式](idle_resume/README.md)。

**首组自动切换已实际验收：** baseline完整保存500步并校验模型／Adam／28份RNG；event001通过最长样本测试，4809个初始模型张量与baseline逐值一致、Adam为空，已进入正式更新。非权重配置及数据/noise计划一致，实际目标权重核验通过。[首组切换验收](FIRST_CANDIDATE_TRANSITION_ACCEPTED.json)。

**2026-11-10提前切换：** 用户改为5800步保存后启动搜索；请求发出时已5817，实际完整保存5818步并校验，28卡已释放。首轮启动发现文件路径入口的`queue.py`遮蔽标准库问题，已在任何正式更新前修复；新增真实入口回归，23项训练检查及32条缓存回读通过。七组队列重新启动，baseline已完成28卡最长样本两次真实更新与恢复检查，并进入正式训练（最近核验step404）；后续以动态状态文件为准。[入口修复](ENTRYPOINT_REVIEW_ACCEPTED.json) · [5818步保存与释放](HANDOFF_ACCEPTED.json)。

# 完整模型loss权重搜索：四机28卡顺序训练

2026-11-10用户要求按已记录方案实现并逐组接续训练。本目录保存独立配置、队列和证据；历史训练源码保持原样。

- 数据：最终train＋dev 284218段／500.001259h、最新caption／数值Event标签及既有五项缓存。
- 模型：LTX全部原生attention／FFN LoRA＋Whisper双路全参数attention＋AV／整段Event／双路Feature。
- 资源：node12／node8／node4各GPU0–7，node11仅GPU0–3；batch28，每rank每步1条。node11 GPU4–7保留。
- 加速：继承已验证单网卡、CPU预取2、512MiB streaming FP64、GPU Adam、all重计算。为batch28独立生成window448的共同duration计划，保持每例noise ordinal确定，完整一轮10151步；各组相同计划。
- 第一轮顺序：baseline → event001 → event020 → event100 → feature00025 → feature010 → feature050。各组从同一预训练初始化和seed开始，不从前组的500步继续学习。晋级只恢复该组自己的checkpoint。
- 阶段：500 → 1500 → 3000 → 6000；完整一轮预算不随短阶段缩短。每500步保存，阶段停点使用原生stop_after。

## 实际状态与验收

`EARLY_HANDOFF_CONTROLLER.json`记录当前提前切换／训练队列进程身份，`HANDOFF_STATE.json`／`HANDOFF_ACCEPTED.json`记录旧去Event组5818保存退出及28张训练卡释放。原`HANDOFF_CONTROLLER.json`属于已退出的6000步等待程序，不再作为活跃进程入口。

`STATE.json`记录当前候选与阶段；`runs/<candidate>/status.json`和rank_logs记录真实训练，`STAGE_<step>.json`仅在完整checkpoint校验后写入。不要依据静态配置判断已开训。

`PREFLIGHT_ACCEPTED.json`包含32条真实缓存回读、最终样本数及当前源码指纹。`TESTS_ACCEPTED.json`／`entry_review_tests.log`包含23项原生全LoRA与搜索回归；检查权重到梯度、样本／noise完整性、严格恢复、参数覆盖、冻结范围及文件入口导入。大模型真实28卡最长样本两次更新与恢复在每组首次启动时执行，结果见该组`LONGEST_SAMPLE_CHECK.json`；CPU检查不代替GPU验收。

## 切换与启动

当前使用`early_handoff.py`响应用户提前切换，等待5818完整保存并通过NVIDIA只读检查确认28卡无计算进程后启动搜索。不会清除用户STOP或终止预留卡任务。SSH只读检查遇到瞬时断连会重试；不会因观察超时而猜测训练已退出、重启训练进程。

共同Python：`/private/liqi_workspace/project_code/Omni-LTX-stage2/.venv/bin/python`；运行时需`PYTHONPATH=src:scripts`并在项目根目录执行。

- 准备：`python -m p2_loss_search28.queue prepare`（已完成，不应随意覆盖不可变配置）。
- 缓存预检：`python -m p2_loss_search28.queue preflight`。
- 本次提前切换：`python experiments/p2_loss_search28_20261110/early_handoff.py`（已启动，勿重复执行）。
- 已释放资源后的队列接续：`python -m p2_loss_search28.queue run`。

队列使用文件锁；每组启动前复核28卡空闲，不与前组重叠。故障时记录退出并保留checkpoint，不猜测SSH失联意味着训练已停止，不盲目重启。重启队列会验证并恢复本组最近完整checkpoint；禁止手改run身份或删除旧checkpoint。

## 后续晋级与测评

第一轮七组500完成后，状态为awaiting_generation_review。后续需要真实生成评价证据，不按加权训练loss自动排名。晋级文件格式：

```json
{
  "target": 1500,
  "basis": "generated_av_quality",
  "candidates": ["baseline", "event020"],
  "evidence": ["/absolute/path/to/generation_review.json"]
}
```

写入`promotion_1500.json`，同理为3000／6000。所列证据路径必须存在。这只是输入格式，不代表具体候选已获晋级。

生成评分和晋级控制见[独立执行目录](../p2_loss_search28_evaluation_20261110/README.md)。`stage_driver.py`先等待全部七组500完成；`search_controller.py`在评分完成后使用冻结规则决定晋级，再调用本队列严格恢复各自checkpoint。控制器已实现并通过CPU顺序／停止／来源校验测试；实际GPU评分与跨阶段接续尚待运行验收，不能声称已找到最优系数。

统一200视为调参验证集。固定64初筛，后续200；证据不足保留候选，不强制3／2组，额外更新预算写入决策。独立最终保留集、增加生成种子、与已有最好模型比较仍待完成。组合／局部细化候选需独立冻结，当前七组之外不擅自创建运行。

## 停止

按用户此前要求，每小时执行一次只读监测；记录在`hourly_monitor/LATEST.json`，检查控制器真实PID／启动时间、训练步时／有限值、四机所选28卡资源，不杀进程、不自动猜测故障重启。首次50个正式训练步的数值及各组梯度检查见`runs/baseline/FIRST50_ACCEPTED.json`。

写本目录`STOP`：切换等待或组间退出；已在运行的候选需同时写该候选输出目录`STOP`，训练会完整保存后退出。候选STOP不会被队列删除。停止预约时先核验EARLY_HANDOFF_CONTROLLER.json／CONTROLLER.json中的PID和启动时间，避免在旧组退出和新组启动之间出现竞态；旧HANDOFF_CONTROLLER.json不再代表活跃程序。
