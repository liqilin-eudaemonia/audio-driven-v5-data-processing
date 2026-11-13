# 完整模型loss权重搜索：四机28卡顺序训练

2026-11-10用户要求按已记录方案实现并逐组接续训练。本目录保存独立配置、队列和证据；历史训练源码保持原样。

- 数据：最终train＋dev 284218段／500.001259h、最新caption／数值Event标签及既有五项缓存。
- 模型：LTX全部原生attention／FFN LoRA＋Whisper双路全参数attention＋AV／整段Event／双路Feature。
- 资源：node12／node8／node4各GPU0–7，node11仅GPU0–3；batch28，每rank每步1条。node11 GPU4–7保留。
- 加速：继承已验证单网卡、CPU预取2、512MiB streaming FP64、GPU Adam、all重计算。为batch28独立生成window448的共同duration计划，保持每例noise ordinal确定，完整一轮10151步；各组相同计划。
- 第一轮顺序：baseline → event001 → event020 → event100 → feature00025 → feature010 → feature050。各组从同一预训练初始化和seed开始，不从前组的500步继续学习。晋级只恢复该组自己的checkpoint。
- 阶段：500 → 1500 → 3000 → 6000；完整一轮预算不随短阶段缩短。每500步保存，阶段停点使用原生stop_after。

## 实际状态与验收

`HANDOFF_CONTROLLER.json`记录切换进程身份，`HANDOFF_STATE.json`／`HANDOFF_ACCEPTED.json`记录旧去Event组6000保存退出及28张训练卡释放。

`STATE.json`记录当前候选与阶段；`runs/<candidate>/status.json`和rank_logs记录真实训练，`STAGE_<step>.json`仅在完整checkpoint校验后写入。不要依据静态配置判断已开训。

`PREFLIGHT_ACCEPTED.json`包含32条真实缓存回读、最终样本数及当前源码指纹。`TESTS_ACCEPTED.json`／`regression.log`包含17项原生全LoRA与搜索回归；检查权重到梯度、样本／noise完整性、严格恢复、参数覆盖及冻结范围。大模型真实28卡最长样本两次更新与恢复将在每组首次启动时执行，结果见该组`LONGEST_SAMPLE_CHECK.json`；CPU检查不代替GPU验收。

## 切换与启动

切换程序观察现有去Event组到6000，写该组STOP，等待完整保存并通过NVIDIA只读检查确认28卡无计算进程后启动搜索。不会清除用户STOP或终止预留卡任务。若启动切换时已超过6000，会完整保存实际停止步数并如实记录。

共同Python：`/private/liqi_workspace/project_code/Omni-LTX-stage2/.venv/bin/python`；运行时需`PYTHONPATH=src:scripts`并在项目根目录执行。

- 准备：`python -m p2_loss_search28.queue prepare`（已完成，不应随意覆盖不可变配置）。
- 缓存预检：`python -m p2_loss_search28.queue preflight`。
- 预约切换：`python -m p2_loss_search28.handoff`。
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

当前统一测评集用于选参后视为调参验证集；独立最终保留集与源视频／对话重叠审计、主指标和退化约束仍需落实。生成测评接入及自动选择尚未完成，不能声称已找到最优系数。组合／局部细化候选需独立冻结，当前七组之外不擅自创建运行。

## 停止

写本目录`STOP`：切换等待或组间退出；已在运行的候选需同时写该候选输出目录`STOP`，训练会完整保存后退出。候选STOP不会被队列删除。停止预约时先核验HANDOFF_CONTROLLER身份，避免在旧组退出和新组启动之间出现竞态。
