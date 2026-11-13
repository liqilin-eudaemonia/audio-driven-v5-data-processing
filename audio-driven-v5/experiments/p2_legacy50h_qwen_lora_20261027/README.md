# 原50h四组权重接续 Qwen LoRA（2026-10-27）

> **当前执行方式已更新：四组各6卡并行，共三机24卡。** 每卡累积4个样本，global batch仍为24；某组完成后自动将资源扩给未完成组。E0完整保留至第105步后继续，其他三组从原权重启动。以[当前四组实时状态](parallel_queue/state.json)及[吞吐实测、正确性与恢复说明](performance/README.md)为准。下方串行调度说明保留作历史记录。

用户已确认：分别从原四组第1,268步权重接续，原始30,415段50h、seed=20261023、global batch=24，每组再训练1,268步。每组使用node17/node12/node4共24张H200；顺序E0→E1→E0F→E1F，三机冗余控制器自动接续。

实时情况以[队列状态](queue/state.json)、各组`runs/<组名>/status.json`和`metrics.jsonl`为准。启动不代表训练完成。新阶段检查点从0重新编号，第1,268步对应累计2,536次更新。

| 组 | 生成链路目标 | 原权重目录（项目根目录下，均step_001268.pt） | 新配置 |
|---|---|---|---|
| E0 | AV | experiments/p2_preflight_fix_20261026/production/runs/20261023/E0 | [配置](configs/E0.json) |
| E1 | AV＋0.05 Event | experiments/p2_e1_train50h_lora_tb_20261026/runs/20261023/E1 | [配置](configs/E1.json) |
| E0F | AV＋0.1视频Feature＋0.1音频Feature | experiments/p2_feature01_train50h_20261026/production/runs/20261023/E0 | [配置](configs/E0F.json) |
| E1F | AV＋0.05 Event＋0.1视频Feature＋0.1音频Feature | experiments/p2_feature01_train50h_20261026/production/runs/20261023/E1 | [配置](configs/E1F.json) |

## 数据和参数边界

使用原始30,415段，保持原始音频latent、原事件标签和原Feature teacher；不是后处理重选的29,975段。四组逐一核对原run的有序样本、目标、标签摘要与缓存身份，512×512规格不变。复用已验证的原类别统计，不重读所有大体积latent。训练逐样本验证缓存哈希与标签；不加载旧input hidden。

固定Gemma文本不变。Qwen使用实际speaker音频＋listener首帧，在线forward后由Connector读取hidden；不拼接规划，不做规划生成SFT。Qwen text transformer q/k/v/o投影增加rank=8、alpha=8的LoRA，学习率1e-5；Qwen主干、视觉与音频编码器冻结。在线hidden保留梯度，接收各组对应AV／Event／Feature损失。

保留各组原Connector、LTX text-attention LoRA、事件头、可选Feature decoder的全部张量；显式键名迁移严格检查形状和覆盖，不随机重置已有模块。新Qwen LoRA的B矩阵为零。第二阶段新建AdamW优化器，50步warmup；随后本阶段断点恢复同时恢复优化器、24 rank RNG和步数。Connector LR=3e-5，Feature LR=3e-5，事件头LR=1e-4，LTX LoRA LR=1e-5。

不加Event的组仍训练用于诊断的事件probe，但输入detach，Event梯度不进入Connector/Qwen/LTX；因此其生成链路仅由表中目标更新。Feature关闭组不建decoder、不读teacher、无对应优化器组。

## 队列、保存和恢复

- [三机配置](cluster.json)：172.29.15.91、172.29.252.11、172.29.225.51，每机GPU0–7。单卡microbatch=1，全局batch=24。
- [控制器](../../scripts/p2_legacy_qwen_queue.py)：三机各有独立后台控制器，通过共享文件锁只允许一个运行；控制器退出时guard清理自身worker，备用控制器接续。
- 每100步及阶段结束保存权重、优化器和RNG；首步前保存step_000000。完整checkpoint同时有SHA256完成标记，忽略未完成／损坏checkpoint并选择最近有效版本恢复。
- 首组实际训练到第2步后主动保存退出，随后自动严格恢复，验证24卡恢复路径。此过程不增加预算、不重复训练样本位置。`queue/resume_verified.json`记录阶段，实际恢复成功看续跑日志与后续步数。
- 失败时停止该次训练全部worker，重新检查显存，从最近有效checkpoint重试；每组每控制器最多12次，连续故障写入blocked状态。20分钟不出步触发整组重试；资源不足时等待，不终止其他项目进程。
- 为与现有推理任务共享显存，训练allocator上限64GiB，启动要求每卡至少67GiB空闲；LTX和Qwen梯度检查点开启。卡被占用时利用率和步速会受影响。
- TensorBoard位于`runs/<组名>/tensorboard`：AV总损失及视频/音频分项、Event及七类分项、Feature两路、加权总损失、各优化器组梯度范数/LR、步时与显存。
- [冻结源码/配置哈希](queue/frozen.json)、`source_snapshot/`保留本次本地源码副本。尚未将本次新增代码推送GitHub。
- `queue/STOP`会停止队列并终止该队列的worker；中途停止最多丢失最近一次保存后的进度。恢复需移除STOP，再启动控制器。

手工启动控制器（当前已有后台实例，不要重复操作）：

```bash
/private/liqi_workspace/project_code/Omni-LTX-stage2/.venv/bin/python \
  scripts/p2_legacy_qwen_queue.py \
  --root /private/liqi_workspace/project_code/audio-driven-v5/experiments/p2_legacy50h_qwen_lora_20261027
```

手工单组严格恢复入口为`p2_legacy_qwen_lora.py train --config ... --resume ...`，分布式必须沿用24卡。新checkpoint属于matrix16格式，推理用`p2_matrix16.inference.restore(..., backend=p2_legacy_qwen_lora.backend)`恢复，再用同模块`conditions`在线计算包含已训练Qwen LoRA的条件；不能复用旧冻结Qwen hidden评测新权重。

## 验证与解释边界

- 30项迁移/矩阵测试通过；3项队列选择完整checkpoint及24卡配置测试通过。
- [真实最长片段GPU验证](longest_gpu_smoke.json)：原数据11.95秒、E1F原权重，完整AV/Event/Feature反传和优化器更新成功；Qwen三路梯度非零，峰值reserved显存62.82GiB。单卡验证不等于24卡正式训练验收。
- 各组数据准入报告在`runs/<组名>/preflight.json`，原源码审计、数据身份、统计均留存。
- 这轮比较是“各自训练一轮后，再开放Qwen LoRA训练一轮”。对比上一轮权重时同时改变了训练预算和Qwen可训练边界，不能单独归因于LoRA；若要证明LoRA的独立贡献，需要相同步数继续冻结Qwen的对照。

## 正式启动验收

四组原30,415段数据准入全部通过；首组24卡训练已从第2步检查点严格恢复并继续更新。启动验收时至少完成第30步，140个Qwen LoRA B矩阵均已更新，checkpoint包含24 rank RNG和全部优化器状态，TensorBoard有32项指标。详细记录见[startup_verified.json](startup_verified.json)和[恢复验证](queue/resume_verified.json)。GPU占用快照见[gpu_startup_inventory.json](gpu_startup_inventory.json)；卡利用率随计算／同步阶段变化，并非每一秒都100%。实时步数以队列状态为准。
