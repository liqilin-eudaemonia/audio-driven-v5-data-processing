# 新50h方案A整合、条件缓存与guided八组训练

2026-10-27启动。当前真实状态：[单卡任务池](pool/state.json)、[整合/缓存接续](director/status.json)。正式训练开始后看 [四条训练链](training_queue/state.json)。这些文件是实时状态；本文不把排队任务写成已经完成。

输入为重新后处理/筛选的29,975段新50h，512×512。沿用交接原模型、prompt、硬校验、自检与3次尝试。通过原审计重放后才进入训练。用户已明确批准：**8组只使用最终成功发布的同一批样本，拒绝清单完整保留；每阶段步数为ceil(成功条数/24)**。并非强行保留29,975条，也不回填旧描述。

| 链 | 第一阶段 | 第二阶段 |
|---|---|---|
| av | 标注指导，冻结Qwen，AV | 同组S1最终权重＋Qwen LoRA，AV |
| av_event | 标注指导，冻结Qwen，AV＋0.05 Event | 同组S1最终权重＋Qwen LoRA，相同损失 |
| av_feature | 标注指导，冻结Qwen，AV＋0.1视频Feature＋0.1音频Feature | 同组S1最终权重＋Qwen LoRA，相同损失 |
| av_event_feature | 标注指导，冻结Qwen，AV＋0.05 Event＋两路各0.1 Feature | 同组S1最终权重＋Qwen LoRA，相同损失 |

所有组seed=20261023、global batch=24。S2只继承同组S1模型，新建优化器；阶段内严格恢复模型、优化器、step、随机状态。Qwen LoRA接收对应生成/辅助损失的梯度，不进行规划文本SFT。

## 数据与缓存

- [输入/模型/策略](pipeline.json)、[原输入分块清单](PREPARED.json)。`chunks/*/run/`保留原worker的protocol、源码、attempts、results、summary。
- 原脚本上下文溢出会整块退出。调度器保留已发布结果、隔离未完成输入；CPU长度判断仅帮助分组，隔离样本仍通过原worker执行。无法处理的单条保留原错误。GPU/SSH失败只重试，不当作拒绝。
- `chunks/*/audit.json`严格重放成功结果；`export.json`记录通过、拒绝及待隔离成员。
- 通过结果用原`p2_plan_cache`英文模板呈现。音频字段不改变；原单字中文按用户要求保留，不静默翻译。新事件与原标注有来源、哈希对应。
- `plans/samples/`存逐样本实际指导文本；`plans/cache/text/`为Gemma；`plans/cache/input_plan/`为输入＋同一指导文本的Qwen hidden，包含全部输入和计划位置。
- 原视频VAE、后处理音频VAE、七类事件与冻结teacher缓存复用；本次不修改canonical事件。关闭Feature的组不读取teacher。
- 最终共同成员在`view/observations.json`；完整拒绝清单`view/rejected.json`；数量、时长、预算在`view/COMPLETE.json`。文件在整合结束后发布。
- `CACHE_VALIDATED.json`记录所有选中样本的缓存身份和事件统计。新的cache writer验证张量与SHA；旧大payload在实际训练读取时校验SHA，预检查不重复通读所有大文件。

## 资源和自动接续

三机各8张H200；[节点/通信配置](cluster.json)。原四组固定文本LoRA训练优先保留，任务池按剩余显存叠加，每卡最多一个新单卡任务。共享卡原生整合worker使用memory=0.46、batch=8；独占卡使用memory=0.80、batch=16。原训练重启/扩卡时，本流程只暂停自己占用对应卡的worker，保留已有同协议产物供恢复。

每个成功块立即进入两类缓存。首次成功缓存触发`canary/s0/`真实GPU检查，与其他整合/缓存叠加。全体成员及缓存冻结后，`s0/`对最终最长媒体/长文本执行8种配置的真实forward/backward、S1→S2初始化、断点恢复和梯度路径检查。**GPU S0尚未通过时不能声称训练已经就绪或已验证。**

`TRAINING_READY.json`仅在最终缓存与GPU检查均成功后发布。随后等待旧四组训练全部完成，启动4条并行链：每条6卡（三机各2卡）、每卡microbatch=1、累积4；S1完成立即启动同组S2。某条链完成后，剩余链从完整checkpoint扩到12/24卡，累积改为2/1，global batch和样本ordinal不变。

沿用已验证的LTX全层、Qwen梯度重计算，训练allocator上限128GiB。每100step保存、阶段结束保存、TensorBoard、有限值检查继续生效。三机各有备用控制器，共享锁保证单一调度者；worker由进程组托管。失败保留日志并退避重试，源码/身份不符会阻止发布后续阶段。

## 入口、状态与恢复

- [CPU接续导演](../../scripts/p2_guided_chain_director.py)，[单卡资源池](../../scripts/p2_guided_chain_pool.py)。二者均已由三机独立后台进程运行。
- [正式训练入口](../../scripts/p2_guided_chain_train.py)，[配对训练调度](../../src/p2_guided_chain/training_queue.py)，[真实GPU S0](../../scripts/p2_guided_chain_s0.py)。
- `configs/matrix.json`和8份实际配置在共同成员冻结后发布；此前的通用模板仍位于[16组索引](../../configs/p2_matrix16/matrix.json)。
- `director/error.json`、`pool/tasks/*.json`、`pool/attempts/`保留错误与重试证据；旧错误文件可能保留，结合当前status时间判断是否已恢复。
- `training_queue/jobs.json`记录各组当前stage、配置、最终权重和GPU槽位；`parallel_runs/`记录训练、TensorBoard与拓扑迁移。

手工恢复只需在任一节点执行以下两个命令，已有控制器持锁时新进程只做备用：

```bash
/private/liqi_workspace/project_code/Omni-LTX-stage2/.venv/bin/python scripts/p2_guided_chain_pool.py --root experiments/p2_new50h_guided_chain_20261027
/private/liqi_workspace/project_code/Omni-LTX-stage2/.venv/bin/python scripts/p2_guided_chain_director.py --root experiments/p2_new50h_guided_chain_20261027
```

需要主动停整条链时创建根目录`STOP`。不要删除旧实验的STOP、改写原样本或把失败任务标记为成功。重启前查看具体错误；硬盘损坏、模型文件缺失或长期系统故障不能靠无限承诺解决。

## 验证和比较边界

本轮新增接续测试见[测试代码](../../tests/test_p2_guided_chain.py)，原整合/缓存/矩阵回归亦已运行。最终真实GPU结论以`canary/s0/COMPLETE.json`、`s0/COMPLETE.json`为准。

拒绝后实际时长可能小于50h。新8组之间可按共同成员、seed和预算比较；不能将其与原30,415段固定prompt实验直接解释成“只改prompt”的单变量比较。原始整合结果是弱标签，程序重放不等于人类确认语义正确。
