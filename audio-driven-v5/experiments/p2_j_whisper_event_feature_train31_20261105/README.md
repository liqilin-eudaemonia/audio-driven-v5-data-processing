# 原始H＋AV＋Event＋Feature＋Whisper：四机31卡正式训练

2026-11-06用户调整：第8,000步完整保存后停止训练，转入评测。独立`stop_at_8000.py`已启动，在看到第8,000步更新日志后写入本目录`STOP`；训练会完成原有的第8,000步保存，再正常退出，控制器见STOP后不会恢复。状态见`STOP_AT_8000_ARMED.json`、`STOP_AT_8000_TRIGGERED.json`和`STOP_AT_8000_COMPLETE.json`（后两项按实际进度生成）。原9,215步预算和检查点身份配置保留，停止于8,000步不代表完整epoch结束。建议先评测已有6,000步与8,000步，同一固定Seamless200／ResponseNet随机200及历史采样协议，保留31卡与历史batch差异说明。

用户授权四机31卡联合训练，留一张卡跑对比方法。**留出node8（172.29.104.237）的物理GPU7**，训练只使用该节点0–6卡；其余node17／node12／node4各用0–7卡，共31卡。启动器明确设置每个子进程的全局rank，不使用假设每机等卡数的静态torchrun启动方式。

## 实验条件

- 使用旧500h／285,638条train，复用原始未SFT Qwen H、完整指导文本的Gemma、最终裁框／净化listener AV、teacher、speaker Whisper和19类片段Event标签。
- 损失保持 `AV + 0.05 Event + 0.025 video Feature + 0.025 audio Feature`。
- 冻结Qwen、Gemma、teacher、Whisper encoder和LTX主干，训练Connector／Event heads／Feature decoders／LTX文本LoRA／Whisper独立音视频attention及adapter五组参数。
- 沿用实测加速：全层重算、GPU驻留Adam、512MiB streaming FP64 SUM、后台CPU预取2批和每卡一次一条样本。
- **有效batch31**，完整一轮**9,215步**，每500步保存完整模型／Adam／31份rank RNG和最终步。31卡不让某一个rank每步额外累积第二条样本。
- 从与32卡实测相同的预训练权重和seed20261023**重新开始正式训练**，旧18步保留为测试记录；没有迁移或混入那18步。初始模型逐值对照见 `INITIALIZATION_ACCEPTANCE.json`。
- duration496＝31×16，保留前两步随机顺序后按时长分组。所有285,638个真实样本与噪声ordinal一一对应且恰好出现一次，尾批补齐27条；原32卡版本补齐26条。

**比较口径：** 数据、标签、缓存、结构、损失、学习率和初始权重不变；global batch32→31、跨卡分组与梯度聚合发生变化，不能称作优化轨迹完全相同。原32卡6000步为192,000次样本曝光，对应31卡约6194步（192,014次）。完整一轮的真实样本数量一致。6194是等样本量参考，不代表已额外配置该步保存；当前保存间隔仍是500。

## 路径与状态

- [正式配置](config.json) · [物理GPU与全局rank](PLAN_ACCEPTANCE.json) · [集群配置](cluster.json) · [预留卡记录](RESERVED_GPU.json)
- [队列状态](queue/state.json) · [实时训练状态](status.json) · [损失与梯度](metrics.jsonl) · [TensorBoard](tensorboard/)
- [启动验收](STARTUP_ACCEPTANCE.json) · [初始权重一致性](INITIALIZATION_ACCEPTANCE.json) · [最长样本显存检查](LONGEST_SAMPLE_CHECK.json)
- [当前GPU资源](queue/resources.json) · [控制器日志](queue/controller.log) · `queue/attempt_*/ranks/rankXX.log`
- [独立启动入口](../../scripts/p2_whisper_event_feature_train31.py) · [8项启动器回归](../../tests/test_whisper_event_feature_train31.py)
- [32卡加速实测](../p2_j_whisper_event_feature_20261105/OPTIMIZATION_32GPU.md) · [模型结构与缓存交接](../../related/P2_AV_Event_Feature_Whisper联合实验代码交接_20261105.md)

已经验收过的285,638条全量缓存收据与统计复用，复用证明写入 `preflight.json`。数据身份相同；另独立重建并完整校验31卡采样计划、真实成员与噪声ordinal。没有修改现有 `src/p2_whisper_event_feature/`、旧32卡配置、测试权重或缓存；31卡适配仅在独立入口中安装受身份摘要保护的运行时适配。

## 故障恢复与停止

控制器最多自动尝试5次；重启时核对最新完整检查点SHA并恢复模型、Adam、rank RNG及原计划，不把不完整临时文件当检查点。后台检查有限损失、联合权重公式、五组梯度和停滞情况；不会停止其他任务，也不会领取预留GPU7。

在本目录创建 `STOP`，或在 `queue/` 创建 `STOP`，会请求当前完整更新后保存并退出。若控制器自行退出，应先检查日志和最近检查点；确认没有存活同批worker、STOP已按意愿解除后，执行：

```bash
../Omni-LTX-stage2/.venv/bin/python scripts/p2_whisper_event_feature_train31.py run \
  --output experiments/p2_j_whisper_event_feature_train31_20261105
```

从项目根目录运行上述命令；若已有控制器，它会因独占锁而拒绝重复启动。

## 后续推理读取

31卡检查点仍采用同一个联合模型结构，源码身份额外包含本独立入口。加载时先激活私有LTX并安装适配，再调用原推理接口：

```python
from p2_whisper_av.bootstrap import activate_ltx
activate_ltx()
from p2_listen.contracts import read
from p2_whisper_event_feature_train31 import install
install(read("experiments/p2_j_whisper_event_feature_train31_20261105/config.json"))
from p2_whisper_event_feature.inference import load_model
model, strategy, sampler = load_model(checkpoint_path, device="cuda:0")
```

需把项目 `src/` 和 `scripts/` 加入Python路径。推理不读取真实listener teacher或Event标签。
