# 四组Qwen×LTX：Seamless固定256条联合测评

Seed137已于2026-11-13北京时间17:50:45全部完成（1024条生成与四组评分汇总）。[Seed238接续与双seed平均结果](../p2_joint_qwen_ltx_test256_two_seeds_20261113/README.md)。仅node8（172.29.104.237）GPU0–7、node12（172.29.252.37）GPU0–7，共16卡。node11保持暂停。

## 范围与进度

四组：Qwen8256＋LTX6800、Qwen7500＋LTX6800、Qwen2752＋LTX6000、原生Qwen＋LTX6800。各256条，共1024条；复用原64条×4组＝256条，仅新增768条生成。

- [实时状态](status.json) · [结果表](RESULTS.md) · [完整指标与有效覆盖](RESULTS.json)
- [冻结实验清单](PLAN.json) · [复用明细](REUSE.json) · [启动代码](run.py)
- [256条选择与分布](../seamless_test_tuning256_20261113/README.md) · [逐条起止时间](../seamless_test_tuning256_20261113/selection.csv)
- [CPU接入检查](CPU_ACCEPTANCE.json) · [首次现场验收](LAUNCH_ACCEPTANCE.json)

原64条完整保留，新增192条只按当前完整test6624的来源场景、处理后音轨全零／非零和时长分层选取，不读取模型结果。每条原长视频只取1个pair，优先来源组均衡。共196条naturalistic、60条improvised；音轨非零76条、全零180条；平均5.990秒（全集6.004秒）。非零不等于人工确认有语音。该集合用于test权重调参，不称独立dev。

seed137、Euler30、视频CFG3／音频CFG7、512²、25fps保持。每个Qwen使用自己的规划文本、hidden和Gemma条件。2752已有全256条条件；另外三组各复用64条，补提192条，共576份新条件。speaker Whisper缓存全复用。

## 调度与评分复用

每组4个缓存分片，各含16条旧条件和48条新增来源，原生规划batch16、Gemma batch8不变。先并行12个实际补提worker，其余4卡可立即运行2752生成。

生成总并发上限14，通常每组3～4个worker，另有2卡接续解码、评分；收尾按未完成组合数提高每组worker上限，避免只剩一个组合时卡被固定的每组2worker限制闲置。始终使用原生样本锁。一个GPU同时只放本控制器的一个任务，外部占用卡跳过。检查资源每10秒，状态每15秒更新。

每组评分共10个分片：前4片精确复用原64条的逐条分数、音频结果和FID／FVD特征，后6片各32条新增样本；全部结束后用原生合并器重算256条全局FID／FVD，绝不平均64／192两份分数。保留原始contract与旧文件路径，逐条GT行SHA、模型／规划／seed／几何身份已核验。旧结果不覆盖。

规划仍按512 token、1024 token重试；达到上限后保留真实文本、标记截断，不改写、不伪造EOS、不删除难例。上一轮7500的一条截断样本继续保留。图像、音频有效覆盖与表情／ΔResp的缺失情况如实报告。

## 产物与恢复

- 各组视频／音频：`combinations/qwen<步数>_ltx<步数>/generated/`。
- 条件：`conditioning/qwen<步数>/conditions/`。
- 控制器PID：CONTROLLER.json；主日志：controller.log；各worker日志：logs/。
- 最终完成收据：COMPLETE.json。未出现该收据前不视为完整结束。

本轮控制器在node12后台独立运行。暂停本轮可创建本目录STOP文件；控制器关闭自己的worker连接并退出，保留已提交产物。恢复前确认旧控制器退出、两机仍获授权，移除STOP，再在node12执行：

```bash
cd /private/liqi_workspace/project_code/audio-driven-v5
/private/liqi_workspace/project_code/Omni-LTX-stage2/.venv/bin/python -u experiments/p2_joint_qwen_ltx_test256_16gpu_20261113/run.py queue
```

flock防止两个控制器重复运行；缓存、生成、评分均复用完成收据。代码及协议指纹见PREPARED.json，运行时不修改冻结源码。恢复沿用首次ACTIVATED.json的设备UUID核验。
