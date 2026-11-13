# Qwen × LTX 固定 test64：node8＋node12 联合测评

2026-11-13 北京时间16:30:56已完成：16组×64条全部生成、评分与汇总结束（复用320条、新增704条），本轮worker已全部退出。图像／分布指标与音频均覆盖64条；部分表情／ΔResp仍按原生有效覆盖报告。仅使用 node8（172.29.104.237）GPU0–7、node12（172.29.252.37）GPU0–7，共16卡。node11所有本项目测评保持暂停，不参与本轮。

## 范围与结果

Qwen2752／7500／8256 × LTX6000／6300／6500／6700／6800，加原生Qwen × LTX6800控制组，共16组，每组固定64条。复用此前SFT2752的320条生成及评分，新增704条生成。沿用seed137、Euler30、CFG视频3／音频7、512²和25fps。

- [实时状态](status.json)：生成、latent、已汇总组数与实际工作进程。
- [结果表](RESULTS.md) · [完整指标与有效覆盖](RESULTS.json)。进行中只展示已汇总的组。
- [冻结清单与组合](PLAN.json) · [两机配置](cluster.json) · [运行入口](run.py)。
- 每组媒体及分片评分：`combinations/qwen<步数>_ltx<步数>/`；旧2752结果位置见PLAN.json，保留原文件。

每个Qwen分别生成自己的规划、hidden states和Gemma条件；共192份新条件。共用已经验证的speaker Whisper、首帧和负条件。先12个缓存分片（各16条，保持原生规划batch16），随后动态分配16卡生成、解码、评分；单组合最多2个生成进程，原生样本锁避免重复。生成期间最多2路解码、2路评分，收尾按可用卡扩展。评分沿用原生公式，各分片特征合并后计算全局FID／FVD。

此64条由用户指定从test中抽取，用于权重调参；与完整test成绩分开。保留原有指标缺失和有效覆盖说明，不将缺失值记为0。

## 启停与恢复

控制器在node12后台运行，PID见CONTROLLER.json，日志见controller.log；子进程日志位于logs/。进程失败有界重试，完成收据和原生锁支持断点复用。停止只影响本轮：在本目录创建STOP；控制器退出时关闭自己的远程worker连接。紧急停止可核对CONTROLLER.json中的PID与命令后向控制器发送SIGTERM。

确认旧控制器已退出且仍授权这两机后，移除STOP，在node12用以下命令恢复。不要另启第二个控制器，不清理完成产物。

```bash
cd /private/liqi_workspace/project_code/audio-driven-v5
/private/liqi_workspace/project_code/Omni-LTX-stage2/.venv/bin/python -u experiments/p2_joint_qwen_ltx_test64_16gpu_20261113/run.py queue
```

首次资源UUID已记录ACTIVATED.json。恢复时会跳过外部占用卡，控制器flock防止重复启动。PREPARED.json冻结运行源码、数据和协议；改变配置前先停止，不覆盖旧结果身份。

[node11暂停与恢复记录](NODE11_PAUSE_README.md)。

## 规划截断记录（本轮运行修复）

同一条 `naturalistic_V03_S0975_I00000271_local_b2d6b0b27b8c` 首次在7500和8256出现重复生成。8256在断点重试后正常结束；7500连续三次仍达512／1024 token上限，触发旧调度器失败退出。已保留首次失败文件与日志，恢复本轮任务。

本轮对长度耗尽的统一处理：仍使用相同生成参数和原512／1024 token重试，随后保留模型真实输出，标记`finish_reason=length`、`planner_truncated=true`，用该文本提取自己的hidden和Gemma继续生成。不添加伪造EOS、不清理重复、不换成标注或其他权重的规划，也不剔除样本。它属于模型规划失败的证据，不能报告为“全部正常EOS”。原生提取函数仅在实验目录复制并改这项异常处理，共享运行库未改；详见cache_recorded_truncation.py和TRUNCATION_POLICY.json。已正常完成的条件不重算，重试batch组成变化保留在attempts记录中。
