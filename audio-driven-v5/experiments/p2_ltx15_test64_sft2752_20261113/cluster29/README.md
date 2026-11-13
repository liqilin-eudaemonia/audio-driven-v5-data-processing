**已结束：北京时间2026-11-13 14:11:47。** 实际三机21卡完成15×64＝960条生成及评分汇总，node4未接入。像素／FID／FVD／音频覆盖完整，部分表情／反应指标有效帧不足，详见[结果](../RESULTS.md)与[最终验收](FINAL_AUDIT.json)。

# 四机29卡扩容

node12/node8/node4各GPU0–7，node11 GPU0–4；node11 GPU5–7排除。2026-11-13用户已确认卡释放，交接已完成，三机21卡队列已启动（控制器见CONTROLLER.json）。保留交接时128条视频／129条latent，节点4暂缓。node4原地址与备用入口均无法建立SSH，先启动可达三机21卡；运行中的资源检查会在node4恢复且确认8卡空闲后加入。node11 GPU5–7始终排除。

同15权重×固定test64＋SFT2752，共960条。原始推理代码、采样与协议保持，扩容实际设备另记ACTIVATED.json。已完成条件、latent、视频及评分全复用；每权重最多两个生成worker，原生文件锁共享领取；释放的卡接续解码和评分。每节点每轮最多启动两项，减少大模型同时读取；不导出额外模型副本。

6项CPU测试通过，包括卡号和占用检查、负载分配、交接标记归属、原生4进程领取与崩溃恢复、冻结输入校验。15份完整SHA通过。实际29卡吞吐待释放后验收。一次只读检查中node4 SSH超时，其余3机连通但卡仍被占用；启动前重新检查。

## 启动和恢复

在node11、确认卡释放后执行：

```bash
cd /private/liqi_workspace/project_code/audio-driven-v5
nohup /private/liqi_workspace/project_code/Omni-LTX-stage2/.venv/bin/python -u experiments/p2_ltx15_test64_sft2752_20261113/cluster29.py activate --released --allow-unavailable > experiments/p2_ltx15_test64_sft2752_20261113/cluster29/controller.log 2>&1 < /dev/null &
```

启动先核查29卡，再冻结旧控制器，让当前样本完成后交接。10分钟无法排空则回退旧队列；不会杀其他任务。占用冲突会阻止移交；显式allow-unavailable允许仅跳过SSH不可达节点，其余节点先启动。

ACTIVATED.json生成后恢复可将上述`activate --released`替换为`queue`，但需确认旧cluster控制器退出、资源仍获授权。通过原子完成收据跳过已完成样本；不得与原run.py控制器同时运行。暂停创建本目录STOP，退出时关闭自有远端worker；恢复前移除本任务STOP。未完成样本重新领取。

- 调度：`../cluster29.py`；配置：`cluster.json`；冻结校验：`PREPARED.json`
- CPU测试：`../test_cluster29.py`
- 状态：`status.json`和父目录`status.json`
- 日志／进程／卡身份：`logs/`、`processes/`、`devices/`
- 交接记录：`HANDOFF.json`；卡UUID：`ACTIVATED.json`
- 媒体和分权重评分：父目录`step_*/`
- 汇总及初筛：父目录`RESULTS.md`、`RESULTS.json`、`SCREENING.json`
