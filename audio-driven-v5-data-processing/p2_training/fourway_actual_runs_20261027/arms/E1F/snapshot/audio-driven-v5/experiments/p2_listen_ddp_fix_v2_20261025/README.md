# P2 确定性反向与 FP64 梯度累加复测

2026-10-25；`172.29.104.220` 的物理 GPU 4/5/6，叠加已有任务。此目录独立保存第二轮修复的真实 GPU S0，正式效果训练未启动。

- 数值策略：`p2-deterministic-training-v2`；BF16 主前向、FP32 事件头、严格确定性反向、FP64 microbatch 梯度累加及跨卡 SUM。
- 50 条真实弱标签；事件 1、可用背景弱 0、不确定区域 mask；6 个有效类别，摇头无正例继续屏蔽。
- seed 20261027、global batch 8；单卡与双卡 E0/E1 各 7 步；E1 5→7 恢复；6→7 两次重放；新检查点 source-only 独立生成。
- 最终参数阈值仍为 `2e-5`。进程退出成功不等于数值验收成功，以严格汇总为准。

入口与证据：

- `run_gpu_checks.py`：按依赖调度全部 GPU 检查；读取 `job_status.json` 查看运行状态。已有结果目录不要重复启动。
- `unit_tests.json`：105 项 CPU 回归及代码指纹。
- `cpu_ddp.json`：小模型两进程与单进程、E0/E1 各 3 步一致性。
- `preflight.json`：缓存及源码身份；`protected_before.json`：旧输入/历史产物保护哈希。
- `runs/s0_20261027/`、`ddp/runs/s0_20261027/`：训练日志、身份及检查点。
- `ddp_comparison.json`：单/双卡两组严格对照。
- `replay.json`：恢复状态、forward 与更新重放。
- `generated_source_only/`：生成媒体、源输入隔离及实际媒体解码验证。
- `summary.json`：仅在全部严格检查通过后产生。

```bash
cd /private/liqi_workspace/project_code/audio-driven-v5
/private/liqi_workspace/project_code/Omni-LTX-stage2/.venv/bin/python \
  scripts/check_p2_listen_gpu_pair.py \
  --single experiments/p2_listen_ddp_fix_v2_20261025 \
  --distributed experiments/p2_listen_ddp_fix_v2_20261025/ddp \
  --seed 20261027 --steps 7 \
  --output experiments/p2_listen_ddp_fix_v2_20261025/ddp_comparison.json
/private/liqi_workspace/project_code/Omni-LTX-stage2/.venv/bin/python \
  experiments/p2_listen_ddp_fix_v2_20261025/validate_results.py
```

根因诊断、第一轮仅启用确定性后的残余失败及最终结论见[修复报告](../../related/P2_单卡双卡数值一致性修复与验证.md)。旧标签、旧数值策略检查点与旧失败记录保留；严格续训不会把旧策略检查点当作本策略检查点。
