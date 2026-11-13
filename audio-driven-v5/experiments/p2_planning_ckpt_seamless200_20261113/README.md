# 七组 Qwen：Seamless 固定诊断200

已完成：1400/1400，均非空、正常EOS、零重试，约14分钟。验收见 `FINAL_AUDIT.json`；四条文字抽查见 `QUALITATIVE_REVIEW.json`，尚无全量语义准确率排名。

用户指定 node11 GPU7。原模型、SFT500/1000/1500/2000/2500/2752，同200条真实speaker音频和listener首帧，共1400条规划。采用当前 `evaluation/SEAMLESS_CURRENT.json` 的转正版本；固定200包括48条非零listener音轨和历史选择152条，属于诊断子集，不能代表完整test的无偏表现。

仅运行Qwen规划；不运行LTX视频生成。沿用已验证的原生加载、BF16/SDPA、batch16、greedy、512 token上限（未EOS则1024重试），逐个加载并核对适配器张量。目标描述和listener后续媒体不进入输入。现有自动标注随结果并列展示，尚不提供未经核验的行为/时间准确率排名。

- 启动与进程：`LAUNCH.json`、`DEVICE.json`
- 输入/权重/协议：`protocol.json`、`samples.json`、`PREPARED.json`
- 进度：`progress_0.json`、`run.log`
- 输出：`results/<arm>/<id>/original.json`
- 完成：`COMPLETE.json`、`RESULTS.json`、`index.html`（自动汇总）
- 错误：`FAILURE.json`；排除原因后用相同命令恢复，已完成结果复用。

恢复必须在node11运行：
```bash
CUDA_VISIBLE_DEVICES=7 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false /private/liqi_workspace/project_code/Omni-LTX-stage2/.venv/bin/python -u /private/liqi_workspace/project_code/audio-driven-v5/experiments/p2_planning_ckpt_seamless200_20261113/run.py
```

完成后页面：http://localhost:8876/listen_benchmark_review/planning_ckpt_seamless200_20261113/
GPU4由GLARE使用；本任务只使用GPU7。参考方法后台队列按其原有可抢占规则让行，输出保留。
