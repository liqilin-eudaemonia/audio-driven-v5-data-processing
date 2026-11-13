# 当前确认方案：原生视频 8 fps＋高召回点头补充

[完整交接、prompt、参数、命令和输入格式](../../related/P2_原生视频8fps与点头补充打标交接.md)。

本目录是整理已有结果形成的 **15 段候选包样例**。不是新一轮 GPU 推理，不是全量重打标或训练标签。

- `bundles.jsonl`：每段一行；也可逐条读取 `results/<sample_id>.json`。
- 94 条整段 VLM 原文事件，独立补充 16 个点头候选；没有自动追加摇头或其他四类专用检测结果。
- `vlm_observations` 保留 82 个 video8 原始请求（含 2 个无效窗口）；`vlm_full_events` 保留整段原文；`nod_candidates` 单独保留来源和时间，尚未语义去重合成最终训练 prompt。
- `summary.json`、`audit.json`：计数与校验结果。
- `nod_replay_check/`：参数化 CPU 点头入口仅复跑 1 段，recall 完整分数与区间和旧版逐项一致；未改历史 15 段结果。
- `check_decode.log`：1 段真实源视频的 CPU 输入一致性检查；新包装只准备 video8，张量、帧证据、prompt 与旧分支一致。
- `code/`、`code_manifest.json`：此次整理的本地代码/文档快照。

新配置：`configs/listen_visual_native8_nod_v1.json`；入口：`scripts/listen_visual_native8_nod.py`。新 GPU worker 未做端到端推理验证，正式启动前需小样本运行检查。

复现已有 15 段导出（在项目根目录）：

```bash
python scripts/listen_visual_native8_nod.py export \
  --manifest experiments/listen_glare_nodding50_20261029/samples.json \
  --vlm-results experiments/listen_visual_input50_20261029/results \
  --nod-results experiments/listen_glare_nodding50_20261029/results \
  --output experiments/listen_visual_native8_nod_20261029
python scripts/check_listen_visual_native8_nod.py
```

本轮只读旧结果、新建候选包，不更新生产标签、音频、latent 缓存或现有训练清单。缺点头结果会报错；不能将其当成无点头导出。
