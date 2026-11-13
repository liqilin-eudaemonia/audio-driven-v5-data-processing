# 原生 8 fps＋点头：代码轻量整理（15 段）

**当前只完成 CPU 前处理，没有使用 LLM 整理结果。** 用户提示本机不宜叠加显存后，立即停止本次 30B 初始化进程及其子进程，确认全部退出。没有完成的模型响应，未继续启动 GPU；`status.json` 记录停止状态，`run.log` 保留尝试记录。现有其他任务未停止。

[查看整理前后 HTML](http://localhost:8876/listen_benchmark_review/prompt_versions_20261029/glare_nodding/semantic_light/)。在 Cursor 转发 8876；左侧视频及同步音频，右侧所有原始有效观测与代码初步整理结果。可以筛选 VLM、点头或完全重复组，也可下载本段 JSON。

## 结果与处理边界

15 段，365 条有效观测（VLM 整段＋滑窗＋16 个点头候选）→359 条，只折叠 6 条完全重复。2 个无效窗口的原响应完整保留在各段 `preprocessing.invalid_responses`，不送入事件整理。

本步仅做：

1. VLM 局部时间加窗口起点，统一为片段时间；保留原始局部时间及响应摘要。
2. 明确 `vlm_video8`／`glare_nod_recall` 来源，不把检测器的 `Head nodding` 冒充 VLM 原句。
3. 只有来源类型、全文、起止时间完全一致的记录才折叠。比较键以 1ns 精度消除浮点加法尾差，但保留第一条原始数值及所有来源原时间，不合并近邻事件、不取外包区间。
4. 每组保留全部 `original_observations`；不删除推演、不改写主语、不建立固定行为映射、不吞掉点头。

**相近描述与跨窗近似重复仍在，这是未完成语义阶段的中间结果，不是最终生成指导或训练标签。** 365、359 都不是去重后的真实动作次数。

## 文件与代码

- `inputs/<sample_id>.json`：每段初步整理记录及完整来源。
- `jobs.jsonl`：原计划供 LLM 读取的任务，当前不运行。
- `preprocessing.json`：逐段计数、原始包哈希、原计划使用的 Plan A prompt 与审核 prompt。
- `audit_light_review.json`：15 段原文/计数/来源展示，重复筛选，媒体 HTTP206，同步音频、慢放跳转，JSON 下载和移动端校验。
- `scripts/prepare_native8_nod_semantic_pilot.py`：CPU 对齐及精确去重。
- `scripts/build_native8_nod_light_review.py`、`scripts/native8_nod_light_review.html`：页面。
- `scripts/check_native8_nod_light_review.py`：校验。

复现 CPU 阶段（不会加载语言模型）：

```bash
cd /private/liqi_workspace/project_code/audio-driven-v5
python scripts/prepare_native8_nod_semantic_pilot.py
python scripts/build_native8_nod_light_review.py
FONTCONFIG_FILE=/tmp/native-fps-fonts.conf /tmp/native-fps-review-qa/bin/python scripts/check_native8_nod_light_review.py
```

现有 `run_listener_semantic_prompt.py` 为原计划小样本试跑新增可选 tensor parallel 和 prefill token 上限参数，默认不启动；本轮没有验证新 GPU 路径的完整推理。后续若继续 LLM 阶段，应先安排合适资源，不能自动恢复本机叠加任务。
