# Omni 整段自然语言打标：固定 prompt v1

2026-10-30，按用户确认的四项收敛调整，在同一批 50 条 `native_simple` 样本上完成最后一轮独立试验，并冻结 prompt 为 `native-av-natural-nod-v1`。固定版本表示后续使用这一版作为基准，不表示全部描述已通过人工质量验收。

[有声复核页面](http://localhost:8876/listen_benchmark_review/native_simple/natural_caption_nod_final/) · [最终完整 prompt](prompt.txt) · [逐样本实际请求](prompts/) · [输入与生成参数](protocol.json) · [原始结果](results/) · [已知问题前后对照](known_case_comparison.json)

## 输入和固定规则

- Qwen3-Omni 权重 `/private/ckpts/Qwen3-Omni/Instruct`；listener 512×512 原生视频 8 fps，同步净化 listener 音轨。音频在内存按已有配方置零，不裁掉时间段，不新增 WAV。
- 同一批 50 条全部重新推理，无旧描述、旧转写或旧结果输入，不复用任何旧输出。
- 点头候选沿用同片段 GLARE recall：38 条共 50 个区间，12 条列表为空。模型仅收到 `id/start/end/observation`；时间以当前 pair 开始为零。区间数不能当作点头次数。
- 根据实际输入波形计算并核验 `listener_audio_is_silent`；本批 40 条全静音、10 条非全静音。true 表示每个采样为零；false 不自动证明有发声。
- 写一段连贯英文，描述动作顺序、持续状态、同时发生和真实重复；逐个核对点头候选。只描述可见动作与实际可听声音，禁止心理、意图推演和输入元数据说明。
- 解码：temperature=0，max_tokens=4096，repetition_penalty=1.05，单段正则约束。空输出、未正常结束或长原文循环，最多一次同 prompt 重试，repetition_penalty=1.12；每次原文先落盘，不能靠截短或改写假装通过。本轮未触发重试。

## 实际结果与边界

| 项目 | 上轮独立打标 | 固定 v1 |
| --- | ---: | ---: |
| 正常结束 | 49/50 | 50/50 |
| 重复截断 | 1 | 0 |
| 有候选且正文含 nod 的片段 | 27/38 | 31/38 |
| 已知四条静音片段的虚构说话内容再次出现 | 4 | 0 |

nod 是文字检索统计，不能当成动作准确率或所有区间均被覆盖。上轮 27 条还包含重复截断样本。

明确未解决的问题：

- S1337 全静音，仍写 `while speaking`、`as he speaks`；这是静音约束仍会失效的反例。
- 7 条有候选但没有明确写 nod；其中有些描述头部上下运动，不能直接算 7 条真实漏检。S2043 未明确融入点头；S0805 虽写 nodding，仍未清楚表达两段候选在早期与片尾的位置。
- S0699 本轮只引用一次 Okay；此前用户确认需要分别保留两次，因此发声重复次数完整性仍有问题。
- 仍有 `pensive`（S5193）、`as if preparing to speak or reacting to internal stimuli`（S5268）、`as if about to speak`（S1460）。
- 页面规则提示 25 条带秒数表述、6 条带 `Assistant:` 前缀。所有原文保留，未做事后删句修饰。
- 上轮重复截断的 S1806 本轮首次生成正常，但仍有前缀和秒数表述。

因此：本轮完成运行、输入及播放一致性验收；语义质量仍需复核，不自动作为最终训练标签。静音标记有效改善四个已知案例，但并未保证杜绝幻觉。

## 验证、源码与复现

- `audit.json`：50 条身份、实际 prompt、四字段、静音标记、媒体身份、独立推理及八卡完成情况通过；`semantic_quality_verified=false` 明确不声称语义验收。
- `browser_validation.json`：有声播放、点头区间跳转正常，50 条浏览器净化 PCM 与 Python 输入哈希一致；10 条有声/40 条静音，手机宽度无溢出，无 JS 异常。
- `cpu_validation.py/json`：50 条 prompt 检查；可识别上一轮 S1806 长文本循环，保留正常的多个短词 yeah。
- `attempts/`：每次生成立即保存的原始文本及解码参数；`results/` 保存选取的原始输出，没有语义改写。
- `code/`、`code_hashes.json`：实际推理时冻结的源码。`delivery_code/` 为最终交付源码及哈希；推理结束后修复了检查列表的共享引用问题，使生成失败原因与后加的语义复核标记分开保存。没有重新推理或更改本轮原始结果；审计以立即保存的 attempt 为准。
- node4 八卡各完成 6/6/6/7/6/6/6/7 条；0–3 卡显存预算 0.55，4–7 卡 0.85，前四卡其他任务保留。每卡独立模型、单请求完整 prefill、decode-only CUDA Graph、2 路 CPU 预准备。推理耗时合计 139.18 秒，是 50 条调用之和，不是含模型加载的墙钟时间。

```bash
cd /private/liqi_workspace/project_code/audio-driven-v5
python scripts/audit_native_caption_nod_direct.py --run experiments/listen_native_caption_nod_final50_20261030
python scripts/build_listen_native_caption_review.py --run experiments/listen_native_caption_nod_final50_20261030 --output-name natural_caption_nod_final
```

后续另起试验目录使用 `scripts/prepare_native_caption_nod_direct.py --run NEW_DIR` 冻结输入及当前源码，再用 `scripts/run_listen_native_caption_pilot.py worker --run NEW_DIR --rank 0..7`，逐卡设置 CUDA_VISIBLE_DEVICES；部署环境见 protocol 与 launch 记录。当前入口用于这批 50 条，不是全量队列入口。此次未启动全量自然语言重标或训练，也未改动其他运行任务。
