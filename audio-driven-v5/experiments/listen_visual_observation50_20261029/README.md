# 新观察式prompt：同50条视觉打标验证

本轮仅测试新观察式视觉prompt，不执行方案A文本整理，也不运行音频／ASR。旧prompt在相同媒体上重新运行，作为实验对照。已完成50/50条、586次输出；[结论和统计](RESULT.md)及[原文例子](REVIEW_NOTES.md)已保存。新增约束减少了推演措辞，但重复区间和输出失败仍需处理。

## 成员与输入

直接复用 `../listen_visual_observable50_20261029/samples.json` 的50条随机成员，无增删或重新挑选。来源是新50h的29,975条训练子集；seed=20261029，50个不同pair，共321.777秒。

模型 `/private/ckpts/Qwen3-VL-8B-Instruct`。使用原始listener预裁框，512×512、8fps、JPEG85；整段＋3秒窗口／1秒步长，两组各293个请求。每个窗口两组共用同一份解码帧和时间戳。

## 完整新prompt

其中duration为实际窗口时长，格式与原prompt一致。

```text
Watch this {duration}-second listener clip. Describe all directly visible movements and facial-expression changes you recognize.
For each occurrence, give its start time, end time (seconds from clip start), and a brief description of what visibly changes.
Describe physical observations only, without interpreting their psychological or communicative meaning.
Represent a continuous action or state as one interval, not separate entries for successive frames. Keep genuinely separate occurrences separate.
Return JSON with an events array. Each event has start, end, modality ("visual"), and behavior (description).
```

旧prompt通过 `annotation_prompt(..., prompt_variant='backchannel_visual_simple_v2')` 原样生成。这次同时改变任务表述和持续事件划分要求，是整个候选prompt的对照；不能把收益单独归因于删除backchannel一词。

没有新增固定类别、情绪标签或主观评分字段。不额外设置system message，沿用模型chat_template。每次请求的完整prompt及原始响应均保存。

## 推理与复现

沿用上轮参数：bfloat16，temperature=0，max_tokens=4096，seed=20261029，JSON schema相同。4个worker分别位于本机GPU0–3，每个20%显存，max_model_len=49152、max_num_seqs=2、max_num_batched_tokens=4096、eager。每次generate同时输入旧／新prompt，避免单独改变生成批量；不停止已有训练。

在项目根目录执行；rank和GPU依次取0–3：

```bash
CUDA_VISIBLE_DEVICES=0 \
LD_PRELOAD=/private/software/conda/envs/vllm/lib/libstdc++.so.6 \
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
/private/software/conda/envs/vllm/bin/python \
experiments/listen_visual_observation50_20261029/run.py --rank 0 --shards 4 --memory .20

python experiments/listen_visual_observation50_20261029/analyze.py
```

## 检查口径

- 保留原始响应，不自动修复失败时间、不补填遗漏事件、不用后处理掩盖prompt问题。
- 推演关键词仅是诊断线索，不是准确率；动作词也包括否定句，不是动作召回率。
- 候选数量包括整段与重叠滑窗，不能作为独立真实事件数量。
- 同文短区间仅是疑似冗余线索；必须区别真实重复动作。
- 额外对照上轮旧prompt重跑一致性，防止把运行波动都归因于新prompt。
- 原生产标签、训练清单、缓存和默认prompt保持原状。

`results/<sample_id>.json` 包含全部窗口两组输出（含未通过校验者）；其中事件时间是窗口局部时间，需加记录的start偏移。

`descriptions_*.json` 仅收录校验通过的原文候选，时间已换算到片段局部时间。`analysis.json` 保存统计和错误清单。`samples.json`、`protocol.json`、`code_manifest.json`及`code/`保存来源和复现证据。
