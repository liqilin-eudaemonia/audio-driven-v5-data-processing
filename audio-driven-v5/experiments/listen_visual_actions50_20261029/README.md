# 允许动作名称的观察式prompt：同50条配对验证

本轮验证“允许直接可见的动作名称＋非穷尽动作提示”是否改善观察式标注。只测视觉prompt，不运行方案A整理、音频或ASR，不覆盖生产数据。已完成50/50条、586次输出；[结论与完整统计](RESULT.md)、[原文对照](REVIEW_NOTES.md)、[点头措辞逐条核对](nod_text_review.json)。新版未证明动作覆盖改善，暂不替换正式方案。

## 对照与数据

- 新50h同一批50条，直接复制上一轮samples.json；不重新抽样、不排除效果不佳样本。
- 对照：上一版observation_v1；候选：observed_actions_v2。两组都重新推理，每个窗口放入同一批次。
- Qwen3-VL-8B-Instruct；原listener裁框、512×512、8fps、JPEG85；整段＋3秒窗口／1秒步长。
- 每组293个请求，共586次输出；字段、时间原点、结构化JSON约束、temperature=0、max_tokens=4096、seed保持一致。
- 4个worker按片段时长估算工作量平衡分配；改变任务分配，不改变成员、输入或每窗配对方式。分配清单保存在assignments.json。

## 新prompt

```text
Watch this {duration}-second listener clip. List all directly visible actions and facial expressions you recognize, including sustained expressions.
Use familiar action names when supported by the visible movement: nodding, shaking or turning the head, head tilts, raising or furrowing eyebrows, gaze shifts, blinking or closing the eyes, smiling, visible laughing movements, shoulder movements, hand or arm gestures, and posture changes. These are non-exhaustive reminders, not required events. Only report what is visible within the crop.
Naming an observed action such as a nod or smile is allowed. A nod requires visible vertical head movement and return, not just a lowered head pose. If the action name is uncertain, describe the observed movement instead.
Describe the person and the visible action, without inferring thoughts, feelings, intentions, communicative meaning, or audible speech from the image.
Give each occurrence its own start time, end time (seconds from clip start), and a brief description. Represent a continuous action or state as one interval, not separate entries for successive frames. Keep genuinely separate occurrences separate.
Return JSON with an events array. Each event has start, end, modality ("visual"), and behavior (description).
```

这次同时加入动作命名许可、提示清单、持续表情及点头运动说明。测试的是整套候选prompt，不能把效果单独归因于其中一句。保持开放式description，不强行限定类别，不要求每类都出现，不加入心理含义。

对照prompt逐字复用上一轮run.py中的observation_prompt；两组完整实际prompt均随响应保存。

## 运行

在项目根目录执行，GPU与rank分别取0–3：

```bash
CUDA_VISIBLE_DEVICES=0 \
LD_PRELOAD=/private/software/conda/envs/vllm/lib/libstdc++.so.6 \
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
/private/software/conda/envs/vllm/bin/python \
experiments/listen_visual_actions50_20261029/run.py --rank 0 --shards 4 --memory .20

python experiments/listen_visual_actions50_20261029/analyze.py
python experiments/listen_visual_actions50_20261029/cross_run.py
```

vLLM 0.19.0，bfloat16，每卡20%显存配额，max_model_len=49152、max_num_seqs=2、max_num_batched_tokens=4096、eager。现有训练不停止。详细参数见protocol.json，代码快照见code_manifest.json及code/。

## 结果口径

原始输出不做语义重写、去重或时间修补；失败也保留。关键词出现次数包含否定／不确定描述，只是描述诊断，不是动作真值或召回率。没有逐帧人工GT，不能证明±0.2秒达标。示例动作变多并不自动意味着真实覆盖变好，也要留意提示清单引入的臆测。

- results/<sample_id>.json：全部响应、原prompt、媒体帧摘要、校验状态；事件时间是窗口局部时间。
- descriptions_*.json：仅有效响应的原文候选，时间已加窗口偏移，统一到片段局部时间。
- analysis.json：统计、异常、同窗输入及prompt检查。
- cross_run.json：与上一轮成员／媒体一致性及观察式对照重跑波动。
- samples.json：原媒体来源、裁框、旧整合事件及源记录摘要。
