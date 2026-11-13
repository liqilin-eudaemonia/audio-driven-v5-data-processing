# 50条视觉prompt配对验证（2026-10-29）

## 目的与状态

验证仅在原视觉打标prompt末尾增加“只描述直接可见动作／表情，禁止心理、意图和交际意义推演”，能否减少原始输出中的推演，同时检查结构、时间和描述变化。已完成50/50条、586次配对输出。结论与完整统计见[RESULT.md](RESULT.md)和[analysis.json](analysis.json)：推演措辞减少但未消除，重复与动作描述覆盖仍有问题。

这是独立实验，不覆盖全量原始打标、整合标签、训练数据或缓存。不运行音频分支，也不能据此评价音频标注。

## 成员与输入

- 来源：`../p2_train50h_postprocessed_reselected_20261027/`，新50h，29,975条。
- 对 sample_id 排序后使用 Python random.Random(20261029).sample(..., 50)，不按现有描述好坏挑样本。
- `samples.json` 保存50个sample_id、原始媒体来源与裁框、历史整合事件、来源行SHA256。
- 按生产 full_raw.prepare 重读原视频，采用已确认listener预裁框缩放512×512，8fps及原JPEG85输入像素。
- 整段＋3秒窗口、1秒步长；相同窗口下新旧prompt共用相同帧及时间戳。与旧全量相同输入逻辑，但这轮重新运行两组；不能视为旧全量响应的逐位重放。

## 模型与prompt

模型 `/private/ckpts/Qwen3-VL-8B-Instruct`，bfloat16；temperature=0，max_tokens=4096，结构化JSON约束。两组均不额外设置system message，使用模型chat_template、enable_thinking=False。

旧prompt（duration替换成当前片段／窗口秒数）：

```text
Watch this {duration}-second listener clip. List all visible backchannel behaviors you recognize. Give each occurrence its own start time, end time (seconds from clip start), and a brief description.
Return JSON with an events array. Each event has start, end, modality ("visual"), and behavior (description).
```

新prompt完整保留以上两行，仅追加：

```text
Describe only directly visible actions and expressions. Do not infer psychological states, intentions, motivations, or communicative meanings.
```

独立入口：`run.py`。此次没有切换生产prompt默认值。

## 运行与复核

`protocol.json` 保存参数。4个独立worker分别处理 samples[rank::4]，每个占一张GPU的20%显存配额；配对batch=2，max_model_len=49152，max_num_batched_tokens=4096，eager。现有训练不停止。与历史全量运行相比batch/调度不同，新旧两组在本轮完全一致。

```bash
# 以下rank/GPU取0、1、2、3；在项目根目录执行
CUDA_VISIBLE_DEVICES=0 \
LD_PRELOAD=/private/software/conda/envs/vllm/lib/libstdc++.so.6 \
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
/private/software/conda/envs/vllm/bin/python \
experiments/listen_visual_observable50_20261029/run.py --rank 0 --shards 4 --memory .20

python experiments/listen_visual_observable50_20261029/analyze.py
```

`results/<sample_id>.json` 保存两组所有窗口的prompt、完整原始响应、媒体帧SHA256、输出token数、结束原因、解析结果和时间校验状态。`partial/` 是中间产物；以 `results/` 为完整样本。时间错误不自动修补，不因效果不好替换样本。

`analyze.py` 校验完整成员、新旧prompt只相差追加句、同窗媒体一致、双组全窗口齐全；统计只纳入完整通过结构和区间校验的响应，异常单列。`analysis.json` 及 `descriptions_*.json` 保留详细结果。

统计解释：

- 推演关键词是待检查线索，并非人工准确率；既可能漏检，也可能误命中表情词。
- 事件数是整段与重叠窗口产生的原始候选数，尚未跨窗去重，不能当作独立真实事件数。
- 动作词出现或消失仅说明描述变化，不等于准确召回或漏检。
- 没有逐帧人工真值，本实验不能证明时间误差≤0.2秒。
- 本轮不对新输出再做语义清洗；直接检查prompt的效果。
