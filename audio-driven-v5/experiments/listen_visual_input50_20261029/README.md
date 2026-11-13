# C prompt：多图与原生视频输入对照

**已完成50/50条、879次请求并通过配对审计。** [结果与判断](RESULT.md) · [原文对照](REVIEW_NOTES.md) · [带声音HTML](http://localhost:8876/listen_benchmark_review/prompt_versions_20261029/?comparison=inputs)。全部整段输出通过结构校验；短窗共30次无效输出原样保留。没有替换训练标签。

固定已确认的观察式 C prompt，对同一新 50h 子集的 50 条样本比较三种视觉输入。未经人工复核不覆盖生产标签。原始视频来自 listener 本人的归档成员，按已确认的裁框裁成 512×512。

| 组 | 输入 | 目的 |
|---|---|---|
| image8 | 8fps 独立图片＋逐帧文字时间戳 | 本轮同期重跑基线 |
| video8 | 相同 8fps 帧，以原生 video tensor＋metadata 输入 | 隔离图片／视频输入方式的影响 |
| video16 | 16fps 原生视频输入，包含上述全部8fps帧 | 检查更密集采样的影响 |

## 公平性与限制

- 三组相同 C prompt、整段＋3秒／1秒步长、temperature=0、seed=20261029、JSON schema、max_tokens=4096；每个窗口的三组放进同一次 `LLM.generate`。
- 50 条成员、原始裁框、切分起点完全一致；每条 8fps JPEG 哈希须与上一轮原始输入一致。为了只比较表示方式，三个分支沿用相同 JPEG85 传输。输入来自原视频一次解码，**不是把视频重新编码后再抽帧**。
- 原生视频是官方支持的预解码帧张量＋视频metadata，走 `multi_modal_data.video`，不是把多张独立图片改个名字。关闭 `do_sample_frames` 和 `do_resize`；分辨率、帧数保持显式规格。
- 原生视频处理器按两帧组织视觉编码，奇数帧末尾会按官方逻辑补齐一次；此行为、帧数、token数均记录。视频8fps与多图8fps的视觉token数量不同，这是输入方式本身的区别。
- 原生 metadata 使用源帧率与整数帧位置。非帧对齐的切分起点会产生最多半个源帧周期的时间取整误差，逐帧保存；未隐藏、没有修改模型输出时间。该误差不是模型标注误差的估计。
- 此外，当前 vLLM 原生视频实现以每两帧的平均时刻作为组时间戳，并用一位小数格式化；多图基线文字时间戳保留六位小数。前述 metadata 误差不包含这一步格式化，也不是事件边界准确率。输入表示的时间粒度差异是本轮比较的一部分，不能把少量小数或短区间减少直接判为更准。
- 模型不输入音轨。网页里的 listener／speaker 试听沿用人工复核功能。
- 不把条目数、关键词数当作动作召回率；否定／不确定动作描述保留，不能算确认检出。没有人工边界真值，不能声称达到±0.2秒。

## 资源与续跑

用户要求减小单机影响后，最终使用 node17／node12／node4，各3个进程、GPU 0–2，每进程显存配额20%，CPU线程2。每台模型加载串行，每台源视频读取解码最多1路。总9路推理、最多3路源视频解码。已有其他任务没有停止。

`launch.json`、`concurrency.json` 保存最终分配。此前8路单机尝试遇到 vLLM 默认 encoder cache 16,384 tokens 上限；修正视觉预算为 `MM_SIZE` 后通过长片段验证。该调整只提高允许的缓存容量，不缩小512裁框或抽掉帧。早期日志和短片段试跑保留，不混作正式完成结果。

`partial/<sample_id>.json` 每完成一个窗口的三组即原子保存；重启只续跑缺失窗口。`results/` 只保存整条完成的结果，解析／时间不合法的模型响应也原样记录，不自动修补。任务重新分配时只停止本实验进程，保留检查点。

尾段运行中，node12 三路进程数分钟未更新检查点；其余两节点已完成分片，因此只停止本实验的 node12 进程，按原 rank 将未完成样本移到 node17 两路、node4 一路续跑。`launch_recovery.json`、`node12_recovery.json` 保存迁移记录。未改成员、prompt、模型参数或完成窗口结果。

```bash
# 在对应节点运行；rank 按 launch.json，GPU 按其 gpu 字段。
CUDA_VISIBLE_DEVICES=0 \
LD_PRELOAD=/private/software/conda/envs/vllm/lib/libstdc++.so.6 \
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
/private/software/conda/envs/vllm/bin/python \
experiments/listen_visual_input50_20261029/run.py --rank 0 --memory .20

python experiments/listen_visual_input50_20261029/analyze.py
python scripts/build_listen_visual_prompt_versions_review.py
/tmp/p2-review-qa/bin/python scripts/check_listen_visual_prompt_versions_review.py
```

不要在有worker运行时重新分配 `assignments.json`。正常续跑无需 `--init`。

## 输出

- `protocol.json`、`samples.json`、`assignments.json`：冻结实验规格与样本分配。
- `results/`：原文、局部时间、prompt、完整帧证据、原生视频metadata和token量。
- `analysis.json`：结构通过、短区间、重复、宽区间、推演词线索；含整段与所有窗口两种统计。
- `nod_mentions.json`：全部含点头关键词的原文，保留否定线索供复核。
- `COMPLETE.json`：仅50条／879次请求齐全且配对审计通过后生成。
- `code_manifest.json` 与 `code/`：冻结代码快照；运行调整历史见日志。
- 全部完成后接入 [原有带声音审核页](http://localhost:8876/listen_benchmark_review/prompt_versions_20261029/?comparison=inputs)，选择“C：多图8fps vs 视频8fps vs 视频16fps”。

后续：[8／16 fps关键分歧复核、随机一致项抽查与独立测速](native_fps_decision/README.md)。此前统计不能代替真实动作准确性评价；这一轮复用原始结果，不修改训练标签。
