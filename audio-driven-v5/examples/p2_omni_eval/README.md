# OmniResponse 指标输入

使用 [pairs.example.json](pairs.example.json)，路径相对清单解析。每条记录由 `sample_id + sample_seed` 唯一标识；各 seed 的 sample_id 集合必须相同，同一 sample_id 的 GT 不能随 seed 改变。

默认指标为 FVD-32、UTMOSv2、SyncNet。视频必须为真实 25 FPS、PTS 从 0 连续，GT/生成尺寸与帧数相同。独立 listener 音频也从 0 开始；容许原生片段非整数帧时长导致的小于一帧时差，不接受明显时长错配。不填帧、不用 `-shortest` 掩盖截断。不要把 speaker 音频填进 `gt_audio` 或 `generated_audio`。

生成失败仍保留记录：`generated_video` / `generated_audio` 可以缺省，但必须给出 `generation_error`；该行会进入失败分母。缺失文件、重复 ID、种子集合不一致在准备阶段直接拒绝。

启用 `--metrics fd` 时，每行还需：

```json
{
  "gt_features": "features/dev_0001_gt.npy",
  "generated_features": "features/dev_0001_seed137.npy",
  "feature_schema": "具体提取器版本、权重哈希、列顺序和归一化方法"
}
```

数组为有限数值 `T × C`，两侧必须使用同一提取器及语义；允许有效帧数不同，报告分别列出帧数。FD 汇总时按帧加权，**不是 Inception FID**。参考目录没有通用 3DMM 提取器，本接口不会擅自拿 Connector 特征替代它。

启用 `--metrics text` 时，每行还需：

```json
{
  "generated_text": "实际生成音频的转写或实际生成文本",
  "gt_text": "listener GT 转写",
  "text_provenance": "转写来源、模型版本或人工标注说明"
}
```

文本可为 null 或空串，这些行会标为不适用并计入覆盖率，完整集合分数为 null；不会偷偷启动 ASR，也不会把 speaker 文本当作 listener 生成文本。METEOR、ROUGE-L、BERTScore 是参考相似性，不能代替回应恰当性；Distinct 按整个明确支持的文本集合计算。

从已完成的 P2/GLARE generation work 导出：

```bash
P2_PY=/private/liqi_workspace/project_code/Omni-LTX-stage2/.venv/bin/python
"$P2_PY" scripts/p2_omni_eval.py export-generation \
  --work /absolute/path/e1_generation \
  --gt-targets /absolute/path/targets.json \
  --output /absolute/path/e1_omni_pairs.json
```

`targets.json` 为数组，每行至少有 `sample_id` 和独立 listener 的 `target_audio`（也接受 `gt_audio`）；可以直接使用原生目标清单，若有 media_hashes.target_audio 会核对。生成检查点、完成标记、视频和 listener.wav 都由原 GLARE 导出器核验，失败行保留。
