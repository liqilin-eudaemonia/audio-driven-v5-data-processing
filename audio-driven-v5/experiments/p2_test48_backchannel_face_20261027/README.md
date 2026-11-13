# 独立 test 48 条快速验证子集

48 条，合计 **285.1945 秒（4 分 45.1945 秒）**，平均 5.942 秒。全部来自既有独立 test，与 50h 训练子集在 pair、会话、参与者及连通组上无交集；没有从 train 切出测试数据。验收结果见 [validation.json](validation.json) 和 [COMPLETE.json](COMPLETE.json)。

## 筛选标准与结果

沿用 50h 的“大脸主体＋稀有类别补充”：原始人脸短边 p10 主体门槛 224 像素、有效帧比例至少 98%、无记录的原始／语义／时间标注问题、至少一个非微笑事件。同一 pair 最多 8 段，源时间不重叠，包括相反 listener 方向。

评分保持不变：`0.5×min(face_p10/384,1) + 0.3×类别数/7 + 0.2×min(封顶事件密度/12,1)`。每类最多计 3 条，微笑最多计 1 条，密度分母至少 5 秒；只用于排序，不删改原文标签。依据静音后保留的事件统计，同一事件的多个支持区间只计一次。

此次直接使用此前已完成同规格标注与后处理的 **512 条 test 候选**，210 条达到上述质量门槛。先选 43 条主体，再保证七类至少各有一条，最后填满 48 条。主体已覆盖七类，剩余 5 条仍从主体池补足，没有降低人脸门槛。48 条的小测试集不套用 50h 的每类 2,000 条补充目标。

- 原始人脸短边 p10 最小 **304.02**、中位数 **328.81** 像素；p50 中位数 **336.92** 像素。
- 35 个 pair、6 个连通组；每段候选事件中位数 12，非微笑候选中位数 5。
- 含之前 24 条测试样本，另外选入 24 条。与旧 test24 有重合，不能将两者作为相互独立的评测集合。
- 本轮筛选没有使用任何生成结果或评测得分，也没有重新运行打标模型。

| 类别 | 含该类的片段数 | 候选事件条数 |
|---|---:|---:|
| 点头 | 25 | 63 |
| 摇头 | 1 | 1 |
| 微笑 | 36 | 505 |
| 大笑／笑声 | 11 | 37 |
| 皱眉 | 22 | 107 |
| 惊讶 | 32 | 170 |
| 音频回应／发声 | 24 | 42 |

一段可以包含多类。标注为模型弱标签，候选条数不等于人工确认的真实动作次数；达标候选池只有 1 条摇头，本集保留该条。该集合按大脸和事件丰富度挑选，用于快速诊断，不代表整个 test 的自然分布，也不是人工事件真值集。

## 数据入口与读取

- [manifests/observations.json](manifests/observations.json)：配对 speaker 音频、listener 首帧、时长、test 身份。
- [manifests/targets.json](manifests/targets.json)：listener 512×512 裁剪视频、原始 listener 裁片音轨引用。
- [integrated_labels.jsonl](integrated_labels.jsonl)：后处理后的原文整合事件与时间区间，保留排除审计。
- `audio_postprocess/recipes/<sample_id>.json`：官方词／句／VAD 并集与多字中文排除规则对应的虚拟静音配方。
- [sample_ids.txt](sample_ids.txt)、[samples.csv](samples.csv)、[selection.json](selection.json)、[selection_policy.json](selection_policy.json)：冻结名单、逐条统计、排序与入选原因。
- [summary.json](summary.json)、[media_references.json](media_references.json)、[dataset_contract.json](dataset_contract.json)、[provenance.json](provenance.json)：汇总、媒体和来源身份。

`media/<sample_id>/` 提供视频、首帧、speaker.wav、listener.wav。WAV 是原音轨的裁片，**不是静音后新 WAV**。已有 24 条媒体通过目录链接复用，其余 24 条导出同规格媒体；保留原 test24 目录及原始数据，不要删除依赖。

读取静音后的 listener 参考波形：

```python
from pathlib import Path
from p2_listen.contracts import read
from listen_audio_postprocess.recipes import read_listener_audio

root = Path('/private/liqi_workspace/project_code/audio-driven-v5/experiments/p2_test48_backchannel_face_20261027')
obs = read(root / 'manifests/observations.json')
sid = obs[0]['sample_id']
recipe = read(root / 'audio_postprocess/recipes' / f'{sid}.json')
waveform, sample_rate, metadata = read_listener_audio(recipe)
```

不要绕过配方，把 `targets.json` 的原 WAV 当成处理后参考音频。`config.json` 是媒体导出配置，不是训练配置；本次没有生成训练 latent 或启动训练。`targets.admitted=true` 仅表示媒体结构验证通过，test 及弱标签身份不变。

## 验证与复现

执行逐条媒体文件哈希、源 manifest 身份、裁剪与时长、实际虚拟音频读取检查；核对 listener 裁片 WAV 与模型标注时收到的 WAV 校验值一致。复核源标签哈希、视觉事件保持原样、train/test 身份隔离和所选源区间互不重叠。代码包含三项回归检查：48 条及稀有类别覆盖、缺失类别报错、重叠源片段不重复选择。

```bash
cd /private/liqi_workspace/project_code/audio-driven-v5
/private/liqi_workspace/project_code/Omni-LTX-stage2/.venv/bin/python \
  scripts/select_processed_test_panel.py \
  --candidates experiments/p2_test24_backchannel_face_20261027 \
  --train experiments/p2_train50h_postprocessed_reselected_20261027 \
  --catalog data_corpora/seamless_listen_local_full/catalog.json \
  --output experiments/p2_test48_backchannel_face_20261027 \
  --count 48 --workers 8

python -m unittest discover -s tests -p test_processed_test_panel.py -v
```

输入和脚本哈希冻结于 provenance；完成后重跑直接返回。需要更改筛选条件时使用新输出目录，保留当前固定测试集。未生成图或网页。
