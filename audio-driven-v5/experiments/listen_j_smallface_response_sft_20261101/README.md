> **后续已按用户要求包含500h并扩展为全量，当前请使用[全量回应文本监督数据](../listen_j_full_response_sft_20261101/README.md)。本目录仅保留此前小脸子集，文件未改动。**

# 小脸候选：speaker audio＋listener首帧 → listener回应文本

已完成引用清单导出；三份文件的SHA256和行数核对通过。16条真实媒体读取通过（8条train、4条dev、4条test），5项错配／篡改拒绝检查通过；覆盖原标注裁框与最终框相同及不同的情况。未对全部813,726条解码媒体。

## 用途与范围

用于准备Qwen-Omni的回应文本监督数据：**输入是speaker音频片段和listener单张首帧，目标是已有listener回应原文**。不把listener未来视频、listener音轨、原音频转写或目标文本混入模型输入。

来源为新J数据最终画质通过、有caption的保留集。选择原始人脸短边p10<224 px，剔除500h试选已补入的21,584条。train/dev/test保持独立，训练只读取train。没有新增人脸下限；小脸仍沿用既有画质通过条件，不恢复画质排除、待核或无人脸片段。

“未入选”指未进入本轮500h试选，不能表述为因画质不合格而被丢弃。部分184–224px片段还受有声补充条件及500h容量限制影响。也未宣称与历史50h/300h清单全部不重合。

| 划分 | 片段数 | 总时长 |
|---|---:|---:|
| train | 800,579 | 1,320.42h |
| dev | 5,806 | 9.77h |
| test | 7,341 | 12.09h |
| 合计 | 813,726 | 1,342.28h |

精确数值、文件SHA256和数据范围见`summary.json`。`COMPLETE.json`仅表示引用清单导出完成；实际媒体抽样读取结果见`audit.json`，不表示已解码验证全部媒体或已启动训练。

本轮不按“必须有声”或“必须有动作”筛选，安静倾听标签仍保留。无需重新打标、运行GPU或生成LTX latent。

## 文件

- `train.jsonl`、`dev.jsonl`、`test.jsonl`：逐条数据清单，含两个输入引用及原文目标。
- `index.sqlite`：按sample_id或split查询JSONL文件名、字节位置和长度，支持随机读取。
- `crop_config.json`：读取最终修正版裁框的配置快照。每条记录绑定其规范JSON哈希。
- `example_reference.json`：一条完整引用格式示例。
- `summary.json`、`COMPLETE.json`：数量、时长、划分与导出校验。
- `audit.json`：真实媒体抽查、文件完整性、错误输入拒绝检查。

这里采用“引用清单＋按需读取”，**没有批量新增WAV或PNG**。speaker音频的实际文件位置通过源pair结果的`normalized_audio.speaker`读取，不能通过替换listener文件名猜测。listener视频仍从原TAR成员按偏移读取；只解码到片段开头附近，提取一张首帧。

## 数据格式与输入隔离

每行包含：

- `inputs.speaker_audio`：speaker参与者ID、源pair结果路径及SHA256、clip索引、源时间范围；读取的是同一pair的speaker归一化音轨，不套用listener静音掩码。
- `inputs.listener_first_frame`：listener参与者ID、原视频TAR成员及偏移、pair起点、最终修正版裁框测量记录与SHA256；输出512×512 RGB。
- `target.text`：原始listener整段回应caption，逐字保留。它是监督目标，不能作为user输入的一部分。
- `provenance`：原统计文件与字节位置、原文源文件哈希、原训练准入状态、音频状态等。仅用于核验，不应整体拼进prompt。

首帧采用与24条复核页面一致的“最接近pair起点的原始PTS帧”规则，误差不超过一个源帧周期；并非整段视频的第一帧。裁框版本为`quality-bounded-crop-v1-20261030`。旧框caption与最终裁框不同的情况保留溯源，读取结果的`audit.annotation_crop_differs`会明确标记。

speaker音频按同一源时间范围截取，保持内部静音间隔。当前原音频为48kHz单声道；读取器返回真实采样率和float32波形，后续训练processor须按模型要求显式重采样，不能仅把48kHz数组冒充其他采样率。

原caption由完整listener音视频及补充点头证据生成；这里的训练任务是依据speaker和listener初始外观预测一种回应描述，不代表首帧和speaker音频能够唯一决定真实后续动作。已有标注中的漏标、推演或声音冲突没有在本次重写。

## 读取示例

在项目根目录使用`PYTHONPATH=src`，运行下列代码：

```python
import json
import sqlite3
from pathlib import Path
from listen_response_sft import load_example

work = Path('experiments/listen_j_smallface_response_sft_20261101')
config = json.loads((work / 'crop_config.json').read_text())
conn = sqlite3.connect(f'file:{work}/index.sqlite?mode=ro', uri=True)
conn.row_factory = sqlite3.Row
location = conn.execute(
    "SELECT * FROM samples WHERE split='train' ORDER BY sid LIMIT 1"
).fetchone()
with (work / location['file']).open('rb') as f:
    f.seek(location['offset'])
    row = json.loads(f.read(location['bytes']))
example = load_example(row, config)
speaker_audio = example['inputs']['speaker_audio']       # mono numpy float32
sample_rate = example['inputs']['speaker_sample_rate']   # 当前48000
listener_image = example['inputs']['listener_first_frame']  # PIL RGB 512×512
response_text = example['target_text']                   # assistant监督目标
```

`load_example`只返回上述音频和单图作为inputs。源信息、质量测量、caption、pair音频身份或时间不匹配会报错，不静默用原框、混音或其他参与者替代。读取实际speaker音频前按来源SHA256核对；文件校验缓存绑定路径、大小、修改时间和期望哈希。

## 代码、复现及检查

- [导出入口](../../scripts/export_j_smallface_response.py)
- [按需读取器](../../src/listen_response_sft/data.py)
- [真实媒体及完整性检查](../../scripts/check_j_smallface_response.py)

```bash
python scripts/export_j_smallface_response.py \
  --output experiments/listen_j_smallface_response_sft_REPRO

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python scripts/check_j_smallface_response.py \
  --data experiments/listen_j_smallface_response_sft_REPRO
```

导出使用一个CPU进程顺序读取现有统计；真实媒体核查串行执行。未做全量视频解码。源时长可能已量化到48kHz采样点，允许与六位小数端点之差不超过一个音频采样周期，不改写实际端点。

## 与500h及评测集的关系

该train清单与当前500h试选片段ID不重叠，但可以来自同一个train会话或人物；不是相对于500h的独立泛化测试集。原dev/test按既有连通组划分保持独立，不能混入训练。

保留原`training_admitted`值，不用本次引用导出覆盖历史准入状态。本轮准备了数据读取契约，没有修改Qwen-Omni训练器、对话模板、loss或启动训练。正式训练应由训练入口按任务明确管理准入，而不是静默丢弃所有历史flag为false的行。
