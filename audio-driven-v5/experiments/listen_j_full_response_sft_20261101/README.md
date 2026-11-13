# 全量Qwen-Omni回应文本监督数据

**test 大脸子集（2026-11-02）：** [test_face_p10_ge224.jsonl](test_face_p10_ge224.jsonl)，仅从下述最终 test 中筛选原始人脸短边 `p10≥224px`，共 **4,943 条／8.420423 小时**。[派生清单核验](test_face_p10_ge224.summary.json)。逐行原样保留，ID 集与原索引筛选结果完全一致；不重新切分、打标或复制媒体，沿用本页读取方式及 crop_config.json。原 train/dev/test 清单和 index.sqlite 不变，原索引仍指向全量清单。

已完成全量导出与验收：1,107,323个唯一ID与最终保留池完全一致，0遗漏、0多余、0源元数据不一致；285,638条500h成员标记逐条核对通过。三份清单哈希／行数、16条真实媒体读取及5项错配拒绝检查通过。未逐条解码全部媒体，未启动训练。

## 当前范围

按用户最新要求，将当前500h试选与此前小脸数据合并，覆盖**新J数据最终画质通过且有标注的全部保留集**，取消本次回应文本监督任务的人脸尺寸门槛。原画质排除、待核、无人脸数据不重新纳入。

| 划分 | 片段数 | 总时长 |
|---|---:|---:|
| train | 1,086,217 | 1,820.42h |
| dev | 8,822 | 15.32h |
| test | 12,284 | 20.51h |
| 合计 | 1,107,323 | 1,856.25h |

训练只使用train；dev/test单独保存，不混入训练。这套全量清单是用户当前要使用的回应文本监督入口；此前小脸清单保持原样作为历史子集。

**当前train/dev文本筛选入口（前缀修复后）：** [train 690,337条／1,155.44h](../listen_j_train_dev_response_prefixfix_20261102/current/train.jsonl)、[dev 5,238条／9.10h](../listen_j_train_dev_response_prefixfix_20261102/current/dev.jsonl)，配套同目录`index.sqlite`和`crop_config.json`。在此前直接通过＋有声捞回集合上，新增4,091条，并清理此前已保留7,921条的开头前缀。仅前缀删除使用`original_caption_strip_chat_prefix_v1`；沿用当前读取器。原始全量清单及源标注未修改。[处理记录与剩余原因](../listen_j_full_response_filter_20261101/README.md#traindev前缀修复与当前入口2026-11-02)。

**当前test文本筛选入口：** 初筛后已清除144条开头聊天前缀并按原规则复判，新增通过87条。最新通过**6,624条／11.04717h**，使用[当前test.jsonl](../listen_j_test_response_prefixfix_20261102/current/test.jsonl)及同目录`index.sqlite`、`crop_config.json`。其余5,643条拒绝、17条待复核均保留，详见[归档和原因说明](../listen_j_full_response_filter_20261101/README.md#test前缀修复与当前清单2026-11-02)。读取器已支持`original_caption_strip_chat_prefix_v1`：用原始文件与统计验证`original_text`，仅允许删除记录中的开头前缀，实际返回清理后的`target_text`。下述初筛数为历史记录；本目录全量原始清单未修改。

2026-11-02已对test独立执行与train/dev原始初筛相同的文本判断：通过6,537条／10.85h，剔除5,730条／9.64h，待复核17条。通过子集在[独立test筛选入口](../listen_j_test_response_filter_20261102/filtered/summary.json)，其`filtered/test.jsonl`、`index.sqlite`及`crop_config.json`沿用下方读取方式。原始12,284条仍由本目录的`test.jsonl`提供；本次只筛选原始文本，不改写、不解码媒体，不执行放宽外观要求的有声捞回。[8条通过原文样例](../listen_j_test_response_filter_20261102/passed_examples.json)。

## 每条数据的三项内容

1. **speaker audio**：对应pair的speaker归一化音频，按相同源时间范围截取，保留静音间隔，不套用listener静音掩码。
2. **listener first frame**：pair起点附近最近PTS的原始视频帧，使用最终修正版`quality-bounded-crop-v1-20261030`裁框，输出512×512 RGB。
3. **listener回应标注**：已有整段caption原文，逐字保留，作为文本监督目标。

模型输入只有speaker音频和listener单张首帧。listener后续视频、listener音轨、caption和统计元数据不混入user输入。安静倾听样本也保留。

部分caption来自历史旧框；首帧统一最终修正版框，读取审计保留实际标注框与最终框是否不同的信息。不重新打标、不改写原文，不把弱标签或原文声音冲突变成人工真值。

## 文件与500h标记

- `train.jsonl`、`dev.jsonl`、`test.jsonl`：完整引用清单。
- `index.sqlite`：sample_id、split、源时间、人脸大小及JSONL字节位置，支持随机读取。
- `crop_config.json`：最终裁框配置快照，与每条记录中的哈希绑定。
- `summary.json`、`COMPLETE.json`：导出范围、数量、精确时长、清单SHA256。
- `audit.json`：真实媒体抽查、清单文件完整性和错配拒绝检查。
- `coverage_audit.json`：独立核对全量ID集合、原划分、元数据和与旧小脸／500h的合并关系。

采用**清单＋按需读取**，没有批量复制音频、图片或生成latent。speaker实际WAV路径从源pair结果的`normalized_audio.speaker`解析，不猜测文件名；listener首帧从原TAR成员读取。

每条全量记录新增`provenance.in_current_500h_trial`：

- `true`：属于当前500h试选，共285,638条，全部是train。
- `false`：其他最终保留数据。

500h内部包含264,054条大脸主体和21,584条稍小脸有声补充。全量train还包含此前小脸train的800,579条。dev/test分别补齐3,016／4,943条p10≥224数据，不能漏掉这7,959条，也不能把它们计入train。

协议schema沿用读取器已支持的`j-smallface-response-sft-reference-v1`；该名称是历史协议版本名，不代表当前清单仍限于小脸。全量范围由`summary.scope=full`、逐条`selection_reason=all_final_quality_retained`和实际清单决定。

## 读取示例

在项目根目录设置`PYTHONPATH=src`：

```python
import json
import sqlite3
from pathlib import Path
from listen_response_sft import load_example

work = Path('experiments/listen_j_full_response_sft_20261101')
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
speaker_audio = example['inputs']['speaker_audio']
sample_rate = example['inputs']['speaker_sample_rate']
listener_image = example['inputs']['listener_first_frame']
response_text = example['target_text']
```

speaker音频为mono float32，返回实际采样率（当前48kHz）；训练processor须按模型要求显式重采样，不能只更改采样率标记。首帧为PIL RGB图像。`target_text`只能作为assistant监督目标，不能把整条引用JSON作为输入prompt。

读取器核对统计、原文、pair参与者、时间、音频文件SHA256、最终裁框测量记录及配置；错配报错，不静默退回旧框、混音或其他参与者。

## 代码与复现

[导出入口](../../scripts/export_j_smallface_response.py)支持`--scope full`；[读取器](../../src/listen_response_sft/data.py)与此前小脸任务共用。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python scripts/export_j_smallface_response.py \
  --scope full --output experiments/listen_j_full_response_sft_REPRO

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python scripts/check_j_smallface_response.py \
  --data experiments/listen_j_full_response_sft_REPRO

python scripts/check_j_full_response_coverage.py \
  --data experiments/listen_j_full_response_sft_REPRO
```

导出仅使用一个CPU进程读取现有统计；检查脚本只抽样解码真实媒体。完整ID／行数／文件哈希核查与全量音视频解码是不同验收范围，未宣称逐条听看全部媒体。

本轮未修改Qwen-Omni训练器、对话模板或loss，未启动训练；保留原`training_admitted`来源标志，不静默改写。原caption是基于完整listener音视频生成的监督目标，speaker音频和首帧并不能唯一决定其后续动作。

原500h视频生成训练试选仍保留，本次合并仅扩展回应文本监督数据入口。[此前筛选约定和历史清单](../../related/P2_J数据筛选与500h有声补充方案记录_20261101.md)。
