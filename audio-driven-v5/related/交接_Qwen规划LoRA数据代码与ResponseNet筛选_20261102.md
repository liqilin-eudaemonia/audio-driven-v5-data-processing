# 交接：Qwen-Omni规划LoRA训练与ResponseNet子集筛选

**后续状态更新（2026-11-02）：** 用户已要求完成评测加速。现已启用GPU检脸／关键点／情绪、共享GT缓存及并行像素指标；全部推理结束后自动使用四台机器的已释放GPU分担评测。因此推理完成不等于四机已可立即启动LoRA，需先确认评分释放资源。[加速与当前调度](../experiments/p2_metrics_acceleration_20261102/README.md)。下文交接快照保留原时间点状态。

> 后续更新：新JSONL＋SQLite Dataset、独立训练／集群配置、dev评估策略已完成；全清单、12条真实媒体和14项回归验证通过。未启动正式训练，新数据GPU试训尚未完成。详见[最新接入交接](P2_Qwen规划LoRA新筛选数据接入_20261102.md)。以下“待接入”内容保留为本交接初始状态，不再代表当前代码。

更新日期：2026-11-02。项目根目录：`/private/liqi_workspace/project_code/audio-driven-v5`。本文用于接手后开展两项工作；本次只整理文档与已有检查证据，没有启动新LoRA训练，也没有创建或切换正式测试子集。

## 1. 当前结论与待完成事项

| 工作 | 已有资源 | 下一步 |
|---|---|---|
| Qwen-Omni规划LoRA | 新筛选train 690,337条、dev 5,238条；引用读取器；旧数据上完成两轮的SFT训练代码 | 将新JSONL＋SQLite读取器接入训练器，建立独立配置和输出目录，验证真实输入与标签后再训练 |
| ResponseNet筛选 | 原586条引用清单；全量音轨非零检查；48条候选的语音／能量检测 | 复核有声候选，结合来源与时长抽取固定子集；128条是建议，尚未冻结名单 |

重要区别：这里的LoRA是“speaker音频＋listener首帧→回应描述”的文本SFT，不是AV／Event loss更新Qwen hidden的S2。已完成的两轮规划LoRA使用旧285,638条500h数据，新筛选数据尚未用于那两轮训练。

## 2. LoRA训练数据入口

### 2.1 当前应使用的数据

目录：`experiments/listen_j_train_dev_response_prefixfix_20261102/current/`。

| 划分 | 条数 | 片段时长 | 文件 |
|---|---:|---:|---|
| train | 690,337 | 1,155.44 h | [train.jsonl](../experiments/listen_j_train_dev_response_prefixfix_20261102/current/train.jsonl) |
| dev | 5,238 | 9.10 h | [dev.jsonl](../experiments/listen_j_train_dev_response_prefixfix_20261102/current/dev.jsonl) |

同目录的[index.sqlite](../experiments/listen_j_train_dev_response_prefixfix_20261102/current/index.sqlite)、[crop_config.json](../experiments/listen_j_train_dev_response_prefixfix_20261102/current/crop_config.json)、[summary.json](../experiments/listen_j_train_dev_response_prefixfix_20261102/current/summary.json)配套使用。JSONL是引用清单，媒体保留在原位置，并非独立WAV／图片目录。

保留数据来自三部分，计数为不重复的新增项：

| 来源 | train | dev |
|---|---:|---:|
| Qwen3-30B初次严格筛选通过 | 556,690 | 4,436 |
| 放宽无关外观要求后捞回有声回应 | 129,606 | 752 |
| 聊天前缀清除并复判后新增 | 4,041 | 50 |
| 合计 | 690,337 | 5,238 |

筛选模型是Qwen3-30B-A3B-Instruct-2507；训练基座是Qwen2.5-Omni-3B，两者用途不同。筛选主要读取目标文本，没有对所有样本逐条核验媒体语义。

有声捞回规则：从原先未保留项中选净化listener音轨非零、文本明确描述listener发声的候选；按用户要求放宽衣服、头发、背景等描述，优先保留原文，拒绝发声不明、实质矛盾或依赖外部证据的标签。非零音轨与文本判断不等于人工确认真实发声。外观描述尚未批量删除，也尚无它对训练影响程度的定量结论。

前缀修复只删除开头`Assistant:`等已记录的聊天前缀及紧随空白，不改写正文。此前已保留7,921条的前缀也已清理，这些不额外计入新增4,091条。读取器通过`target.original_text`、`removed_prefix`保留追溯。

原始未筛选全量：train 1,086,217、dev 8,822、test 12,284，见[原始数据读取说明](../experiments/listen_j_full_response_sft_20261101/README.md)。不要误用旧严格通过清单或有声捞回`combined/`替代上方最新`current/`。

独立SFT test最新通过6,624条，位于`experiments/listen_j_test_response_prefixfix_20261102/current/test.jsonl`；不并入train/dev。它与本文ResponseNet的586条不是同一数据集。

### 2.2 实际输入、目标和读取器

- 输入：配对speaker音频＋listener最终修正版裁框的单张首帧＋system/user规划指令。
- 目标：清单中保留的listener自然语言回应描述，仅作assistant监督。
- 不把listener音轨、后续视频、目标描述或speaker转写混入user条件。
- 首帧读取输出512×512 RGB，之后仍会经过Qwen processor自己的图像缩放；512×512不等于Qwen模型实际视觉token分辨率。
- speaker按源pair中的`normalized_audio.speaker`与原时间窗读取；不能误套listener净化／静音门控。返回实际采样率，目前通常48kHz，接Qwen前须真实重采样到16kHz。
- 部分历史caption对应旧标注框，实际输入统一最终框；`annotation_crop_differs`保留这种差异，没有重新生成所有caption。

读取器：[src/listen_response_sft/data.py](../src/listen_response_sft/data.py)，入口`load_example(row, crop_config)`。在项目根目录、`PYTHONPATH=src`下可按下面方式读取一条，不触发训练：

```python
import json
import sqlite3
from pathlib import Path
from listen_response_sft import load_example

work = Path('experiments/listen_j_train_dev_response_prefixfix_20261102/current')
crop_config = json.loads((work / 'crop_config.json').read_text())
conn = sqlite3.connect(f'file:{work}/index.sqlite?mode=ro', uri=True)
conn.row_factory = sqlite3.Row
loc = conn.execute("SELECT * FROM samples WHERE split='train' ORDER BY sid LIMIT 1").fetchone()
with (work / loc['file']).open('rb') as f:
    f.seek(loc['offset'])
    row = json.loads(f.read(loc['bytes']))
example = load_example(row, crop_config)
# inputs: speaker_audio, speaker_sample_rate, listener_first_frame
# assistant label: example['target_text']
```

SQLite连接应由各DataLoader worker独立持有。媒体／首帧／源文本／前缀变换的来源校验沿用现有读取器。

### 2.3 筛选、捞回和前缀处理代码

| 用途 | 入口 |
|---|---|
| 初次严格筛选 | [scripts/filter_full_response_sft.py](../scripts/filter_full_response_sft.py) |
| 有声回应捞回 | [scripts/recover_voiced_response_sft.py](../scripts/recover_voiced_response_sft.py) |
| train/dev前缀修复与复判 | [scripts/fix_train_dev_caption_prefixes.py](../scripts/fix_train_dev_caption_prefixes.py) |
| 证据引用格式重试 | [scripts/retry_prefix_quote_failures.py](../scripts/retry_prefix_quote_failures.py) |
| 完整筛选、拒绝和追溯说明 | [筛选README](../experiments/listen_j_full_response_filter_20261101/README.md) |

旧拒绝、待复核项及理由均保留。前缀修复目录下的`current/rejected/`只含本次前缀候选中仍拒绝的3,021条，不代表所有未保留数据。其他拒绝项在原严格筛选及有声捞回目录的`filtered/`和`batches/`。

## 3. LoRA训练代码及需要补齐的接入

| 模块 | 文件／目录 |
|---|---|
| 单节点训练入口 | [scripts/p2_planning_sft.py](../scripts/p2_planning_sft.py) |
| 四机DDP启动器 | [scripts/p2_planning_cluster.py](../scripts/p2_planning_cluster.py) |
| Trainer、loss和保存 | [src/p2_planning_sft/train.py](../src/p2_planning_sft/train.py) |
| 旧Dataset、完整prompt、collator | [src/p2_planning_sft/data.py](../src/p2_planning_sft/data.py) |
| 基座和PEFT LoRA | [src/p2_planning_sft/model.py](../src/p2_planning_sft/model.py) |
| 旧轮次续训逻辑 | [src/p2_planning_sft/continuation.py](../src/p2_planning_sft/continuation.py) |
| 旧正式配置 | [configs/p2_planning_sft_3b_32gpu.json](../configs/p2_planning_sft_3b_32gpu.json) |
| 四机连接与NCCL配置 | [configs/p2_planning_cluster_32gpu.json](../configs/p2_planning_cluster_32gpu.json) |
| adapter自由生成入口 | [scripts/p2_planning_generate.py](../scripts/p2_planning_generate.py) |
| 训练与验收说明 | [原SFT实现记录](P2_Qwen回应规划LoRA_SFT代码准备_20261101.md) |

现有`PlanningDataset`仍读取旧`train_cache/manifest.json`、`media_audit/`和`references/`；`train.py`正式train使用`config['train_cache']`，dev使用旧24条检查集中的4条dev。**不能仅将train_cache改成新current目录就直接开跑。** 新引用读取器存在，但训练器尚未适配新数据。

接手后的具体工作：

1. 为新JSONL＋SQLite增加Dataset适配，调用`load_example`，转成collator要求的`sample_id/split/duration/image/audio/target`，音频真实重采样到16kHz。按清单时长建立`lengths`供现有时长分组使用。
2. 将train/dev显式接到新清单，更新训练身份记录：清单哈希、裁框配置、数据ID、prompt与代码；保留旧数据模式便于复现历史实验。
3. 新建配置和输出目录，不能覆盖旧两轮结果。新数据不符合现有“同数据追加一轮”的严格续训条件，不要直接`--resume checkpoint-2232`。从原始3B重新训练，或以旧adapter初始化再开新优化器，是待确定的实验选择。
4. 复用现有assistant-only collator和全局有效token归一化loss；检查标签清理后文本、EOS、前缀mask、输入角色、首帧裁框和音频窗口。做真实小批反传、保存和重载生成检查，再进行四机吞吐检查。
5. 新dev有5,238条，不能照搬旧4条dev每100步全量评估的低成本预期；确定固定诊断子集与完整dev评估频率。训练轮数、保存频率和初始化方案尚未为新数据冻结。

### 可复用的训练设置

基座`/public/2_CKPT/3_PreTRAINED/Qwen2.5-Omni-3B`。teacher forcing next-token prediction，只对assistant描述及`<|im_end|>`做交叉熵；system/user/多模态前缀/padding为`-100`。无AV或Event loss。

Thinker语言模型36层的q/k/v/o及gate/up/down投影LoRA：rank32、alpha64、dropout0，约5,987万参数；视觉／音频编码器、原语言参数与lm_head冻结。标准PEFT adapter与旧AV S2自定义LoRA接口不能直接混用。

旧四机设置：每卡8、global batch256、LR峰值1e-4、warmup3%、cosine、fused AdamW、BF16、SDPA、TF32；关闭梯度重算，每卡4个持久数据worker、pin memory、时长分组、短音频动态padding、loss仅对监督位置转FP32。旧数据实测约0.692秒/步只是历史窗口，新数据从原媒体按需取帧可能有不同供数瓶颈，不能直接当作新训练ETA。

不需要为文本SFT重提Gemma、LTX latent或冻结Qwen hidden缓存。是否预取／缓存speaker波形和首帧，应根据新读取器实测决定。

共享Python：`/private/liqi_workspace/project_code/Omni-LTX-stage2/.venv/bin/python`。从项目根执行`python scripts/p2_planning_cluster.py`仅展示旧配置命令，不启动；`--launch`才会训练。新Dataset接入前不要直接用默认配置启动，否则仍跑旧500h。推理环境不要混入评测环境的Python3.10依赖路径。

### 历史权重与效果边界

- 一轮：`experiments/p2_planning_sft_3b_32gpu_20261101/final_adapter`，完成1,116步。
- 两轮：`experiments/p2_planning_sft_3b_32gpu_epoch2_20261101/final_adapter`，累计2,232步，`COMPLETE.json`确认成功。
- 数据均为旧285,638条train；当时4条dev只作流程诊断。
- [同24条生成对照](P2_Qwen规划LoRA两轮24条对照_20261101.md)：描述更完整，但第二轮未一致优于第一轮，仍有虚构外部证据等问题；不能宣称已经证明AV质量提升。

## 4. ResponseNet原数据与已查明的问题

主入口：[OmniResponse references.jsonl](../../OmniResponse/evaluation/responsenet_pairs/prepared_20261102/references.jsonl)，586条、约1.22小时，来自135个有pair的官方源样本。所有角色固定：speaker为条件音频，listener为要生成的人。

媒体位于`evaluation/responsenet_speaker_j/cuts/`。推理读取`speaker.audio`与`listener.video`首帧；GT为完整listener视频和listener音轨。当前输出512×512，参考图与GT从640×640显式bicubic缩放，保留各条原始FPS和时间窗。

### 4.1 ASR来源

- 官方词时间得到514条pair；本地Whisper large-v3词时间补出72条。
- 原官方ASR没有可用词、导致未切出pair的样本，才进入补跑；不意味着官方ASR文件一定不存在。
- 本地Whisper输入官方角色分轨，结果独立存在`local_asr/`，未修改官方`test/text`。
- ASR词时间是整段源视频的绝对时间；映射到pair时减去`source_start`并处理边界。
- 37条全零listener音轨仍有ASR文本，逐条来源均为本地Whisper。可能涉及词时间跨进静音区间或识别幻觉，尚未逐条做根因归类。
- 不能用`ASR word_count > 0`定义有声，也不能以`official_word_time`分组宣称完整复现论文协议。

来源与代码：[HANDOFF](../evaluation/responsenet_speaker_j/HANDOFF.md)、[处理说明](../evaluation/responsenet_speaker_j/README.md)、[补跑脚本](../evaluation/responsenet_speaker_j/transcribe_remaining.py)、[切分代码](../evaluation/responsenet_speaker_j/segment.py)。

### 4.2 实际listener音轨检查

| 结果 | 条数 |
|---|---:|
| 完全全零 | 538 |
| 非零 | 48 |
| 非零中通过现有能量门槛的声音活动 | 34 |
| 非零中VAD检出语音 | 25 |

语音累计约10.28秒，单条累计最大约1.48秒。检测采用现有`SpeechDetector`与`activity`：Silero VAD、-50dBFS能量门槛；多声道先均值合并、重采样16k。25条是自动语音检测阳性，不是人工确认的发声全集；短“嗯”、笑声等可能漏检。538条“全零”只描述当前官方分轨切片，不据此推断原始未分轨录音的所有声音。

已保存[586条逐条音轨检查](../experiments/planning_sft_responsenet_handoff_20261102/responsenet_gt_audio_audit.jsonl)和[汇总及37条异常ID](../experiments/planning_sft_responsenet_handoff_20261102/responsenet_gt_audio_summary.json)。没有根据这些检查删除样本或修改ASR。

## 5. 建议的子集筛选步骤

用户要减少测试量；此前讨论128条候选，尚未确认最终数量或生成正式名单。

1. 先复核48条非零listener音轨候选，区分真实语音／非语言发声、微弱残留或噪声。可以先全部纳入候选，避免丢失本来就少的短回应；不能将48条直接全部标为“有声真值”。
2. 若按128条方案，初始候选可用48条非零＋80条全零；全零部分按原始对话来源和时长分位数分层、固定种子抽取。来源数量可能多于名额，优先分散覆盖，无法保证覆盖每个来源。同来源多条优先时间不重叠、间隔较远。
3. 检查动作强弱覆盖，可用头姿／人脸关键点位移作代理，注意镜头运动。覆盖不足时在相同发声／时长层内替换候选，记录理由。
4. 画面难度先用于识别损坏、错位或检测失败；情绪／回应类型无可靠标签时先不设硬配额。不要按模型生成效果挑样本，也不要把技术故障静默当作低质量样本删除。
5. 冻结正式ID、种子、来源、时长层、发声分组和纳入理由。四权重共用同一清单、同一生成规划和seed；已生成结果按ID复用。
6. 按静默／发声组报告结果。48＋80是有意提高非零音轨覆盖率，整体分数不代表原586条自然分布；小子集FID/FVD波动较大，适合初筛。零散短回应不宜强行计算长语音UTMOS，应明确N/A和有效覆盖。

需要交付的筛选产物建议为独立目录中的`references.jsonl`、`selection.jsonl`、`summary.json`与未选ID清单。保留原586条主清单。现有绑定入口默认要求完整586条，缩小测试集时需让生成、绑定、metric manifest和最终合并的expected ID集合统一指向新子集，不能只删生成任务。

## 6. 当前推理队列与接手注意事项

运行目录：[p2_testsets_4ckpt_24gpu_20261102](../experiments/p2_testsets_4ckpt_24gpu_20261102/README.md)。名字中的24gpu是历史名称，当前实际node17／node12／node4／node8四机32卡。

当前范围仍是ResponseNet 586条×4权重＝2,344个结果，seamless暂停。顺序为AV＋Event 6,000→8,927，再仅AV 6,000→8,927；使用未微调Qwen-Omni-3B生成共享规划，不能默认换成上述规划LoRA。

指标独立在172.29.189.119空卡运行，现有八项视频指标加UTMOS，生成与评分并行；没有自动改成128条。此次[队列快照](../experiments/planning_sft_responsenet_handoff_20261102/queue_snapshot.json)记录交接时进度；持续变化以运行目录`status.json`、`metrics_status.json`、`GENERATION_COMPLETE.json`和`COMPLETE.json`为准。交接快照（2026-11-02 05:37 UTC）中，两组AV＋Event权重各586条已全部导出，队列进入仅AV第6,000步权重的推理。

主代码：[src/p2_testsets_eval](../src/p2_testsets_eval)、[CLI](../scripts/p2_testsets_eval.py)。当前已有生成无需因筛子集重跑；若后续切换队列范围，应先停止相应控制器和worker，保留完整已提交结果，再统一重分片和评测入口，不能在worker仍读取旧分片时直接覆盖清单。

机器：node17 `172.29.15.91`、node12 `172.29.252.37`、node4 `172.29.225.51`、node8 `172.29.104.233`，均root／22端口。推理与训练机器信息分别以各运行目录`cluster.json`和规划训练cluster配置为准，不手抄历史错误地址。评测机连接配置在当前队列`metrics_cluster.json`，端口22；20476不提供SSH服务。

## 7. 接手后的推荐顺序

先读取本文、最新数据summary、训练Dataset／train.py，确认新数据接入缺口；并读取队列实时状态，避免与正在运行的推理争用四机。随后准备新LoRA数据适配及独立配置，同时完成ResponseNet候选复核与固定子集清单。新训练初始化／预算和正式子集数量是待落实项，本文的建议不等于已经启动或切换。
