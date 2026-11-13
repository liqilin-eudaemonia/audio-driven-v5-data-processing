# P2 多类 Backchannel：打标、原文整合与旧代码迁移说明

更新：2026-10-25。本文供修改旧版二类 backchannel 代码使用，汇总当前已经确认的要求，并区分已实现流程与待实现的训练接口。本次仅编写说明，没有修改训练代码或启动任务。

**目标：开放式发现所有可识别的 listener 反应，保留模型原文和逐次时间；整理跨窗重复后，派生“六类视觉反应＋一类音频回应”的多标签训练目标。开放原文不受七类限制，七类是首轮训练接口，不是原始打标词表。**

## 1. 先统一版本与现状

| 层 | 当前状态 | 后续修改要求 |
| --- | --- | --- |
| 原始打标 | 已实现视觉、纯音频、ASR分路；全量范围按原始预裁框≥384筛选 | 继续保存原始输出，不在推理worker中合并或改写 |
| 原文整合 | 50段CPU流程已实现，版本`listen-existing-result-fusion-verbatim-v2` | 复用代码；补全量原始响应到整合输入的适配器，不能宣称已经全量接通 |
| 七类归属 | 50段已有模型语义归类；原文整合沿用已有类别 | 类别是附加信息，保留依据；不恢复关键词硬映射 |
| 训练目标导入 | 当前候选尚未准入训练，缺完整逐类已知覆盖 | 实现独立七类导入、准入、coverage/unknown与网格投影 |
| P2模型/损失 | 当前`src/p2_listen`仍为两路实现 | 改为视觉六输出＋音频一输出；同步配置、loss、缓存、恢复和评测 |

**核对到的旧代码实际类别是`listener_vocalization`＋`head_nod`，不是视觉`laugh`＋`head_nod`。** 如果另一份旧代码确实只有点头／笑声，也按本文的模态定义迁移：旧“笑声”究竟表示可听发声还是可见笑容，必须依据旧标签来源辨别，不能按名称直接复用通道。

本文关于当前打标与整合行为以代码和冻结运行配置为准；训练接口是后续实现要求。旧《代码修改要求》中的二类列表、两个单输出头与示例配置已过时；旧计划的原裁框448默认值不能悄悄替代当前已确认的≥384打标范围。若pilot另取更高画质子集，必须单独记录筛选和实际数量。

## 2. 数据与模型打标规则

### 2.1 当前数据范围与媒体输入

- 候选总索引：`data_corpora/seamless_listen_local_full/accepted_audio_candidates/index.json`。
- 本轮已确认打标子集为**原始listener预裁框边长≥384像素的722,706段**，原记录合计1,224.604152小时。该数是任务范围，不是当前完成量。
- 保留既有切分、音质要求、primary/onset_reserve身份及起始活动标记；本轮队列限train，不能把它当作dev/test标签池。
- 裁剪采用已确认的`crop_proposals.listener`的`x/y/size`，不是`listener_crop=null`时回退到原画。按half-up取整后裁剪，再bicubic缩放为512×512；筛选阈值用**未取整原框size**，不是缩放后分辨率。
- 缺框／无效框单独记录，视觉不能用原画替代；音频的成功记录不因此变成视觉成功。标注、训练目标和推理首帧的裁框身份必须一致。
- 视觉按8 FPS取样，保留实际时间戳。音频取`normalized_audio.listener`对应listener独立音轨，不能使用混音或speaker音轨。
- 音频按原始源时间定位样本范围，记录样本下标和采样率；现有传输经PCM16并重采样至16kHz，音频模型和ASR使用相同listener来源。
- 训练的speaker条件音轨与上述listener标注/目标音轨用途不同，loader不得按同一个`audio`字段混用。

实际配置与输入实现：[冻结运行配置](../experiments/listen_raw_crop384_20261024T131300Z/config.json)、[full_raw.py](../src/listen_benchmark/full_raw.py)、[运行说明](../experiments/listen_raw_crop384_20261024T131300Z/README.md)。不要把推理代码16秒的防截断预算误当成新的数据切分规则。

### 2.2 模型分工

| 分支 | 当前模型 | 输入 | 输出用途 |
| --- | --- | --- | --- |
| 视觉 | Qwen3-VL-8B-Instruct | listener裁剪视频帧 | 开放视觉动作／表情描述及时间 |
| 音频事件 | Qwen3-Omni Instruct，当前30B-A3B系列路径 | 独立listener音频，纯音频输入 | 开放声音／回应描述及时间 |
| 转写与词时间 | Qwen3-ASR-1.7B＋Qwen3-ForcedAligner-0.6B | 同一listener音频 | 转写、每个词及其已有对齐区间 |

本轮不是Qwen3-Omni原生视听联合打标，也不使用Joy作为当前视觉主分支。视觉笑的动作和可听笑声分别观察、分别保存，不能跨模态自动补标签。

每段同时做整段与3秒滑窗，步长1秒，最后窗口可短于3秒；整段与窗口完全相同时只算一次。保存窗口位置，原始响应使用窗口局部时间。视觉／音频事件推理当前`temperature=0`、`max_new_tokens=4096`，使用JSON结构约束；这些参数不代表ASR也使用相同生成配置。

### 2.3 实际使用的简洁prompt

来源：[prompts.py](../src/listen_benchmark/prompts.py)。`{duration}`是本次输入窗口时长，以下仅以占位符替代运行时数字，正文沿用当前模板。

视觉：`backchannel_visual_simple_v2`

```text
Watch this {duration}-second listener clip. List all visible backchannel behaviors you recognize. Give each occurrence its own start time, end time (seconds from clip start), and a brief description.
Return JSON with an events array. Each event has start, end, modality ("visual"), and behavior (description).
```

音频：`backchannel_audio_simple_v2`

```text
Listen to this {duration}-second listener clip. List all audible backchannel behaviors and sounds you recognize. Give each occurrence its own start time, end time (seconds from clip start), and a brief description.
Return JSON with an events array. Each event has start, end, modality ("audio"), and behavior (description).
```

这里要求所有识别到的backchannel、每次发生的起止时间和简短描述，**不把七类列表塞回prompt限制发现范围**，也不恢复之前过长的逐项限制。模型没写出的事件不能推定不存在。ASR转写可以含错误，强制对齐不能保证转写内容正确。

### 2.4 原始输出必须先保存

每次请求保存sample_id、request_id、分支、模型与prompt版本/哈希、采样参数、窗口起止、原始响应、解析状态、源记录/媒体身份、裁框及音轨证据。原文与解析结果分开，失败也有记录，修复结果另存且可追溯。

```text
raw/{visual,audio,asr}/results/*.jsonl
```

原始事件的`behavior`不改写；规范化后的字段另存。时间转换只进行一次：

```text
clip_start = window_start_s + event.start
clip_end   = window_start_s + event.end
```

源视频起点只在定位原媒体时使用，不再加到clip局部训练标签中。保存原局部时间、窗口偏移和转换后时间；拒绝越界、非数值、非正时长，不能静默裁剪来“修好”标签。结构通过率不是语义正确率；失败、无事件、未检测必须区分。

## 3. 信息整合规则：只整理已有证据，保留原文

当前实现与完整说明：[existing_result_fusion.py](../src/listen_benchmark/existing_result_fusion.py)、[原文整合规则](P2_打标信息原文整合说明.md)。本步骤使用CPU，无新模型推理，无人工答案填入。

### 3.1 分开保存四种内容

1. **原始观测**：模型原文、原始时间、窗口、模态与来源，完整保留。
2. **独立候选事件**：从已有证据选取代表区间与原描述，不等同于训练真值。
3. **相关／分歧证据**：重复窗口、宽区间、不同词或不同声音解释，保留原文，不额外断言一次发声。
4. **回应组／共现关系**：关联同时发生的点头、表情、音频回应；组的外包区间不是持续发声或持续动作标签。

### 3.2 原文规则

- ASR候选保存参考转写原词／原短句，不添加`Says`模板，不改语言、词语或含义。
- 视觉和非语言音频主描述从已有`source_descriptions`选择，多个来源版本全部保存，不另写摘要。找不到来源描述才沿用输入观测描述。
- 不将“嘿嘿”改成笑声，不用局部`Yeah`替换整段`Ed`；已有笑声和其他转写都作为各自来源保留。
- 原始输入、原文来源、选用时间和被单列的分歧可追溯。人工反馈用于改进通用规则与回归检查，不写样本ID特判，也不自动视为逐事件人工真值。

### 3.3 去重与独立次数

- 同内容、不同来源窗口、时间重叠且起止均相差≤0.2秒的记录，才可能属于同次事件。开放事件匹配还使用上游已有类别／语义标识；这不是新做关键词分类。
- 组内两两符合条件，不靠一条宽区间传递连成长链；有歧义不强行并组。代表边界选已有记录，不平均、不取最大外包区间。
- 同一窗口中的多次相同词、长间隔重复词要保留；不能因都是`yeah`或都是点头就合成一次。
- 当前ASR参考优先整段，缺失时用词记录最多的窗口；按词序选择已有有效区间。`Uh huh`可为一个回应短语，`yeah, yeah`仍按各次保留。
- 找不到足够区间时单列未定位项，不能均分一条宽区间来补次数。零时长不充当有效发声区间。
- 词描述／通用发声描述／时间重叠的其他解释可关联为证据，既不重复计数，也不由代码裁决哪一种描述正确。
- 视听时间分别保留。点头发生在几次`yeah`中间，不会把这些声音串成持续长音。

**0.2秒是期望精度与候选匹配容差，不是已验证所有标注达到的精度。** 上游已产生的宽区间也不能仅靠保留原文修复。

## 4. 七类训练目标：附加类别，不替换开放描述

固定顺序必须统一为：

```python
VISUAL_TYPES = ("head_nod", "head_shake", "smile", "laugh", "frown", "surprise")
AUDIO_TYPES = ("listener_vocalization",)
EVENT_TYPES = VISUAL_TYPES + AUDIO_TYPES
TYPE_TO_MODALITY = {**dict.fromkeys(VISUAL_TYPES, "video"),
                    **dict.fromkeys(AUDIO_TYPES, "audio")}
```

以上是迁移要求示意，尚未写入当前训练代码。原始数据模态使用`visual/audio`，模型属性使用`video/audio`，应在适配层明确转换。

| 类别 | 中文与证据 | 不能直接判为此类的情况 |
| --- | --- | --- |
| head_nod | 视觉点头，能辨认上下往复动作 | 静态低头、躯干位移或单次姿态变化 |
| head_shake | 视觉摇头，能辨认左右往复动作 | 单次侧转、看向旁边 |
| smile | 视觉微笑，支持持续区间 | 普通嘴部运动；不能从大笑自动复制微笑 |
| laugh | 视觉笑的动作／大笑 | 仅听到笑声、仅张嘴、说话 |
| frown | 视觉皱眉，有眉部下压／聚拢证据 | 仅写“思考”“不赞同”等推断 |
| surprise | 可见惊讶表情，有组合面部证据 | 单独抬眉、睁眼、张嘴或心理推断 |
| listener_vocalization | listener可听回应／发声，如yeah、okay、嗯、Uh huh、笑声等 | speaker串音、环境声、普通呼吸；归属不明待核验 |

七个维度独立，允许点头＋微笑＋发声同时发生；不使用互斥softmax。“无事件”不另设为第八类。视觉笑与音频笑声分别进入`laugh`和`listener_vocalization`，不能强制相同时段。音频首轮只有一个通道，不把每种回应词设成新类别。

当前模型语义附加归类见[semantic_mapping.py](../src/listen_benchmark/semantic_mapping.py)：理解单个动作及来源，只新增类别、理由和证据，结构校验不使用关键词决定类别。单个原子动作当前最多一个类别；不同事件在同一时刻可以多标签。原文同时含多个动作时，不能把整句话涉及的所有类别都贴到同一个小区间；没有独立边界则保留复合描述及未决状态。

七类之外的眨眼、视线、手势等开放内容继续保存，不强行塞入最近类别。当前50段沿用已有语义结果；全量若缺类别附加层，需要另行批量归类，这可能调用模型，**不能把这项新语义工作冒称为纯CPU去重**。当前CPU整合本身不调用模型。

## 5. 从候选到训练标签的接口（待实现）

### 5.1 输入与派生产物

输入当前`labels.candidate.jsonl`，保留对应`review.json`和完整原始观测。每个候选至少读取：

| 字段 | 用途 |
| --- | --- |
| sample_id、event_id | 稳定身份；映射、去重和评测按身份追踪 |
| start/end | clip起点秒坐标，使用有效半开区间[start,end) |
| modality、categories | 模态及已有类别，不能从描述关键词重算 |
| description、text、original_descriptions | 原文与词内容，训练类别不覆盖它们 |
| description_source(s)、timing_evidence、source_observation_ids | 描述与边界来源；不同事件来源字段可能不同，不要求每条都含全部字段 |
| status、review_flags、未定位／分歧记录 | 准入和遮罩依据，不能忽略后直接当真值 |

新增独立训练导出器：输出canonical标签、`candidate_event_id → training_event_id`映射、排除/未决原因和版本。保留开放数据不动；`description`不能被替换成固定类别名。建议训练schema升级为`listen-events-v2-visual6-audio1`，实验协议沿用计划中的`p2-listen-event-v2-visual6-audio1`；这两个名称是后续迁移约定，不能宣称旧loader已支持。

每个样本的canonical标签必须有：schema、sample_id、time_origin、duration_s、固定taxonomy、逐类coverage/unknown、事件及来源；同时在实验契约保存`K_train`、mapping版本、标签hash及准入策略。若给旧严格schema增加字段，同步修改validator，不能额外加字段后假定旧loader会忽略。

### 5.2 coverage与unknown

- `coverage[k]`表示该类别已确定标签含义的时间支持，必须有来源；`unknown[k]`表示该类不确定区间。
- 候选没检出、模型返回空列表、类别不启用，都不等于全段负例。不得初始化`coverage=[0,T]`再把未标出部分当0。
- 视觉类别已知不能推出音频已知，某一视觉类别已知也不能外推其他类别。
- `K_train`固定本轮可监督类别集合，未启用类别保留输出通道但整类mask；稀有类不够可靠时不阻塞其他类。
- 现有175条仍`training_admitted=false`；不能只改布尔值就准入。需要可记录的弱标签接受政策／核验、边界状态和逐类已知覆盖，dev/test参考另行核验。
- 音频词边界只表示已观察的对齐支持，例如一个很短的Yeah核心区间不自动包含未定位尾音。

### 5.3 时间网格和损失

视觉标签与mask为`[B,Tv,6]`，音频为`[B,Ta,1]`，两种时间轴长度允许不同。使用实际秒支持区间投影，不按数组下标硬对齐。当前计划视觉输入8 FPS、temporal patch=2约4 Hz时间组，音频约8 Hz；应以实际layout为准，不从原视频25 FPS推导事件头步长。

对完整落入已知覆盖、无unknown/censored/padding影响的bin：

```text
y[k, bin] = duration(bin ∩ union(accepted_event_intervals[k])) / duration(bin)
known[k, bin] = true
```

已知覆盖内没有该类事件时才是0；其他bin的known=false。这里为计算占用率求并集，不删除原始事件内部边界，也不改变事件次数。一个bin内多次快速回应无法由占用标签区分，原始逐次标签仍保留；本轮不新增计数头、词分类或转写loss。

每类用带mask的BCE-with-logits，先按该类全局有效bin数归一化，再对本次有有效监督的启用类取平均。七类全启用时每类1/7，不能把六路视觉loss相加后沿用二类权重。`pos_weight`只根据train已知支持计算并固定；某类没有可靠正负支持时停止启用该类并记录，不让程序崩溃或伪造负例。全unknown时事件loss为graph兼容的零项，基础AV／feature loss仍正常。

## 6. 逐文件修改清单

当前实际路径均相对项目根目录。以下是后续实施任务，不是已完成的代码变更。

| 文件 | 查到的旧行为 | 应修改为 |
| --- | --- | --- |
| `src/p2_listen/contracts.py` | EVENTS只有发声、点头 | 固定七类顺序＋显式类别到模态表；区分标签版本 |
| `src/p2_listen/__init__.py`、`config.py` | v1版本、校验写死两类 | 新协议版本、固定taxonomy、K_train与mapping配置校验 |
| `configs/p2_listen_event_base.json` | types是两类 | 新建七类配置；保留历史配置，避免覆盖旧实验语义 |
| `src/p2_listen/events.py` | `zip(EVENTS,(audio_dim,video_dim))`和两模态zip，只构建两路单输出头 | 视觉头Linear(6)，音频头Linear(1)；按明确模态池化和固定类别顺序输出 |
| `src/p2_listen/annotations.py` | v1严格schema；rasterize也是两模态zip；每类必须有正负否则报错 | 新schema导入、七类逐类投影、K_train与coverage/unknown；缺支持类可屏蔽并记录 |
| `scripts/p2_listen.py`、`src/p2_listen/dataset.py::import_labels` | CLI的import-labels调用旧canonical标签导入 | 接入候选到七类训练标签适配，验证schema、准入与来源，不直接把候选文件当训练真值 |
| `src/p2_listen/model.py` | 构造旧EventHeads | 绑定同一份最终Uv/Ua，传递新head配置；不另建与生成无关的监督支路 |
| `src/p2_listen/training.py` | S0梯度审计含硬编码(head_nod,listener_vocalization) | 所有统计、审计和DDP归一化按七类与显式模态表遍历 |
| `src/p2_listen/features.py`、`cache.py` | events缓存基于旧标签/网格 | 加入schema、taxonomy顺序、K_train、mapping/标签及网格签名；重建事件缓存 |
| `src/p2_listen/audit.py`、`scripts/check_p2_listen_cpu_ddp.py` | 合成样本和权重只含两类 | 覆盖六视觉＋一音频、多标签并发、稀有类全mask和不同时间网格 |
| `src/p2_listen/evaluation.py` | 依EVENTS聚合，旧固定类集合 | per-class/macro按冻结评测类别集，分别报告视觉/音频；未知类不能计零分混入平均 |
| `tests/test_p2_listen_core.py`、`tests/test_p2_listen_evaluation.py` | 二类fixture和断言 | 升级七类测试并保留旧版本拒绝/迁移测试 |

**最容易漏掉的错误：只把EVENTS改成七个，保留两个元素的zip。** Python会静默截断为两项，不一定报错。head构造、forward、rasterize、梯度审计都必须同步消除这种依赖。

建议保持上层训练循环的按类别字典接口：视觉头输出六通道后明确拆成六个`[B,Tv]`条目，音频拆成一个`[B,Ta]`条目；logits、y、known使用完全相同的键和顺序。不要拼成一个假设Tv=Ta的`[B,T,7]`张量。

## 7. 训练、缓存与推理边界

- 事件头读取实际送入LTX的同一份最终U：视觉`Uv→[B,Tv,Dv]→6`，音频`Ua→[B,Ta,Da]→1`。按原计划使用LayerNorm→Linear(128)→GELU→输出层。
- E0 `probe_only`对事件头输入detach，事件头可以训练但事件loss不影响U；E1 `joint_event`保留事件loss到U的梯度。两组同样的七类结构、初始化、数据、mask和噪声协议。
- 维持原有AV与feature训练路径；冻结LTX参数不等于把AV forward包进no_grad。事件扩类不能改变输入几何或让GT决定query数量。
- GT listener视频、音频、事件类别、时间与描述只作训练目标/离线评测。推理仍只用speaker音频、listener首帧、固定文本及外部时长/几何；不得将标签描述塞进Omni/Gemma prompt。
- 旧二类事件缓存、分类头及optimizer状态不得直接当七类resume。新实验重新初始化七类头；公共权重若作为warm start必须显式记录，不能冒充严格续训。
- 输入H缓存只按输入身份、prompt、源模型和布局检查复用；改变标签不应迫使重算H。若裁框/首帧/布局变了，相关缓存须失效。事件缓存必须重建。
- 逐类threshold在dev确定；多类和模态的评测范围固定。事件按类一对一匹配，快速重复不靠长区间合并来提高分数。

## 8. 最小验收与建议实施顺序

1. 先建立七类常量、模态表、新schema和候选适配器，用当前50段检查原文与来源不丢失；尚未准入的候选不能绕过mask直接训练。
2. 用小张量测全部七个输出通道和loss梯度，确认head/rasterize没有zip截断。测试点头＋微笑＋发声同时为正，视觉/音频长度不同仍成立。
3. 测两次Okay分开、重复Yeah次数不被吞、跨窗同次候选减少、嘿/Ed原文不被纠正；原始输入hash不变。测试不要求根据那五组人工意见强制输出某个词。
4. 测零时长、窗口偏移只加一次、无效裁框、末尾padding、unknown、某类无支持、某rank无标签与DDP/累积归一化。
5. 测标签改动不污染输入H缓存、旧二类缓存/头无法误当七类resume；推理隐藏GT标签仍可运行。
6. 再做至少5步真实模型S0，核验E0/E1梯度区别、冻结参数与实际AV生成；然后才扩pilot。当前文档交付不启动这些任务。

验收报告须分别写“原文/数据完整性”“七类工程正确性”“语义与时间精度”。测试通过不能替代对模型漏检、错词、边界精度的评估。

## 9. 可直接交给修改代码者的任务摘要

> 将P2 listen事件监督从旧二类改为固定顺序的head_nod、head_shake、smile、laugh、frown、surprise和listener_vocalization。保留开放原文，原始打标采用Qwen3-VL视觉、Qwen3-Omni纯listener音频及ASR/对齐分路，使用512裁框和简洁逐次事件prompt。复用CPU原文整合，禁止“嘿→笑声”“Ed→Yeah”和Says模板改写。七类归属仅附加，不覆盖原文，不使用关键词硬映射。实现独立训练标签适配、逐类coverage/unknown/K_train；视觉六通道、音频一通道，多标签BCE按类归一化。同步改schema、两元素zip路由、缓存、checkpoint、梯度审计、评测和测试。当前候选未准入训练，全量适配尚待实现；先CPU验证，再真实S0，不自动启动全量训练。

## 10. 入口与证据

- [总索引](../README.md)
- [七类实验计划](P2_倾听事件监督实验计划.md)；[旧代码要求](P2_倾听事件监督代码修改要求.md)（二类内容以本文和七类计划修订）
- [当前整合规则](P2_打标信息原文整合说明.md)、[50段结果与限制](../experiments/listen_existing_fusion_verbatim_20261025/README.md)
- [GitHub原文整合备份](https://github.com/liqilin-eudaemonia/audio-driven-v5-data-processing/tree/dff56f9f8806dc0d4e61dd9b1592a767b489423a/event_labeling/integration_verbatim)，含代码、规则、输入、结果及CPU重放
- [模型语义归类](../src/listen_benchmark/semantic_mapping.py)、[当前全量原始打标运行说明](../experiments/listen_raw_crop384_20261024T131300Z/README.md)

当前50段：175个独立候选（134视觉＋41音频）；682条输入观测完整保留；2个词无法定位单列；15段无候选不等于负例。备份10项整合测试和逐字节重放已通过。这些是当前小样本整合证据，不是七类训练完成或全量结果完成的证明。
