# 全量train/dev：Qwen-30B直接可用文本筛选

2026-11-01，按用户授权处理新的全量回应SFT入口。范围为train 1,086,217条、dev 8,822条，共1,095,039条；test不参与。已于北京时间22:54完成全量导出：保留561,126、剔除531,886、待复核2,027。保留train 556,690条／916.33h、dev 4,436条／7.59h。完成标记和汇总哈希已核对。下方启动与性能记录保留为历史。

- [当前状态及最近批次速度](status.json)
- [GPU实时采样](telemetry.json)
- [后台进程](process.json) · [启动信息](launch.json)
- [冻结配置](config.json) · [完整prompt](system.txt)
- [当前输入入口](../listen_j_full_response_sft_20261101/README.md)

## train/dev前缀修复与当前入口（2026-11-02）

全量扫描发现15,033条以`Assistant:`开头的标签，train 14,925、dev 108。此前保留集合（严格初筛＋有声捞回）已包含其中7,921条，对这些项只删除开头前缀；其余7,112条删除前缀后复判。其他正文、音轨、最终裁框和split不变，原始全量文件不覆盖。

| 划分 | 此前保留 | 本轮新增 | 当前通过 | 当前片段时长 | 本轮拒绝 |
|---|---:|---:|---:|---:|---:|
| train | 686,296 | 4,041 | 690,337 | 1,155.44499h | 2,993 |
| dev | 5,188 | 50 | 5,238 | 9.09755h | 28 |
| 合计 | 691,484 | 4,091 | 695,575 | 1,164.54254h | 3,021 |

本轮新增约9.85446h，待复核0条。直接按严格规则新增3,900条；在尚未通过且净化listener音轨非零的候选中，按原有声捞回规则额外通过191条。后者继续允许衣服／头发／背景等细节，要求明确发声描述且没有严重矛盾或外部证据依赖，没有放宽既定标准。

当前完整通过入口位于`../listen_j_train_dev_response_prefixfix_20261102/current/`，**替代旧的voiced recovery combined入口**：

- [train.jsonl](../listen_j_train_dev_response_prefixfix_20261102/current/train.jsonl)、[dev.jsonl](../listen_j_train_dev_response_prefixfix_20261102/current/dev.jsonl)：此前保留＋本轮新增，已去重，原split保持不变。
- 同目录`index.sqlite`和`crop_config.json`配套使用；用`listen_response_sft.data.load_example(row, crop_config)`读取。`target.original_text`及`removed_prefix`记录确定性前缀删除，返回清理后的`target_text`，原始来源及哈希照常验证。
- [本轮新增train](../listen_j_train_dev_response_prefixfix_20261102/current/recovered/train.jsonl)、[本轮新增dev](../listen_j_train_dev_response_prefixfix_20261102/current/recovered/dev.jsonl)，便于单独检查。
- [本轮拒绝train](../listen_j_train_dev_response_prefixfix_20261102/current/rejected/train.jsonl)、[dev](../listen_j_train_dev_response_prefixfix_20261102/current/rejected/dev.jsonl)；逐条原因在[train.reasons.jsonl](../listen_j_train_dev_response_prefixfix_20261102/current/rejected/train.reasons.jsonl)、[dev.reasons.jsonl](../listen_j_train_dev_response_prefixfix_20261102/current/rejected/dev.reasons.jsonl)。这里仅包含本轮前缀候选的3,021条拒绝项，**不是全量未保留数据**。
- [全部15,033条前缀样本的最终及历次判断](../listen_j_train_dev_response_prefixfix_20261102/current/prefix_decisions.jsonl)，[原文、修改与原始行哈希](../listen_j_train_dev_response_prefixfix_20261102/edits.jsonl)，[前缀处理前完整判断](../listen_j_train_dev_response_prefixfix_20261102/prior_decisions.jsonl)。
- [统计与清单哈希](../listen_j_train_dev_response_prefixfix_20261102/current/summary.json)、[拒绝原因命中统计](../listen_j_train_dev_response_prefixfix_20261102/current/rejection_reason_summary.json)、[最终验收](../listen_j_train_dev_response_prefixfix_20261102/FINAL_AUDIT.json)。

其余未涉及前缀的拒绝／待复核样本，继续保留在本目录`filtered/rejected/`、`filtered/needs_review/`及`../listen_j_voiced_recovery_20261102/filtered/`；原始模型输出和拒绝证据在两次工作的`batches/{split}/*.jsonl`。没有删除任何旧清单或理由。

本轮执行分为原严格规则复判、未通过有声候选复判。两轮最初新增3,939条、171条证据引用不匹配：先用原文直接输入并要求短引用，新增53条；再将仍失败项的quote限制为原文中可选句段，新增99条，最终消除格式待复核。后者只约束证据字符串，keep、issue、category仍由模型按原标准判断，解析和外部证据拦截未放宽。两个阶段的system prompt和判断函数与原工作相同；额外引用修正的user格式及schema变化均在代码中记录。

引用修正结果及原始输出在`quote_retry.jsonl`、`sentence_retry.jsonl`；早期导出保存在`before_quote_retry/`和`before_sentence_retry/`。最终使用`current/`及`COMPLETE.json`。只有通过完整判断与精确引用校验的项才计入新增，不以删掉前缀直接放行。

实现：`scripts/fix_train_dev_caption_prefixes.py`提供prepare/run/export；`scripts/retry_prefix_quote_failures.py`处理证据格式异常，`--sentence-evidence`限制原文句段选择。冻结代码、两轮配置、prompt、逐批结果、GPU日志与引用修正记录都保存在本次目录。执行使用172.29.189.119 GPU7、Qwen3-30B-A3B-Instruct-2507、BF16、1024并发、32768 batched tokens、显存比例0.92、前缀缓存、chunked prefill和CUDA Graph，已释放GPU。

验收检查覆盖：此前全部保留项未丢失且仅前缀改变、所有695,575条索引位置和清单哈希、15,033条前缀修改的原始统计校验、全部候选判断和拒绝原因；另有4条真实音轨／首帧读取、3项前缀测试、7项严格筛选测试、5项有声捞回测试。筛查仍基于文本，不代表全部媒体语义人工验收。test仍使用前一节索引的独立修复入口，不混入train/dev。

## 判断范围

模型只读取全量清单的`target.text`，判断原caption能否**不改写直接用于规划LoRA的SFT监督**。不解码音频或图片，不输入speaker转写、事件统计或目标媒体；输入清单中的speaker音轨、最终裁框、listener首帧引用及500h标记完整保留。

## test前缀修复与当前清单（2026-11-02）

只删除开头的`Assistant:`及紧随空白，正文逐字保留，不调整原筛选prompt或拒绝规则。实查144条：143条原因为聊天前缀，另1条原因为外观描述；清理后使用同一Qwen3-30B、同一推理配置复判，87条通过、57条仍拒绝。当前通过6,624条／11.04717h，拒绝5,643条／9.43783h，待复核17条／0.02930h，总计12,284条。

最新目录：`../listen_j_test_response_prefixfix_20261102/current/`。它替代下方初筛通过清单，旧结果完整保留。

| 内容 | 文件 |
|---|---|
| 当前通过清单 | [current/test.jsonl](../listen_j_test_response_prefixfix_20261102/current/test.jsonl) |
| 拒绝样本引用清单 | [current/rejected/test.jsonl](../listen_j_test_response_prefixfix_20261102/current/rejected/test.jsonl) |
| 每条拒绝原因、引用证据及原判断 | [current/rejected/reasons.jsonl](../listen_j_test_response_prefixfix_20261102/current/rejected/reasons.jsonl) |
| 待复核清单 | [current/needs_review/test.jsonl](../listen_j_test_response_prefixfix_20261102/current/needs_review/test.jsonl) |
| 待复核错误及模型尝试 | [current/needs_review/reasons.jsonl](../listen_j_test_response_prefixfix_20261102/current/needs_review/reasons.jsonl) |
| 全部样本前后判断 | [current/decisions.jsonl](../listen_j_test_response_prefixfix_20261102/current/decisions.jsonl) |
| 144条修改记录、原文、原始完整判断 | [edits.jsonl](../listen_j_test_response_prefixfix_20261102/edits.jsonl) |
| 统计与清单哈希 | [current/summary.json](../listen_j_test_response_prefixfix_20261102/current/summary.json) |
| 原因命中统计（可能重叠） | [current/rejection_reason_summary.json](../listen_j_test_response_prefixfix_20261102/current/rejection_reason_summary.json) |
| 验收结果 | [FINAL_AUDIT.json](../listen_j_test_response_prefixfix_20261102/FINAL_AUDIT.json) |

原因字段：`model_judgment.defects`含类别、原文证据quote和reason；A为外部证据／标注过程或聊天包装，B为无关外观，C为发声状态不明确，D为矛盾或不确定行为。`rule_evidence`记录后置规则及其原文证据。最终以`decision`为准：模型keep=true仍可能被后置规则拒绝。`attempts`保存模型原始输出及解析结果，`error`保存待复核失败信息；`previous_*`保存本次前缀处理前的判断。模型按原prompt仅报告一个决定性问题，记录并非每条文本的全部缺陷。

读取仍使用`listen_response_sft.data.load_example(row, crop_config)`，索引用同目录`index.sqlite`。144条修改记录的`target.format`为`original_caption_strip_chat_prefix_v1`，`original_text`保存原文，`removed_prefix`保存精确删除文本，`text`为实际目标文本。读取器校验原始来源和哈希，只放行这一项确定性变换；其他文本编辑仍报错。所有媒体与最终裁框引用不变。旧原文格式继续兼容。

验收：三路ID并集精确等于原始12,284条、无重复；旧6,537条通过项全部保留；全部JSONL哈希、6,624个索引位置、144条修改项原始统计验证通过。实际读取2条修复后通过样本和1条原文对照的speaker音轨与512×512 listener首帧通过；3项前缀测试与7项筛选回归测试通过。仅文本筛查，未逐条验证音视频语义。172.29.189.119 GPU7已释放。

实现入口：`scripts/fix_test_caption_prefixes.py prepare --work <新目录>`，冻结的`code/driver.py supervise --work <新目录>`执行复判，再用前者`export --work <新目录>`合并完整三路结果。本次冻结代码、配置、prompt和GPU日志均在上述目录。

## 独立test初筛（2026-11-02）

按用户新要求，对原始test 12,284条执行与上述train/dev初筛相同的规则。通过6,537条／10.84873h、剔除5,730条／9.63627h、待复核17条／0.02930h。输出目录为`../listen_j_test_response_filter_20261102/`，通过项在`filtered/test.jsonl`，剔除／待复核在`filtered/rejected/test.jsonl`和`filtered/needs_review/test.jsonl`。通过项配套`filtered/index.sqlite`、`crop_config.json`，沿用原引用读取器。

`PROTOCOL_VERIFIED.json`核对prompt逐字相同、判断／规则函数AST相同、模型和推理参数相同；`FINAL_AUDIT.json`核对三路ID并集等于原始test、无重复、原始引用行逐字节一致、清单哈希，以及所有6,537条通过项的索引位置。文本判断仍是自动候选筛选，未逐条核对音视频语义。没有依据test结果调节prompt，也未将test合并进train/dev。

入口支持`python scripts/filter_full_response_sft.py prepare --splits test --work <新目录>`，随后使用该目录冻结的`code/driver.py supervise --work <新目录>`。本次实际运行在172.29.189.119 GPU7，BF16、1,024并发、2,048条/批、前缀缓存、CUDA Graph和CPU预取；已正常完成并释放GPU。7项筛选测试及5项有声捞回回归检查通过。

## 有声回应捞回（2026-11-02）

已于北京时间09:36完成：201,203条候选中新增捞回130,358条／230.77h片段，剔除63,747、待复核7,098。新增train 129,606条／229.39h、dev 752条／1.38h。合并后train 686,296条／1,145.72h、dev 5,188条／8.97h。`COMPLETE.json`与`MERGED_COMPLETE.json`均已生成；`FINAL_AUDIT.json`记录全量输出哈希、零重复、全部新旧ID及划分一致、28条索引／原文回读，以及新增保留全文无当前规则覆盖的明确外部证据表述。GPU7已释放，未启动训练。

用户确认初筛结果基本干净，下一步从本目录原先未保留的数据中捞回有声回应。此次放宽衣服、背景等无关细节，保留原文，仅要求明确的listener发声以及无实质矛盾／外部证据依赖。既有保留集保持原样。

工作目录为`../listen_j_voiced_recovery_20261102/`。候选来自本轮`rejected/`和`needs_review/`中净化listener音轨非零的记录：train 199,965条、dev 1,238条，共201,203条／353.18h片段。非零波形只作候选门槛；Qwen还须从原caption确认listener说话、笑声、mhm等发声。片段时长不是净发声时长。本轮不重新解码音视频，也不处理test。

完整规则、候选原文、旧分流来源和代码快照分别保存在该目录的`system.txt`、`candidates/{split}.jsonl`、`candidates/{split}.origins.jsonl`与`code/`。当前正式版本先摘录原文证据再判断，允许直接的speaks／utterance描述，不要求额外出现audible；已纠正试跑中把listener说话误认成另一个speaker的问题。历史试跑在`revisions/`，最终试跑结果和助手文字复核在`PILOT_VERIFIED.json`及`pilot_v3_checked_samples.json`。这些检查不代表全量或媒体真值准确率。

正式批次抽查发现模型仍会漏放明确的supplementary data等说明，因此增加独立文本规则，否决明确的补充数据／外部证据依赖及“not visually confirmed”等表述。规则也用于否定行为声明（如“补充数据表明没有点头”）。已在50,390条断点重放最终分流，拦下267条；原模型输出、旧分流和代码保存在`revisions/before_external_evidence_veto/`，记录见`EXTERNAL_VETO_REPLAY.json`，模型推理未重跑。

沿用GPU7、BF16、1,024并发、2,048条/批、32,768 batched tokens、前缀缓存、CPU预取和CUDA Graph。后台监督进程支持批次断点接续，至多三次worker启动；`STOP`在批次边界停止。运行状态见`status.json`、`process.json`、`telemetry.json`和`supervisor.log`。

已生成两个独立入口：

- `filtered/{train,dev}.jsonl`＋`index.sqlite`：本次新增捞回项；完成标记为`COMPLETE.json`。
- `combined/{train,dev}.jsonl`＋`index.sqlite`：原保留项加本次新增捞回项；完成标记为`MERGED_COMPLETE.json`。

两者均逐字节保留原始引用行，并有`crop_config.json`，沿用全量SFT的`listen_response_sft.load_example`读取器。导出校验ID不重复、split一致、原行与caption哈希；合并另外核对两路清单哈希和随机读取索引。只有实际完成才发布相应完成标记；不会自动启动训练。

源码：[recover_voiced_response_sft.py](../../scripts/recover_voiced_response_sft.py)，复用已验证的全量筛选worker及导出路径；5项新增CPU测试覆盖新判定结构、取消旧外观否决、外部证据否决、非零候选门槛、合并原文及重复ID拒绝，原筛选6项测试通过。

## 初筛规则（历史口径）

规则沿用此前100条判断试验：剔除补充信息／检测证据等标注过程说明、无关静态外观或视觉背景、实际发声状态不清、未解决的行为／声音矛盾，以及Assistant等聊天前缀。静音、动作少、姿态基本不变不构成剔除理由。

为减少百万条处理的输出成本，拒绝项只要求一个决定性缺陷、短原文引用和简短理由。判定标准不变，未执行改写。模型判断仍是自动候选判定，不是人工真值，也不证明原caption与真实视频完全一致。

模型结论后叠加明确的文本规则，并分别保存来源：

- 前次试验的视觉background/backdrop句及静态头发外观规则。
- 本次独立dev抽查发现的漏放：没有任何音频／发声描述，或只说没有语音而缺乏其他发声状态说明。这些情况不推定为静音。

输出分为`keep`、`reject`、`needs_review`。无法解析、引用词语和原文不符、正常结束失败、上下文超限等情况进入待复核，不混入保留清单。解析失败最多重试一次。引用若仅大小写或空白不同，会定位回原文真实子串，并同时保存模型原引用；不做模糊词语替换。

## 执行与提速

- 主机172.29.189.119，物理GPU7，UUID `GPU-4d5ee808-e829-c824-c506-cad80ff133f5`；其他GPU保持原任务。
- `/private/ckpts/Qwen3-30B-A3B-Instruct-2507`，BF16、vLLM 0.19.0、TP=1。
- `max_num_seqs=1024`、`max_num_batched_tokens=32768`、`max_model_len=8192`、显存比例0.92。
- 前缀缓存、chunked prefill、FULL_DECODE_ONLY CUDA Graph；先在CPU渲染并批量tokenize，预取线程与GPU推理重叠。
- 每2,048条提交一个原子批次；常驻模型处理后续批次，减少重复加载／预热。temperature=0，输出上限512 token，异常重试上限1,024。
- dev先处理，随后train，两个split始终独立；每批落盘、更新进度，重启从已提交字节位置接续。
- 后台监督进程最多自动重启worker三次；正常结束后自动导出并验收。不会自动启动训练。

前两批共4,096条是真实dev数据，正式接续会复用结果。两批分别在新进程加载后执行，测得约51.8和92.2条/秒，包含首次结构化生成及重试；不能当作常驻稳态速度。资源轨迹和第二批CPU profile已保留，正式吞吐以`status.json`最近批次为准。

## 已完成的检查

1. 源train/dev清单的完整SHA256、字节数及导出完成标记核对通过。
2. 两个真实2,048条批次均已提交。模型结果结构和引用进行程序校验；对12条保留项、6条剔除项做了原文检查，并据发现的发声信息漏放增加两条上述规则。
3. 对旧100条试验回放新增规则，旧41条保留候选没有被新增规则额外剔除；这不代表泛化验收。
4. 6项CPU测试通过：原文引用与决策一致性、大小写／空白对齐不改变词语、缺失声音不推成静音、三路分流保持原始行与split／随机索引、源文件变动拒绝发布、续跑必须按连续已提交批次接续。

截至正式接续前，4,096条为保留2,022、剔除2,063、待复核11；这只是初始dev批次，不代表全量分布。验证阶段前后的原始结果、代码、配置保存在`revisions/validation_v2/`。同一批已生成文本仅重放了引用对齐和后置规则，没有重新推理或修改原caption。

## 完成后的文件

只有自动验收通过后才会发布`filtered/`并写入`COMPLETE.json`：

- `filtered/train.jsonl`、`filtered/dev.jsonl`：保留项的原始引用行，逐字节复制，`target.text`不改写。
- `filtered/index.sqlite`：保留项的随机读取索引，字段与原入口一致，字节位置指向新清单。
- `filtered/crop_config.json`：原最终裁框配置快照。
- `filtered/rejected/train.jsonl`、`dev.jsonl`：剔除项的原始引用行。
- `filtered/needs_review/train.jsonl`、`dev.jsonl`：未得到有效判断的原始引用行。
- `filtered/summary.json`：分split数量／时长、500h成员和净化音轨是否非零的保留分布、输出哈希。
- `batches/{dev,train}/*.jsonl`：每条模型原始输出、有效判断、规则依据、重试记录、原清单字节位置和哈希。

最终逐条核对源ID、split、字节位置、原行哈希、caption哈希、无重复、无遗漏、三路数量之和，并重算源清单完整哈希。被筛掉的文本不会从原数据目录物理删除，也不改变当前AV＋Event训练数据。

保留清单仍可使用原`listen_response_sft.load_example`读取器：从新`filtered/index.sqlite`取位置，在新`filtered/{split}.jsonl`读取原始记录，再传入原接口及`filtered/crop_config.json`。

## 运行维护

实际运行冻结副本`code/driver.py`，不依赖之后对项目脚本的修改。源码入口为[scripts/filter_full_response_sft.py](../../scripts/filter_full_response_sft.py)，测试为[tests/test_full_response_filter.py](../../tests/test_full_response_filter.py)。运行环境Python为`/private/software/conda/envs/vllm/bin/python`。

若需要显式暂停，在此目录创建`STOP`文件，worker会在批次边界停止且监督进程不重启。移除该文件后可重新运行`code/driver.py supervise --work <本目录>`接续；不要并行启动第二个worker。进程锁和GPU身份／空闲检查会阻止重复占用。

## 正式常驻吞吐实测

正式接续后已完成全部8,822条dev并进入train。最近完整train批次端到端约210.5条/秒，采样峰值GPU利用率100%、显存约132.2GiB。进度及剩余时长会随文本长度和重试率变化；此处是运行中快照，未包含最终导出验收。[实测快照](steady_performance.json)。
