# GLARE：原 speaker 音轨韵律优先筛选

2026-11-10。用户已批准先实现完整韵律条件子集方案。本目录独立于当前非韵律全量队列；没有自动停止、替换该队列，也没有启动正式训练。

代码入口：

- [plan / extract / assemble](../../scripts/prepare_glare_prosody_subset.py)：冻结试跑清单、按源音轨提取、导出分 split 清单。
- [审计、源网格映射、缓存](../../src/glare_prosody_subset.py)。
- [官方脚本运行适配](../../src/glare_prosody_source_runtime.py)：直接调用官方 `Qwen2AudioAnalyzer.infer_chunks`、`load_waveform` 和 `normalize_waveform_loudness`，不复制另写生成算法。
- [入选 pair 的非韵律缓存入口](../../scripts/run_glare_prosody_subset_cache.py)：共享原缓存，过滤原有组内的 ordinal，单独写运行状态。
- [原始回答与模型输入音频复核](../../scripts/review_glare_prosody_subset.py)：最多64块，按状态分层导出静态 HTML、FLOAT WAV 和 JSON，不启动服务。
- [测试](../../tests/test_glare_prosody_subset.py)。

## 当前决定：官方实现优先，已恢复全量（2026-11-10后续）

**本节覆盖下方诊断暂停和严格筛选结论。** 用户明确要求“最大限度沿用官方代码，只要能顺利训练，不做太多无用功”，已写入项目[AGENTS.md](../../AGENTS.md)。诊断证据保留，不再作为阻断或额外筛除依据。

全量在node11 GPU7恢复，按当前约25GiB空闲显存使用batch8；原始prompt、完整greedy/max64、默认重复惩罚和官方规范化不改。仅保留必要audio→audios接口兼容、source-grid数据映射、CPU预读、显存/OOM管理和断点恢复。没有补时长prompt、没有early-stop。

**“全量几乎留不下数据”属于之前自加严格原答筛选的结论，不是官方处理结果。** 当前条件按官方`normalize_result`输出，再调用官方`IntensityEncoder.text_to_frame_vector`；NULL为零强度，原答拒答/格式/截断本身不再使pair被剔除。官方合法数值输出（即使定位不可靠）也按官方保留。技术读取/执行错误重试或单列，不能冒充模型NULL。

[实际执行策略](full/execution_policy.json) · [实时状态](full/controller_status.json) · [正常化/向量生成接入](../../scripts/prepare_glare_official_conditions.py) · [3项直接官方函数兼容测试](official_policy_tests.log)。严格审计字段继续保留作观察，不决定数据准入；历史rawcache协议的admission字段不再控制当前输出，当前输出policy由execution_policy和official_conditions身份定义。

3,287条已有完整记录继续复用；3,259条只有早停前缀、缺少完整官方后缀的记录移入`early_prefix_history`后补跑，避免将自定义早停当作官方完整结果。原缓存/原媒体不删除。输出在`full/official_exports/`，最终`full/LATEST.json`指向官方条件清单；原train/dev划分与已有LIA等特征继续复用。当前仍未启动正式训练。

## 当前状态：全量已按用户要求暂停，提取器未通过定位验收

**请以本节覆盖下方“运行中／预计完成时间”的历史记录。** full/controller与worker均已停止，`full/STOP`保留，停前6,546块全部缓存保留。原LIA仍暂停；其他任务未停止。全量没有完成，没有导出正式筛选清单，不能把这些提取失败当成原pair质量不合格。

[排查报告](extractor_diagnosis/REPORT.md) · [8条可播放负对照](extractor_diagnosis/index.html) · [结构化摘要](extractor_diagnosis/SUMMARY.json) · [未准入说明](EXTRACTOR_LABELS_NOT_ADMITTED.json)。

输入/权重核验通过，ASR能对应实际说话内容。单独加输出长度或关闭repetition penalty均不能修复；补2.4s时长让8/8通过格式，但8个“0–0.8s语音＋0.8–2.4s静音”样本全部输出1.39–1.70s的韵律，预测区间RMS严格为0，整段能量仍高于官方静音门限。由此确认格式通过不能代表标签正确；当前提取器时间定位不可靠。

仅执行有界诊断并已结束，没有部署新prompt/生成参数或新的监督定义，没有恢复全量。恢复前需重新通过静音、混合静音、时移及人工抽听定位验收；下面的启动/恢复命令是历史操作说明，不能作为当前提取器已获准恢复的依据。

## 最新提速：不可修复的错误前缀早停（2026-11-10，已部署）

用户要求继续提速后，已进行两轮同输入GPU7对照，实际采用 **batch16＋不可修复错误前缀早停**。其他GPU任务保持不动，原始音轨、J pair边界、模型权重、prompt、FP16及正常生成上限64保持不变。

- [采用验收](capacity_probe/ACCEPTED.json)：64条真实、按输出状态分层的音频块，包含19条合法原回答和45条可提前判错回答。原生两次各14.454／14.446秒，候选6.907秒，配对约2.09倍；合法及未早停回答的raw/token/规范化/审核结果相同，全部64条准入判定一致。
- [完整对照](capacity_probe/RESULT.json)、[首轮无稳定收益的尝试](decode_probe/RESULT.json)：仅last logits、裁掉masked尾部没有稳定收益；batch32/64改变部分生成，早停64也不及早停16，因此未部署。原环境本身已经使用SDPA。
- [生产执行验收与快照](EARLY_EXECUTION_VALIDATION.json)、[44项相关测试](EARLY_EXECUTION_TESTS.log)。生产新阶段已连续67个batch16，稳定约8.1块/秒；过去阶段约3块/秒。GPU共享负载变化使跨阶段比值不能全部归因于该优化，隔离配对收益以上述2.09倍为准。
- [前缀证明与早停模块](../../src/glare_prosody_prefix_reject.py)：只在当前文本已不可能扩展为合法数值行或独立NULL时停止。已完成的时间字段如“2.34, 2.79,”超过2.4秒范围，后缀不能修复；拒答字母前缀和转义换行也属于已不可修复格式错误。未完成数字不能按已完成数字误判；UTF8替换字符、正则超时均保守继续生成。

早停通过LogitsProcessor仅为已判错的行强制EOS。**这些失败回答只保存实际生成前缀，不再声称有完整官方后缀**；收据标记 `early_reject`、原因、`forced_eos`和`full_official_reply_available=false`，并绑定验收/source SHA。合法回答不因此提前停止；写入前再次证明前缀非法，禁止把早停结果准入或变成有效零监督。原缓存继续复用。

这是新增的执行适配，不能称为所有回答均完整运行官方离线生成。新旧“refusal/truncated/parse_error”分类数量也不能直接比较：旧流程继续生成至拒答全文或64token，早停流程在不可修复前缀处分类，新增 `early_rejected_chunks` 单列。数据准入规则相同；没有放宽错误、时间或分数规则。

控制器自动加载验收文件，执行器核对当前prefix源码、原提取契约及64条报告SHA后才启用。源码或报告变化会拒绝运行，需重新验证。现阶段仍只有极少数合法NULL，尚无证据表明韵律事件产出质量已解决；提速不代表可用训练pair数量增加。

## 最新执行：已按用户要求切换至全量（2026-11-10）

原LIA队列已正常停止，保留19,609条完成缓存，执行失败0；原音频阶段284,218条已完成，均不删除。原STOP保留。原status_all已标记为用户切换，不应把未完成误认为新增执行失败。

全量范围仍为冻结284,218条train/dev、500.001259h候选池；不是把500h全部用于train。来源索引与标签子集保留率以最终输出为准。

**当前已进入GPU提取：41,433条speaker音轨、946,304个去重源块，node11 GPU7。** [最新执行验收与快照](EXECUTION_VALIDATION.json)、[部署信息](FULL_RUN_INFO.json)。跨源合批将理论批次从80,056减少至59,144（少26.12%，不是已经测得总耗时降低26.12%）。初期稳定推理约2.9块/s，按单卡当时负载粗估约4天；启动加载与后续负载会影响实时ETA。有效回答比例目前偏低，提取进度不等于可训练pair数。

- [全量控制状态](full/controller_status.json)：来源索引完成后自动启动GPU提取，完成后自动导出封存清单。
- [全量提取状态与实测吞吐](full/status.json)：运行后含已完成块、真实有效率、batch分布、CPU预读/GPU推理/等待时间和ETA。
- [全量协议](full/protocol.json)、[最终清单入口](full/LATEST.json)。后两者在相应阶段完成后生成。
- [全量执行器](../../scripts/run_glare_source_prosody_full.py)、[后台控制器](../../scripts/control_glare_source_prosody_full.py)。
- [执行加速测试](EXECUTION_TESTS.log)：跨源批处理、逐样本PCM一致、限量/全量恢复、读源失败与异常传播。

加速只改执行调度：跨音轨凑满原生batch16，两路CPU预读、64块有界队列；每块保持原source-grid位置及输入PCM，官方FP16/greedy/max64/prompt不变。最多3条来源波形驻留，单条16k float32波形上限512MiB；不额外分配其他GPU。来源JSON索引核验采用4路有界预读，同一原始JSON读取一次兼做SHA和解析。

全量核验发现微秒结束时间与48k采样点时长之间存在约数微秒差异。现已改为核对 `duration == (round(end*sample_rate)-round(start*sample_rate))/sample_rate`（容差1微秒），同时仍核对原冻结row/job/clip身份。这是修正校验过严，不改J切分边界或持续时间；新增测试确认真实样本可通过，1毫秒真实错位仍拒绝。此前小试的VALIDATION属于历史版本；新全量协议绑定修正后源码，不混用旧身份。

后台任务继续仅用node11 GPU7，其他任务不停止。新提取器同时占有原run_all/controller/GPU7锁与本契约mutation锁，防止原队列自动恢复后并发写入。

预热曾受24GiB进程上限触发OOM，并暴露异常traceback持有失败批次张量、递归缩批仍无法释放的问题。已修正为离开except后GC/释放缓存再缩批，并让后续批次沿用成功批量；单块持续OOM直接报系统错误，不据此批量剔除。PackedRuntime在保留至少16GiB物理余量的前提下将上限提高至32GiB，启用expandable allocator，batch16连续实测通过。48条临时OOM记录全部归档并重提取，不保留为音质/韵律排除。含OOM释放验证在内的37项相关测试通过，见EXECUTION_TESTS.log。全量只提取及筛选韵律，不自动启动LIA或训练。

停止新全量：在`full/STOP`创建文件，或给其controller PID发SIGTERM，worker完成当前batch后退出。恢复时移除自己设置的`full/STOP`，在原node11环境运行 `python scripts/control_glare_source_prosody_full.py --plan experiments/glare_current500h_prosody_subset_20261110/full`；完成块自动复用。不要移除原shared-cache目录STOP，也不要恢复原LIA全池队列。

## 准入契约

| 原始结果 | 本次 GLARE 子集处理 |
|---|---|
| 合法韵律事件 | 保留，映射为25fps强度 |
| 原始回答独立、明确的 `NULL` | 保留，已知零强度 |
| 合法数值但全部低于官方0.5阈值 | 保留，单独记 `below_threshold`；不计为显式NULL |
| 拒答、混合NULL、格式错误、越界、分数越界、区间重叠、输出截断 | 排除本次子集，无有效零标签 |
| 输入读取、临时IO、CUDA OOM | 至多3次；OOM先递归减批至1；耗尽后独立记录，可显式重试 |
| 模型资产、API、来源身份或缓存完整性错误 | 整个运行报错退出，防止把系统错误当作大批坏数据 |
| 尚未提取 | `pending`，不算失败 |
| 原完整音轨末尾不足2.4秒 | `uncovered_source_tail`，涉及该尾部的pair暂不入选；不是模型语义失败 |

每个pair所需的所有源块都必须通过；一个块失败就不能把该pair当成完整韵律条件样本。合法事件的空白位置才填零，失败与未知没有有效强度目标。格式通过仍需抽听，不能宣称模型定位已经人工验收。

输入取 `source_result.clips[...].normalized_audio.speaker`，并核对 participant 原始媒体身份；selected row 上的 `source.normalized_audio` 是 listener，不能使用。这里沿用既有全音轨恒定增益/LUFS WAV，再执行官方全音轨重采样及RMS归一化，属于明确记录的数据输入适配，并非声称使用未经处理的原PCM。

以原音轨0秒为原点，每2.4秒一个完整块；仅提取当前清单需要的去重块，不把原视频整段画面解码或导出。普通不足2.4秒的pair可复用其所在源块；只有原音轨真正的不足整块尾部没有官方支持。跨块事件先在源时间上裁到pair，以Decimal相减后只转换一次局部浮点，不再二次两位小数舍入；采用原生floor/ceil、后事件覆盖的25fps栅格。保持原J边界、caption、train/dev及group，test不进入索引。

官方配置固定：2.4s、16kHz、RMS −20dBFS、peak0.98、无min-RMS直接NULL门控、Qwen2-Audio7B、FP16、greedy、max_new_tokens64、batch最多16。保存完整原始回答、官方规范化结果、逐行EOS/截断证据、音频编码与特征合并验证、实际输入PCM SHA和资产SHA。

官方代码存在循环变量写`audios`但实际始终传`audio=`的API兼容问题；独立适配器只按安装版本将该关键字转为`audios=`。原官方源码未修改。严格准入也不采用旧解析器的格式修补，不能把该筛选适配称为原样完全复现论文数据管线。

## 当前验证

最终验证结果写入 [VALIDATION.json](VALIDATION.json)。已准备 [pilot320/protocol.json](pilot320/protocol.json)：320 pair、319条speaker音轨、1,122个源块，约0.538987h，319 train＋1 dev。采样按 `sha256(137:sample_id)` 取最小值，4条canary是同seed的嵌套子集。

首轮真实GPU7验证 [canary4](canary4/)：14块全部返回无效结果，5拒答、9截断，4个pair均被排除。恢复运行复用14条收据，新增推理为0。修订收尾校验后的最终版本在 [canary4_final](canary4_final/) 再次验证，最终统计以VALIDATION为准。这些极小样本不能估计全量保留率。旧whole-pair试验的9条有效缓存不与新源网格混用。

最终版本复测仍为5拒答、9截断，4个pair均排除；14块断点恢复新增推理0。26项测试通过，原冻结缓存协议核验通过。可查看[8块原始回答及模型输入音频](review_canary4_final/index.html)。

320条仅建好清单，未启动其独立GPU提取；assemble复用了上述4条小试的失败收据，当前4条排除、316条待处理，不是320条已完成。全量未启动。既有LIA／音频／reaction／statistics缓存和正在运行的原队列保持原样。正式训练器尚未接入本目录新强度缓存；`full_glare_training_admitted`保持false。

## 使用与恢复

在项目根目录，CPU建表/导出可用本地Python。提取及音频复核使用node11已有环境，不能升级当前共享依赖：

```bash
export CUDA_VISIBLE_DEVICES=7
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD/experiments/glare_current500h_nonprosody_cache_20261110/runtime_deps:$PWD/compare/.envs/glare-deps:$PWD/src:$PWD/scripts"
GLARE_PY="$PWD/compare/.envs/avatarforcing/bin/python"

# 默认仅构建320条；同一参数重跑验证后直接复用。
"$GLARE_PY" scripts/prepare_glare_prosody_subset.py plan
# 仅处理最多16个未完成源块；不是16个pair。
"$GLARE_PY" scripts/prepare_glare_prosody_subset.py extract --limit 16
# 处理当前320条清单的全部依赖块；不是原500h全量。
"$GLARE_PY" scripts/prepare_glare_prosody_subset.py extract --all
"$GLARE_PY" scripts/prepare_glare_prosody_subset.py assemble

# --out必须是新目录；导出声音和原始回答供抽听。
"$GLARE_PY" scripts/review_glare_prosody_subset.py \
  --plan experiments/glare_current500h_prosody_subset_20261110/pilot320 \
  --out experiments/glare_current500h_prosody_subset_20261110/review320 --limit 32
```

在plan目录新建`STOP`可让提取器完成当前batch后退出；恢复前人工移除自己设置的STOP。正常重跑不重复推理任何已完成终态块；只有`--retry-execution`会归档并重试耗尽的临时执行错误，不重试语义错误。源音轨每次最多驻留一条，不积压全池波形。

`cache/<contract_sha>/chunks/`保存源块原始回答/强度/收据；`pairs/`保存通过pair的局部网格强度。缓存绑定模型、官方prompt、数值代码、映射及原音轨身份；pilot和全量成员名单不属于提取契约，**相同代码契约下可以复用**。共享`mutation.lock`保护跨plan写入；原子写入、回读和SHA核验，异常退出不产生已完成假收据。

每次assemble产生独立 `exports/<timestamp>/`，包含 `retained_all.jsonl`、`retained_train.jsonl`、`retained_dev.jsonl`、`excluded.jsonl`、`pending.jsonl`、`SUMMARY.json`和`SEAL.json`；[pilot320/LATEST.json](pilot320/LATEST.json)指向最新完整导出。保留清单保持原行字段，追加`glare_prosody`指针。SUMMARY报告实际保留/排除/待处理小时数、分split时长及来源/长度分布，未解析数据不混进保留率分母。

将来启动原池全量时，必须另建plan并显式`plan --full`；先完成几百条的保留率、质量和耗时检查。本次没有执行该步骤。

## 入选后补齐特征

显式指定已完成导出路径与独立work目录：

```bash
python scripts/run_glare_prosody_subset_cache.py prepare \
  --plan /absolute/path/to/plan --export /absolute/path/to/plan/exports/version \
  --work /absolute/path/to/new_feature_work

# 仅在原全池队列已按调度安排退出后执行；不会自动停原任务。
"$GLARE_PY" scripts/run_glare_prosody_subset_cache.py run \
  --work /absolute/path/to/new_feature_work
```

默认拒绝含pending的导出；诊断小试可显式`--allow-partial`。拒绝空保留集。prepare重新核验每条韵律收据、冻结行及导出seal。run同时独占原 `run_all.lock`、`controller.lock`和`workers/gpu7.lock`；原队列占用时直接拒绝。不会清STOP、停其他进程或自动切换生产队列。

执行沿用已验证fused2 FP32、32个CPU解码、32路检测、64在途组，仅保留原组内入选ordinal，原gid和时间不变；payload写回原shared cache的规范绝对路径，已有合格缓存由原stage检查器复用。新work只承载范围、状态和错误。训练前仍需整合新强度读取、抽听验收、split泄漏核验和正式训练配置冻结；本入口不自动训练。
