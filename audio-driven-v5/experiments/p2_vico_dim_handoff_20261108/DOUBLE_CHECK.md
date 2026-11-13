# ViCo / DIM 对齐：需要你 double-check 的事项

当前工程状态：**EMOCA提取、两版模型test43评分和数值复算已完成；与DIM发表表格的完整等价性尚未确认。** 本清单只针对“引用DIM已发表基线＋按其口径测我们”，不涉及七类事件人工标注，也不要求先重训Learning2Listen。

我在本轮发现并核验了已有的完成产物；此前根据旧审计表把这些工程误报为未完成，现以[本轮验收](VERIFICATION.json)和[原始实测](../p2_vico_literature_protocol_20261108/RESULTS.md)为准。

## 优先核对：需要原作者或其执行材料回答

你不必亲自审代码。以下每项给出了可向DIM作者／有原始复现材料的同事询问的问题；拿到对应文件或明确说明后即可交给我核查。单纯“用了ViCo”不足以确认相同设置。

### 1. 实际测试清单和每条评分范围

- **请确认：** DIM Table1最终用了哪些listener ID、各评分了多少帧？是否保留官方test43全部listener？同一speaker音频对应多个listener时如何处理？音频／动作取最短长度后，5–1024帧筛选具体排除了哪些片段？
- **最好取得：** 实际test manifest，至少包含listener UUID、speaker ID、有效起止帧和最终长度；或当时的预处理测试特征／数据加载日志。
- **我们已做：** 固定官方43个独立listener UUID；同源audio-key覆盖＋视频长度筛选的诊断得到41条加载记录、39个独立listener，但没有作者HuBERT长度，不能断言这就是作者实际范围。
- **证据：** [测试清单](../p2_vico_literature_protocol_20261108/TEST_MANIFEST.json)、[加载范围诊断](../p2_vico_literature_protocol_20261108/LOADER_SCOPE_DIAGNOSTIC.json)；[官方映射代码](../../compare/Dyadic-Interaction-Modeling/code/vico_preprocessing.py)、[加载器](../../compare/Dyadic-Interaction-Modeling/code/dataset/data_loader.py)。
- **确认后能做：** 在已有特征上按正确名单／区间重新评分；若涉及另一listener或未提取帧，再补对应提取。当前不按分数选择测试子集。

### 2. GT动作特征和EMOCA预处理——最重要的差异定位

- **请确认：** Table1的GT是否直接由`EMOCA_v2_lr_mse_20`生成？具体checkpoint、配置、裁脸方式、输入视频分辨率／帧率、漏脸处理、系数选择和任何归一化是什么？
- **最好取得：** 作者用于Table1的GT EMOCA特征。其次是完整提取命令＋配置＋checkpoint标识／哈希；有官方GT特征时可直接定位差异，或在获得许可和匹配证据后使用同一参考特征。
- **为什么必须查：** 我们按发布代码计算GT表情Var为1.45611，论文为0.93；pose Var为0.02327，论文为0.01。这是参考侧的差异，不能靠换我们的生成模型解决。
- **已排除一种解释：** 用全部43条GT的所有16,288帧、不排除任何生成漏脸帧，仍得到exp Var=1.45625、pose Var=0.02328。不是那7帧生成漏检造成了这处主要差距。
- **证据：** [GT统计锚点](../p2_vico_literature_protocol_20261108/GT_ANCHOR.json)、[本轮GT范围诊断](GT_SCOPE_DIAGNOSTICS.json)、[严格加载的权重信息](../p2_vico_literature_protocol_20261108/ENCODER.json)、[我们冻结的提取设置](../p2_vico_literature_protocol_20261108/EXTRACTION_PROTOCOL.json)。
- **确认后能做：** 按有依据的提取设置补跑、核对GT统计，再评分。不会通过调缩放或挑裁剪设置强行凑论文数值。

### 3. 单次生成还是10次择优

- **请确认：** Table1是否对每段采样10次，按GT的完整56维Fréchet距离选一次，再用同一候选报告所有指标？还是采用单次、跨次平均或其他选择？所有基线是否一致？
- **最好取得：** 发表时的评测命令、配置、脚本commit和候选选择说明。
- **我们已做：** 两版均复用事先封存的seed137单次输出；没有按GT择优。发布tester确实存在10次择优，但尚无证据证明它对应发表主表。
- **证据：** [发布tester](../../compare/Dyadic-Interaction-Modeling/code/x_engine_pt.py)、[固定执行协议](../p2_vico_literature_protocol_20261108/protocol.json)。
- **确认后能做：** 若确实需要不同采样预算，再明确方案并补生成；目前无需盲目重跑10遍。

### 4. speaker和时间对齐到底采用哪条路径

- **请确认：** 论文rPCC、P-FD评分使用真实speaker动作吗？listener是否去首帧，以listener第1至T−1帧对应speaker第0至T−2帧？哪些配置对应论文主表？
- **为什么要查：** 发布默认加载器把speaker动作设成全1常量，而常量speaker的Pearson相关系数未定义；这不能解释论文有限的rPCC结果。
- **我们已做：** 提取真实speaker动作，只提供给评分器；沿发布tester采用上述一帧偏移。speaker动作未新增为我们生成模型的条件。
- **最好取得：** 主表评测入口、配置、代码版本；或作者对真实speaker及时间索引的明确说明。
- **证据：** [加载器](../../compare/Dyadic-Interaction-Modeling/code/dataset/data_loader.py)、[tester](../../compare/Dyadic-Interaction-Modeling/code/x_engine_pt.py)、[协议差异矩阵](../p2_vico_literature_protocol_20261108/ALIGNMENT.md)。

### 5. 指标汇总和“原生系数／视频反提取”

- **请确认：** 发表表格是否直接使用当前发布`mymetrics.py`／`eval_utils.py`？SID是逐视频平均还是全数据合并直方图；Var是否把帧与系数全部拉平？各方法的动作是直接输出系数，还是渲染视频后统一重新提取？
- **最好取得：** 对应评分脚本／版本及模型输出来源说明。SID依赖的scikit-learn版本也应记录。
- **我们已做：** FD／P-FD／MSE逐视频等权；SID／Var／rPCC沿发布实现；两版均从视频反提取。全部43条上与原作者函数的16个汇总数完全一致，但这只能证明“与发布函数一致”。
- **剩余影响：** 原生动作系数与视频反提取存在不同误差来源。即便提取器名称一致，也不能自动视为完全相同的测量过程。
- **证据：** [原函数一致性核验](../p2_vico_literature_protocol_20261108/EMOCA_METRIC_PARITY.json)、[原作者指标代码](../../compare/Dyadic-Interaction-Modeling/code/mymetrics.py)、[SID实现](../../compare/Dyadic-Interaction-Modeling/code/metrics/eval_utils.py)。

## 你与合作者可以直接核对的两项

6. **确认论文最终使用哪版我们的模型。** 当前已经测了同一LTX联合8000步的base-Qwen和SFT2752两版。请按研究设计、验证集和实际模型说明确定主表／消融角色，不根据ViCo test成绩临时挑版本。两版均有完整结果，可同时保留并清楚命名。

7. **确认表格结论的措辞。** 目前允许写“基线结果引用DIM等论文；我们采用指定EMOCA及DIM公开指标补测，存在已披露差异”。当前不应写“严格相同设置下全面超过某方法”，也不计算跨来源提升率。最直接的证据是GT统计尚未复现。DIM原版L2L、CANDOR预训练L2L、INFP原版及Avatar Forcing的INFP*必须分行，不能混作同一结果。

## 建议先索取的最小材料包

优先向作者要：**①实际test ID／有效帧范围，②GT动作特征或完整EMOCA提取配置，③主表评测命令／配置（包括采样选择）。** 这三项比索取生成权重更直接；我们当前只补测自己的输出，不需要先运行DIM生成器。

若无法取得上述材料，工程结果仍然有效，但结论保持“按公开实现补测，严格原表等价性未确认”。不能将作者未回复自动当成相同设置。

## 可直接转发的问题草稿

> We are evaluating our RGB listener videos on ViCo and would like to compare them with DIM Table 1 using your released EMOCA/metric pipeline. Could you share (1) the effective test listener IDs and scored frame ranges, including duplicate audio-key handling and length filtering; (2) the GT EMOCA features, or the exact checkpoint/configuration and extraction command; and (3) the Table 1 evaluation command/configuration, including whether the released 10-sample GT-based selection was used? We would also appreciate clarification of the real-speaker input for rPCC/P-FD, the one-frame speaker/listener offset, SID/variance aggregation, and whether each baseline was scored from native coefficients or RGB re-extraction. Our GT variance under the released functions is 1.45611 (expression) / 0.02327 (pose), compared with 0.93 / 0.01 in Table 1, so we would like to resolve the reference-side discrepancy before claiming matched settings.

此处只提供草稿，没有替你发送邮件、issue或其他消息。
