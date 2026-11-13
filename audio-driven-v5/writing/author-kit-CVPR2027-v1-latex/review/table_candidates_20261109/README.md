# 三种候选表格（2026-11-09）

按作者要求，三个候选均将 caption 放在表格下面。当前正文仍加载原 `tables/T1_data.tex`，本轮仅把其 caption 移到两块表格之后并精简说明；没有代替作者选定候选。以下文件可独立引用或替换原表。

| 候选 | LaTeX | 内容与建议位置 |
|---|---|---|
| 1 数据资源比较 | [T1_option_datasets.tex](../../tables/T1_option_datasets.tex) | 倾听组织方式、双方独立音轨、pair 数；适合数据章节或相关工作 |
| 2 生成方法比较 | [T1_option_methods.tex](../../tables/T1_option_methods.tex) | 倾听建模范围、是否联合生成音视频、实际生成模态；适合相关工作 |
| 3 本文数据统计 | [T1_option_statistics.tex](../../tables/T1_option_statistics.tex) | 类别、逐类 train/test/total pair 数、类别数及资源总数；最适合当前数据章节 |

独立预览：[preview.pdf](preview.pdf)。预览通过 `preview.tex` 加载三张候选和现有占位参考文献，不覆盖 `main.pdf`。

## 数字与分类依据

- ViCo **483 个片段 pair**：本地 `01_RLHG_ViCo_ECCV2022.pdf` 第7页 Table 2 和 §4。
- GLARE **107,149 个片段 pair**：本地 `00_GLARE_local.pdf` 第3页 §3、第13页 Table 6。**64,557 是反应实例数，不能填入 pair 数列。**
- ResponseNet **696 个对话 pair**：本地 `50_OmniResponse_NeurIPS2025.pdf` 第6–7页 §4／Table 1；原文明确提供双方分离音轨。它们比短片段长，不直接按 pair 数和切分资源比较规模。
- SpeakerVid-5M：本地 `38_SpeakerVid_5M_2025.pdf` 第7页 §4.4 明确是 speaker audio 与 silent listener video 配对。**5.2M 单人片段和 770K dialogue pairs 都不能替代 listening 分支的 pair 数。**该分支数量本轮保留待核。
- Seamless 独立音轨、其余资源与方法定位承接[相关工作核对](../../../相关工作比较与引用依据_20261107.md)；未在本轮核定的双方音轨支持与 pair 数标为 TBD，不写成“没有”或“原文未报告”。
- OmniResponse：本地原文第5页描述文本／面部生成和 TempoVoice 音频合成；OmniPersona 第2页 Table 1 将其 Joint A&V 标为否。候选2只比较生成机制，不据此推断效果优劣。
- 本文沿用已确认的 retained resource 口径：train **704,368**、test **7,087**，所示总数为二者之和 **711,455**。依据：[最终筛选统计](../../../../experiments/listen_j_upright_response_sft_20261105/final/summary.json)。本轮没有把约500h／284,218条的 AV 训练选集混入该总数。

## 类别表怎么填写

候选3的点头、微笑和可听笑声只演示行结构，并非已经确定的完整类别表。先明确整个 retained train/test 资源要采用的类别定义，再按每个类别的正例 pair 计数。一个 pair 中同类出现多次只计一次；不同类别可以共存，因此逐类计数总和不必等于样本总数。不要把现有训练 Event 的槽位数、启用类别数或评测七类直接写成整个数据集的类别数。

正式统计应能追溯到对应 train/test 清单，说明未确定标签与未标注样本的处理。当前未重新扫描大规模数据，也未补写未经验证的逐类数字。

## 使用

选定后，在数据章把 `\input{tables/T1_data}` 替换成相应候选文件。候选使用独立 label；替换时若正文其他位置仍引用 `tab:data`，同步调整引用或把选中表的 label 改回 `tab:data`。三张候选不需要全部放进正文。

作者明确要求 caption 在下方，本次遵从该要求；模板样式文件未修改。引用键沿用 `references.bib`，正式书目信息继续按 `CITATIONS.md` 补齐。
