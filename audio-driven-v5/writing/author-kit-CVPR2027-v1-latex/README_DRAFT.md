# 英文论文初稿（含中文作者备注）

本版依据[中文主稿](../中文论文初稿_引言与任务数据.md)，使用你提供的CVPR 2027模板。已确定的内容写为英文；待做计划、待核实关系、实验结果与量表要求保留中文蓝色备注。**这是写作工作稿，不是已完成实验或可直接投稿的版本。**

**2026-11-08方法与最终结构更新：** 按作者确认，采用“方法第3章、数据独立第4章、实验第5章”。方法已根据总索引指向的当前配置及实际源码重写为完整英文初稿，覆盖 **AV＋Whisper＋Feature＋LTX整段Event**，补入条件构建、注意力与损失公式。见[方法重写与实现核对](review/方法重写与实现核对_20261108.md)。中文主稿和作者最新引言未改动，实验结果保持待填。

此前的[引言修订说明](review/引言修订说明_20261108.md)及[CITATIONS中的逐段引用定位](CITATIONS.md#2026-11-08-引言引用定位)仍记录引言的写作依据。本轮保留作者最新的简洁数据段及红色备注，只将贡献中的“角色筛选”统一为“按角色组织配对”。

## 当前正文结构

1. **Introduction**：任务价值、已有能力、挑战和三条贡献。
2. **Related Work**：视听对话与回应生成／MLLM级联生成／数据与评价。
3. **Method**：任务定义与总览／直接语音条件／目标特征监督／整段事件监督／训练与推理。
4. **Listening Data Construction**：角色配对与媒体处理／辅助线索标注／数据规模。
5. **Experiments**：实验设置／评价协议／定量比较／定性与人评／五组消融／数据与评价核验。
6. **Conclusion**：证据支持的总结与简短局限，正文待结果完成后写。

这里借鉴的是参考论文的内容分工，并非声称 CVPR 统一规定六章。采用类似INFP的“方法→独立数据→实验”次序。方法开头先定义训练样本提供哪些监督，构建过程在第4章展开；整段事件指标在5.2集中定义。

## 先看这些文件

- [PDF预览](main.pdf)：双栏排版，英文正文、中文备注、草图和表格结构。
- [主入口](main.tex)：按章节加载正文；在Overleaf中选它作为主文件。
- [含义与证据核对](review/含义与证据核对.md)：列明含义处理、待确认技术关系、主张—证据映射与五维自查。
- [引用替换清单](CITATIONS.md)：28个稳定引用键及原文定位，新增Bernini代码复用来源占位。
- [反向段落提纲](review/段落反向提纲.md)：各英文正文段落各自承担什么任务。
- [方法重写与当前目录](review/方法重写与实现核对_20261108.md)：五个方法小节、公式的实现依据、待作者确认项及主张—证据对应。
- [补充材料待补内容清单](../补充材料待补内容清单.md)：逐段精简时保存的数据处理参数、标注规则、模型实现及待补核验；是写作备忘，英文补充材料尚未完成。
- [三种候选表格与预览](review/table_candidates_20261109/README.md)：数据资源比较、生成方法比较、本文数据统计；caption 均在表格下方，待核事实保留 TBD。
- [上一版结构调整说明](review/行文结构调整_20261108.md)：历史方案记录；当前以本页和方法重写说明为准。
- [可上传的源码包](english_draft_source.zip)：自包含LaTeX源码、矢量图及作者说明，不含历史实验媒体。

## 如何继续改

| 内容 | 修改位置 |
|---|---|
| 引言 | `draft_sections/01_introduction.tex` |
| 相关工作 | `draft_sections/02_related_work.tex` |
| 方法（第3章） | `draft_sections/05_method.tex`；其中3.1加载任务定义`03_task.tex` |
| 数据构建（第4章） | `draft_sections/04_data.tex` |
| 实验（第5章） | `draft_sections/07_experiments.tex`；其中5.2加载`06_evaluation.tex` |
| 摘要／结论 | `00_abstract.tex`、`08_conclusion.tex`，当前中文待写 |
| 实验表格 | `tables/T1_data.tex`、`T2a_av.tex`、`T2b_visual.tex`、`T3_ablation.tex`、`T4_human.tex` |
| 标注质量附表 | `tables/T_quality.tex`，可移至附录 |
| 正式BibTeX | `references.bib`，替换占位条目时保留同一key |
| 中文备注样式 | `draft_preamble.tex`中的`\cnnote{标题}{内容}` |

相关工作按你本轮确认的新结构排列：**视听对话与回应生成／MLLM级联生成／数据与反馈评价**。前两段原有任务综述合并进2.1，2.2尚无写定论点，保留中文待补。

文件名前缀沿用原稿，**不代表排版中的章节号**。`03_task.tex`和`06_evaluation.tex`分别由方法与实验文件嵌套加载，勿再在`main.tex`重复加载。本轮调整前的文件、PDF和源码包保存在`review/method_revision_20261108/`；更早的结构备份仍在`review/structure_revision_20261108/`。

原模板保存于[template_original](template_original/README.md)。`cvpr.sty`和`ieeenat_fullname.bst`没有修改。根目录原有的`preamble.tex`、`main.bib`和`sec/`是模板示例文件，当前入口不加载它们；不要将示例引用当作本文引用。

## 图的替换

| 筹备图号 | 当前文件 | 状态 |
|---|---|---|
| F1 | `figures/F1_task.pdf` | 已有任务示意；人物、话语和波形非实测 |
| F2 | `figures/F2_data.pdf` | 已有数据／标注流程；示例线索与caption非实际样例 |
| F3 | `figures/F3_method.pdf` | 已有方法图；保留训练／推理描述来源及监督位置 |
| F4 | `figures/F4_evaluation.pdf` | 本轮绘制的三维评价示意；事件时间只作说明 |
| F5 | `figures/F5_results_layout.pdf` | 本轮绘制的空布局；六行×三类案例，未填任何模型效果 |

替换同名PDF即可保持LaTeX引用；每图也有SVG与PNG。F1–F3原绘图脚本保留在上一级`writing/figures/`；F4／F5绘图脚本为`figures/draw_evaluation_layout.py`，依赖同目录`figure_helpers.py`、CairoSVG、Pillow和PyMuPDF。只有重新绘图时才需Python，编译论文只需PDF。

F1–F5、T1–T4是筹备名称；正式编号由LaTeX自动生成。T2分为两张正式表，另有标注质量附表，因此PDF目前有6张表。所有性能数值为TBD，N/A只表示不适用；不得把草图当作结果证据。

## 编译

```bash
cd /private/liqi_workspace/project_code/audio-driven-v5/writing/author-kit-CVPR2027-v1-latex
latexmk -xelatex main.tex
```

需XeLaTeX、BibTeX、latexmk及xeCJK。中文优先使用Noto Serif/Sans CJK SC；若不存在则回退到TeX Live的Fandol字体。Overleaf编译器请选择 **XeLaTeX**。本地已实际编译并检查PDF。

目前保留中文备注、完整方法细节和较大的草图，**未进行投稿页数压缩**。不通过改页边距或缩小正文来压缩；后续按结果和作者取舍精简段落、图注及附录。隐藏作者备注并不能自动变成可投稿成稿，实验、引用、摘要、标题与结论仍需完成。

## 使用的写作技能与核对

已按要求安装并应用[Research-Paper-Writing-Skills](https://github.com/Master-cai/Research-Paper-Writing-Skills)，路径为`/private/liqi_workspace/.codex/skills/research-paper-writing/SKILL.md`，下次对话可以直接调用。采用了段落主旨、反向提纲、证据对应及表格规范；没有采用会增加新动机、结论或实验范围的建议。

[验证记录](review/VALIDATION.json)记录编译、引用、图表、中文备注和源稿未改动检查；[PDF检查预览](review/)用于本轮排版核对。
