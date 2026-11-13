# 英文论文初稿（含中文作者备注）

本版依据[中文主稿](../中文论文初稿_引言与任务数据.md)，使用你提供的CVPR 2027模板。已确定的内容写为英文；待做计划、待核实关系、实验结果与量表要求保留中文蓝色备注。**这是写作工作稿，不是已完成实验或可直接投稿的版本。**

**2026-11-08引言更新：** 以作者手改英文稿为准完成逻辑与引用整理，保留其应用理由和模型优先贡献顺序。见[引言修订说明](review/引言修订说明_20261108.md)及[CITATIONS中的逐段引用定位](CITATIONS.md#2026-11-08-引言引用定位)。中文主稿及其他英文正文未改动。

## 先看这些文件

- [PDF预览](main.pdf)：双栏排版，英文正文、中文备注、草图和表格结构。
- [主入口](main.tex)：按章节加载正文；在Overleaf中选它作为主文件。
- [含义与证据核对](review/含义与证据核对.md)：列明含义处理、待确认技术关系、主张—证据映射与五维自查。
- [引用替换清单](CITATIONS.md)：27个稳定引用键及原文定位。
- [反向段落提纲](review/段落反向提纲.md)：各英文正文段落各自承担什么任务。
- [可上传的源码包](english_draft_source.zip)：自包含LaTeX源码、矢量图及作者说明，不含历史实验媒体。

## 如何继续改

| 内容 | 修改位置 |
|---|---|
| 引言 | `draft_sections/01_introduction.tex` |
| 相关工作 | `draft_sections/02_related_work.tex` |
| 任务／数据 | `draft_sections/03_task.tex`、`04_data.tex` |
| 方法 | `draft_sections/05_method.tex` |
| 评价／实验 | `draft_sections/06_evaluation.tex`、`07_experiments.tex` |
| 摘要／结论 | `00_abstract.tex`、`08_conclusion.tex`，当前中文待写 |
| 实验表格 | `tables/T1_data.tex`、`T2a_av.tex`、`T2b_visual.tex`、`T3_ablation.tex`、`T4_human.tex` |
| 标注质量附表 | `tables/T_quality.tex`，可移至附录 |
| 正式BibTeX | `references.bib`，替换占位条目时保留同一key |
| 中文备注样式 | `draft_preamble.tex`中的`\cnnote{标题}{内容}` |

相关工作按你本轮确认的新结构排列：**视听对话与回应生成／MLLM级联生成／数据与反馈评价**。前两段原有任务综述合并进2.1，2.2尚无写定论点，保留中文待补。

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
