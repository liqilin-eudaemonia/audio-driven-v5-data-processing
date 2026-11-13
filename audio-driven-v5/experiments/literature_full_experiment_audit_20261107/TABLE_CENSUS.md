# 表格覆盖清单与排除理由

本清单把数据／符号／训练设置／自身消融与跨方法性能比较分开，避免以“只查主结果表”作为无遗漏证明。图中和正文的其他比较另列于PAPER_DETAILS和ROW_PROVENANCE；同一作者方法的参数配置不各算一个独立模型。

| 论文 | 主PDF内全部表号及角色 |
|---|---|
| GLARE | comparison: 2/3；own_ablation_or_metric_sensitivity: 4/8/9/10/11；dataset_statistics: 1/5；detector_or_training_settings: 6/7 |
| ViCo / RLHG | comparison: 3/4/6；own_modality_architecture_ablation: 5/A1；dataset_statistics: 1/2 |
| Learning to Listen / L2L | comparison: 1；own_ablation: 2 |
| PCH / PCHG | challenge_leaderboard: 1；own_talking_task_ablation: 2/3 |
| MFR-Net | comparison: 1/2/3；own_ablation: 4 |
| ELP | comparison_or_control: 1/2/3/4 |
| CustomListener | comparison: 1/2/4；own_ablation: 3 |
| DIM | comparison: 1/2/4；own_ablation: 3 |
| ReactFace | notation: 1；comparison: 2/8/9；own_component_or_loss_ablation: 3/4/5/6/7 |
| DiffListener | comparison: 1/2；own_ablation: 3/4 |
| INFP | comparison: 1/2/3/4 |
| ListenFormer | comparison: 1/2/3/4；own_fusion_ablation: 5 |
| SpeakerVid-5M | dataset_comparison: 1；own_baseline_settings: 2；annotation_prompts: 3/4 |
| ARIG | comparison: 1/2/3 |
| DiTaiListener | comparison: 1/2/3/4/5 |
| Avatar Forcing | comparison: 1/2/3/4；own_ablation: 5/6/7 |
| ECHO | comparison: 1/2/3/4 |
| OmniResponse | dataset_comparison: 1；comparison: 2/4；own_ablation: 3 |
| DyStream | comparison: 1/2/4；own_ablation: 3/5 |
| OmniHuman-1.5 | own_ablation: 1；mixed_own_ablation_and_comparison: 2；comparison: 3/4 |
| OmniPersona | capability_claims_not_performance: 1；comparison: 2/3/4/5/13/15/16；architecture_training_conditions_or_rating_anchors: 6/7/8/9；own_probes_ablation_or_timing: 10/11/12/14/17 |

另外逐页检查的文字补充：

- DIM / embedded_appendix：own_modality_ablation 1；human_comparison 2。
- Learning to Listen / L2L / official_arxiv_appendix：per_identity_comparison 3/4/5/6；TLCC_comparison 7。
- CustomListener / official_CVF_supplement：RealTalk_comparison 5/6。
- DiffListener / official_CVF_supplement：own_modality_ablation 5；human_comparison 6。
- INFP / official_CVF_supplement：own_data_size_memory_ablation 1。
- ARIG / official_CVF_supplement：own_ablation 1；human_comparison 2。
- DiTaiListener / official_CVF_supplement_newer_than_local：long_video_strategy_comparison 6；own_CTM_ablation 7。

保留的不可识别／非文字材料：

- ELP official ZIP contains video only, no PDF/HTML/text scientific supplement
- OmniPersona AppendixA mentions unnamed concurrent-work comparison videos; no model identity can be inferred from unnamed material
- ECHO Fig3 Per-FLOAT label and raster reference numbers conflict with current bibliography; retained as unresolved identity
- OmniHuman1.5 anonymous CS-A/B/C/D and unnamed general Base Model are not mapped to guessed public models

JSON中的出现页包含正文引用页，不能当作表的唯一印刷页。DIM附录重用Table1/2编号，已按阶段另记；DiTai新版CVF补充比本地15页稿多Table6/7，未覆盖或篡改原PDF。
