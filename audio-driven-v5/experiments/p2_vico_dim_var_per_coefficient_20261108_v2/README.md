# ViCo：用户指定的第二种Var定义

用户要求采用“各系数分别跨评分帧计算方差，再对系数取平均”。本次已对GT、联合8000/base-Qwen和联合8000/SFT2752统一计算，结果见[RESULTS.md](RESULTS.md)、[完整JSON](RESULTS.json)、[Var CSV](variance.csv)及[输入／数值审计](AUDIT.json)。

公式为`mean_j Var_t(x[t,j])`，总体方差`ddof=0`。先拼接所有片段的实际评分帧，再按50维表情／6维姿态分别计算；不是逐视频方差的等权平均。保持原43条、16,238个共同有效帧和其他五项指标不变。

| 来源 | Var Exp | Var Pose |
|---|---:|---:|
| GT | 0.904523 | 0.012038 |
| 联合8000/base-Qwen | 0.841254 | 0.014012 |
| 联合8000/SFT2752 | 0.854602 | 0.010620 |

这是用户选择的明确评测定义，不能因更接近DIM发表GT的0.93／0.01就证明作者也采用它。原发布代码的拉平方差保存在完整JSON的`released_variance_reference`和结果对照表，原始封存数据与成绩均未修改。

[固定配置](../../configs/p2_vico_dim_variance_per_coefficient_v1.json) · [可复算入口](../../scripts/p2_vico_dim_variance_readout.py)。CPU执行，无新生成、无新提取、无GPU分配。初版只读导出位于无v2后缀目录，本版补齐了JSON的公式／汇总说明，数值相同；当前使用本目录。

十次择优队列当前为`paused_by_user_for_31gpu_training`，本次未恢复；已将该Var选择写入其`REQUESTED_VARIANCE_READOUT.json`，供下次恢复评分时统一采用。
