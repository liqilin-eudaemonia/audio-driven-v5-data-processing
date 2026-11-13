# 公开权重：ViCo test43 统一 EMOCA 实测

DyStream 和 Avatar Forcing 使用官方公开预训练权重，没有用本轮500h重训。两者均采用listener音轨全零的适配；本表Avatar Forcing的speaker视觉条件为零。真实speaker视频条件的新版另见独立流程。

固定43条，每条seed137一次采样；四个模型共用16238个有效评分帧，候选16245帧。

| 方法 | FD Exp ↓ | FD Pose ↓ | P-FD Exp ↓ | P-FD Pose ↓ | MSE Exp ↓ | MSE Pose ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Ours / joint8000 | 25.892533 | 0.087136 | 27.166891 | 0.087986 | 0.652535 | 0.016007 |
| Ours / joint8000 + SFT2752 | 24.339086 | 0.061817 | 26.074124 | 0.062967 | 0.711805 | 0.012427 |
| DyStream / public weights / listener adaptation | 31.715231 | 0.067969 | 33.394710 | 0.069342 | 0.807346 | 0.013097 |
| Avatar Forcing / public weights / zero speaker visual | 37.633272 | 0.068397 | 39.178973 | 0.071590 | 0.897029 | 0.014181 |

| 方法 | SID Exp | SID Pose | Var Exp | Var Pose | rPCC Exp ↓ | rPCC Pose ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Ours / joint8000 | 4.550953 | 3.524593 | 0.841254 | 0.014012 | 0.010388 | 0.034934 |
| Ours / joint8000 + SFT2752 | 4.648886 | 3.669342 | 0.854602 | 0.010620 | 0.006002 | 0.030621 |
| DyStream / public weights / listener adaptation | 4.396098 | 3.335322 | 0.614315 | 0.009441 | 0.030032 | 0.004435 |
| Avatar Forcing / public weights / zero speaker visual | 4.303114 | 3.678300 | 0.796301 | 0.010247 | 0.042412 | 0.038954 |

GT逐系数方差：Exp 0.904523，Pose 0.012038。

FD/P-FD/MSE逐片段等权；SID沿用DIM的GT拟合K40/20；rPCC沿用其拉平相关差；Var采用用户确认的逐系数跨帧方差再平均，原发布代码拉平方差留在JSON。
这些是本地共同读出实验，不是原论文表格复现；训练集、输入条件和renderer不同。引用文献数值应另列来源，不据此声称严格同协议领先。

[全部指标CSV](metrics.csv) · [原始JSON与逐条覆盖](RESULTS.json) · [验证](FINAL_AUDIT.json) · [入口](README.md)
