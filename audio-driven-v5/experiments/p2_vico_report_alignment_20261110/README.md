# ViCo report 与当前 eval config：数值对照及使用说明

**结论：部分指标接近，但尚未证实整套 report 基本对齐。** ListenFormer 本地重训的 PSNR 与论文两位小数相同，FID 高25.79%；其他方法存在数据集、权重或系数空间差异，不能用裸数值判断复现成功、失败或方法优劣。

本目录记录2026-11-10完成的核查。范围来自飞书评测文档；Avatar Forcing 按要求排除。所有本地输出均为 ViCo 官方 test43。本轮复用既有预测，重新计算指定指标，未重新训练或生成视频。

## 1. 如何读这些表

- `Report` 是论文报告值；`本地`是实际生成结果的评分，不是我们的模型结果。
- `Exp / Pose` 固定为表情／姿态顺序；`—`表示所引用表没有报告该项，不能当成0。
- ListenFormer允许作描述性差值比较，但其本地重训权重和训练划分与作者不同，不是严格复现。
- L2L、DualTalk与DyStream的报告和本地结果存在表示或数据集差异，因此只并列展示，不计算相对差、不据此排名。
- FD在第2节是BFM系数L1距离；后续FD是分布Fréchet距离，二者不是同一指标。SID/Var只列统计量，不能普遍理解为越大越好。

## 2. ListenFormer：本地完整重训与论文

本地完成200轮／173800步；351条训练、46条验证，验证集选择epoch1／869步。作者官方预训练分享已取消，本地找到的是本地重训best.pt和last.pt。

### 2.1 画质与身份（论文Table2）

| 指标 | Report | 本地test43 | 本地−Report | 相对差 |
|---|---:|---:|---:|---:|
| PSNR ↑ | 18.89 | 18.887633 | -0.002367 | -0.01% |
| SSIM ↑ | 0.62 | 0.641760 | +0.021760 | +3.51% |
| 图像FID ↓ | 24.52 | 30.842915 | +6.322915 | +25.79% |
| CSIM ↑ | 0.63 | 0.602755 | -0.027245 | -4.32% |
| CPBD ↑ | 0.17 | 0.193178 | +0.023178 | +13.63% |

PSNR/SSIM本轮使用原官方脚本重新解码43条／16,288帧计算，与归档值完全相同。FID/CSIM/CPBD引用已封存实测；本轮核验结果文件及43条视频哈希，未重跑这三项。评分256×256/30fps；PSNR/SSIM/CSIM/CPBD按帧加权，图像FID按全集计算。

### 2.2 原生动作（论文Table1，test Average）

| BFM L1 ×100 ↓ | Report | 本地逐片段等权 | 本地−Report | 相对差 |
|---|---:|---:|---:|---:|
| angle | 7.35 | 10.299858 | +2.949858 | +40.13% |
| exp | 13.36 | 12.398529 | -0.961471 | -7.20% |
| trans | 5.84 | 5.477706 | -0.362294 | -6.20% |

本轮从43份原生预测和GT重算；与归档一致。论文精确汇总流程尚未确认，保留该差异。低距离不代表反馈自然；该本地模型的视频DECA表情时序方差约为GT的3.2%，已有动作不足现象。

来源：[论文表格截图](sources/listenformer_table2.png)、[原始差值CSV](listenformer_vs_report.csv)、[官方PSNR/SSIM复算回执](LISTENFORMER_BASE_RECHECK.json)、[权重检索记录](LISTENFORMER_WEIGHT_SEARCH.json)。

## 3. DyStream：报告不是ViCo，禁止直接比较

Report来自DyStream论文Table2的RealTalk 38条、MediaPipe；本地是ViCo 43条、视频反提取EMOCA。两侧数据集和特征表示均不同。

| 指标 | Report Exp | Report Pose | 本地 Exp | 本地 Pose |
|---|---:|---:|---:|---:|
| FD ↓ | 0.074000 | 3.192000 | 31.715231 | 0.067969 |
| P-FD ↓ | — | — | 33.394710 | 0.069342 |
| MSE ↓ | 0.018000 | 1.636000 | 0.807346 | 0.013097 |
| SID | 4.586000 | 3.070000 | 4.396098 | 3.335322 |
| Var | 0.275000 | 0.596000 | 0.614315 | 0.009441 |
| rPCC ↓ | — | — | 0.030032 | 0.004435 |

本地主表使用固定共同有效帧16,238帧、seed137单次输出、listener自身音轨全零。与旧封存12项指标最大差2.24e-9。[论文Table2截图](sources/dystream_table2.png) · [原论文§4.4](https://arxiv.org/pdf/2512.24408)。

## 4. L2L：ViCo重训Report与四套公开人物权重

Report来自DIM Table1：作者在ViCo上用EMOCA重新训练L2L。本地四套官方人物权重输出DECA动作，训练数据也不同；没有公开写实renderer用于共同RGB反提取。以下仅并列数值，不代表本地优于Report。

每个单元格均为 **Exp / Pose**；本地每套43条、16,245评分帧。

| 指标 | Report / ViCo重训EMOCA | Conan / DECA | Fallon / DECA | Stephen / DECA | Trevor / DECA |
|---|---:|---:|---:|---:|---:|
| FD ↓ | 33.930000 / 0.060000 | 7.094259 / 0.099602 | 8.080833 / 0.140326 | 6.484761 / 0.118793 | 5.471074 / 0.166756 |
| P-FD ↓ | 35.880000 / 0.060000 | 7.266816 / 0.101545 | 8.268756 / 0.142928 | 6.638364 / 0.121084 | 5.623629 / 0.168365 |
| MSE ↓ | 0.930000 / 0.010000 | 0.173284 / 0.020790 | 0.199416 / 0.027740 | 0.161006 / 0.023947 | 0.138949 / 0.029540 |
| SID | 2.770000 / 2.660000 | 3.337065 / 1.949145 | 3.048272 / 2.178443 | 2.020645 / 2.185899 | 2.988023 / 0.810084 |
| Var | 0.830000 / 0.020000 | 0.030809 / 0.002957 | 0.036581 / 0.003787 | 0.025718 / 0.003160 | 0.026537 / 0.001047 |
| rPCC ↓ | 0.060000 / 0.080000 | 0.044059 / 0.075866 | 0.181860 / 0.215971 | 0.101896 / 0.162534 | 0.046347 / 0.458612 |

四套模型全部列出，未根据GT选择人物。使用当前指标数学定义和时间偏移，但保留DECA原生表示；完整数据见[重算CSV](rescored_metrics.csv)。原文依据：本地`writing/17_DIM_ECCV2024.pdf`第9–11页，明确说明EMOCA重训。

## 5. DualTalk：官方公开权重的迁移诊断

Report来自论文ViCo Table4。本地官方权重与当前HF LFS哈希一致，核心源码与当前官方发布逐字节一致；已有43条self-zero预测，本轮重新评分。

**本地是DECA迁移代理：neck与global head的几何对应关系尚未确认，公开权重是否就是ViCo Table4所用权重也未确认。不能将这一行视为严格同协议复现。**

| 指标 | Report Exp | Report Pose | 本地代理 Exp | 本地代理 Pose |
|---|---:|---:|---:|---:|
| FD ↓ | 22.270000 | 0.050000 | 66.003166 | 0.216280 |
| P-FD ↓ | 23.810000 | 0.050000 | 66.166885 | 0.218324 |
| MSE ↓ | 0.580000 | 0.010000 | 1.347036 | 0.037435 |
| SID | — | — | 1.835040 | 1.930289 |
| Var | — | — | 0.143892 | 0.005231 |
| rPCC ↓ | — | — | 0.569896 | 0.731422 |

[论文Table4](https://arxiv.org/html/2505.18096v1#S5.T4) · [本地完整结果](DUALTALK_PROXY.json) · [官方发布一致性](DUALTALK_RELEASE_PARITY.json)。论文正文提及SID，但Table4只给FD/P-FD/MSE，因此其余Report栏留空。

## 6. 当前eval config与GT锚点

主配置：官方test43；EMOCA_v2_lr_mse_20，原始pose6+exp50；listener[1:T]对应speaker[0:T−1]；FD/P-FD/MSE逐片段等权；SID在GT上拟合KMeans40/20；Var采用所有评分帧按系数求方差再平均、ddof=0；rPCC采用拉平相关系数之差的绝对值。L2L/DualTalk只复用数学定义，不改名为EMOCA结果。

| GT统计 | DIM Report | 当前共同帧集 |
|---|---:|---:|
| Var exp | 0.93 | 0.904523 |
| Var pose | 0.01 | 0.012038 |
| SID exp | 5.03 | 5.041578 |
| SID pose | 4.07 | 3.883174 |

共同16,238帧改为DyStream自身有效16,245帧时，FD Exp仅31.715231→31.705459，但SID Exp为4.396098→4.205501。由于GT聚类随帧集重新拟合，比较时必须固定帧掩码和聚类输入。GT统计接近、函数实现一致都不足以证明完整论文协议一致。

## 7. 本轮做了什么、没完成什么

| 项目 | 本轮执行 |
|---|---|
| DyStream | 129份GT/speaker/pred特征哈希核验，12项指标重算，DIM原函数16个原始统计一致性核验 |
| L2L | 172条公开人物预测核验，四套模型按当前公式重算，保持DECA单独标注 |
| DualTalk | 官方checkpoint及当前代码核验，43条预测重评分，保留迁移代理限制 |
| ListenFormer | 本地重训产物和43条视频核验；原生BFM L1及官方PSNR/SSIM重新计算 |
| 新训练／新生成 | 无 |
| Avatar Forcing | 排除 |

没有取得ListenFormer作者可用预训练副本；DIM/RLHG等仍缺本轮可用核心生成权重。资源检查详情及其他方法状态见[完整调查报告](REPORT.md)。没有设置一个事后挑选的“基本对齐”百分比阈值，也没有用test挑权重／人物或调指标尺度。

## 8. 文件导航

| 文件 | 内容 |
|---|---|
| [README.md](README.md) | 本说明和全部报告／实测数值对照 |
| [report_vs_local_all.csv](report_vs_local_all.csv) | 所有方法逐指标并列表；附可比性和数据来源 |
| [REPORT.md](REPORT.md) | 完整核查过程、结论与资源缺项 |
| [RESULTS.json](RESULTS.json) / [rescored_metrics.csv](rescored_metrics.csv) | DyStream、L2L详细结果 |
| [LISTENFORMER_COMPARISON.json](LISTENFORMER_COMPARISON.json) | ListenFormer逐项差值及逐片段BFM L1 |
| [DUALTALK_PROXY.json](DUALTALK_PROXY.json) | DualTalk原生迁移代理的完整指标 |
| [AUDIT.json](AUDIT.json) / [ADDITIONAL_AUDIT.json](ADDITIONAL_AUDIT.json) | 输入哈希、版本和数值验证 |
| [LISTENFORMER_BASE_RECHECK.json](LISTENFORMER_BASE_RECHECK.json) | 本轮官方PSNR/SSIM复算回执 |
| [COMPLETE.json](COMPLETE.json) | 原实验产物哈希清单，不因新增README改写 |
| [README_DELIVERY.json](README_DELIVERY.json) | 本次新增文档、CSV及生成脚本的哈希 |

## 9. 复跑

在项目根目录执行，沿用原环境，CPU运行；命令会重写本实验目录的结果文件，原归档预测只读。重跑后若浮点结果或文件内容变化，需重新检查产物哈希，不能继续将旧COMPLETE.json当成新结果封存证明。

```bash
cd /private/liqi_workspace/project_code/audio-driven-v5

env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  compare/.envs/vico_more/bin/python experiments/p2_vico_report_alignment_20261110/rescore.py

env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  compare/.envs/vico_more/bin/python experiments/p2_vico_report_alignment_20261110/rescore_additional.py

env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  compare/.envs/vico_eval/bin/python evaluation/vico_challenge_baseline/evaluations/compute_base_metrics.py \
  --gt_video_folder /public/0_DATA/2_Avatar/ViCo/dataset/by_split/test/videos \
  --pd_video_folder experiments/p2_vico_additional_baselines_20261107/listenformer/scoring/listenformer \
  --anno_file /public/0_DATA/2_Avatar/ViCo/dataset/by_split/test/test.csv --task listener
```
