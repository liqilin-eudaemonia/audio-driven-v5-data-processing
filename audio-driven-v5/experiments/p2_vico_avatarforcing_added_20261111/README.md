# Avatar Forcing加入当前ViCo统一评测（2026-11-11）

**已完成：43/43条、图像评分16,288帧、统一动作评分16,238个共同有效帧。** 官方生成权重及已生成视频通过SHA核验后复用；本次没有重新生成43条视频。新增执行五项官方图像评分，并把AF加入现有八行共同EMOCA动作评分，全部九行重新计算。其他八行数值不变。

## AF结果与Report并列

本轮是**音频共同输入适配**：listener首帧＋speaker音轨，listener音轨为零，user_visual为零latent；没有读取未来listener视频/音轨。官方完整多模态输入和论文Table4与此不同，论文采用SPECTRE，本表统一读出EMOCA，时间基准也不同。因此以下只并列展示，不根据数值差归咎于官方，也不宣称已经复现论文。

| 指标 | Report Exp | Report Pose | 本地Exp | 本地Pose |
|---|---:|---:|---:|---:|
| FD | 16.64 | 0.05 | 37.633272 | 0.068397 |
| P_FD | — | — | 39.178973 | 0.071590 |
| MSE | — | — | 0.897029 | 0.014181 |
| SID | 3.12 | 3.0 | 4.303114 | 3.678300 |
| Var | 2.8 | 0.03 | 0.796301 | 0.010247 |
| rPCC | 0.01 | 0.01 | 0.042412 | 0.038954 |

Table4来源及此前SPECTRE复现证据：[V3受控参考图结果](../p2_avatarforcing_table4_reference_mean_20261110_v3/CONTROLLED_REFERENCE_RESULTS.md)、[V4官方256 GT对照](../p2_avatarforcing_table4_canonical_gt_20261110_v4/RESULTS.md)。前者1/8按论文两位小数相同，后者0/8；这些是历史独立实验，不是本次EMOCA对比的通过率。最新[音源分离V5e](../p2_avatarforcing_table4_audio_preparation_cpu_20261110_v5e/README.md)已完成43对/86路并停止，尚未据此重新生成或评分。本次共同评测沿用已冻结音频，不把旧视频标成新分离音轨生成。

## 图像指标同表

| 方法 | PSNR ↑ | SSIM ↑ | FID ↓ | CSIM ↑ | CPBD ↑ |
|---|---:|---:|---:|---:|---:|
| ListenFormer (local retraining) | 18.887633 | 0.641760 | 30.842915 | 0.602755 | 0.193178 |
| DyStream | 18.379981 | 0.618699 | 35.067230 | 0.467174 | 0.226187 |
| L2L conan + GPAvatar | 14.215021 | 0.514727 | 87.993300 | 0.440848 | 0.215046 |
| L2L fallon + GPAvatar | 13.946953 | 0.496049 | 91.571952 | 0.422281 | 0.221127 |
| L2L stephen + GPAvatar | 14.156595 | 0.505480 | 94.723012 | 0.432411 | 0.223262 |
| L2L trevor + GPAvatar | 13.788192 | 0.493226 | 94.580409 | 0.405309 | 0.230406 |
| Avatar Forcing (audio-only adaptation) | 16.183142 | 0.572531 | 39.140994 | 0.466332 | 0.190193 |

AF五项为本次重测，其他六行引用已封存结果；来源与SHA均在[图像CSV](image_metrics.csv)。图像FID是RGB分布指标，不等于动作FD。

## 统一动作同表

| 方法 | FD Exp ↓ | FD Pose ↓ | P-FD Exp ↓ | P-FD Pose ↓ | MSE Exp ↓ | MSE Pose ↓ |
|---|---:|---:|---:|---:|---:|---:|
| joint | 25.892533 | 0.087136 | 27.166891 | 0.087986 | 0.652535 | 0.016007 |
| joint_sft2752 | 24.339086 | 0.061817 | 26.074124 | 0.062967 | 0.711805 | 0.012427 |
| dystream | 31.715231 | 0.067969 | 33.394710 | 0.069342 | 0.807346 | 0.013097 |
| listenformer | 32.334237 | 0.098622 | 32.986166 | 0.099563 | 0.695527 | 0.016980 |
| l2l_gpavatar_conan | 62.679655 | 0.107203 | 64.681366 | 0.108949 | 1.521195 | 0.021275 |
| l2l_gpavatar_fallon | 69.255731 | 0.135133 | 71.364966 | 0.137539 | 1.643143 | 0.026208 |
| l2l_gpavatar_stephen | 74.354409 | 0.124185 | 76.200755 | 0.126149 | 1.711027 | 0.024168 |
| l2l_gpavatar_trevor | 61.132577 | 0.173117 | 62.897938 | 0.174421 | 1.421860 | 0.030332 |
| avatarforcing_audio_only | 37.633272 | 0.068397 | 39.178973 | 0.071590 | 0.897029 | 0.014181 |

43条固定test，listener 1:T对speaker 0:T-1；九行与GT/speaker共同有限帧交集。沿用DIM发布函数，Var按此前确定的逐系数跨帧population方差再平均，ddof=0；原始拉平Var也保存在JSON。数学实现对照16个量最大差0，只证明评分实现一致，不证明论文完整私有协议一致。

[完整动作CSV](motion_metrics.csv)含SID/Var/rPCC；[逐片覆盖与原始结果](MOTION_RESULTS.json)；[全部Report与本地对照CSV](report_vs_local_all.csv)。旧表不同空间的原生行保留各自备注，不能混成同一排序。

## 执行、缓存与恢复

只在node11物理GPU7执行；没有停止GLARE或其他训练。运行脚本：[输入核验与共同动作评分](evaluate.py)、[官方图像评分封装](image_metrics.py)。图像评分复用官方预处理、PIQ FID、ArcFace与CPBD公式，FID/CSIM batch64，CPBD逐片CPU并行并按原帧数加权。已完成metric回执按锁复用。

FID原脚本每个小批次重新构建Inception；本次仅通过PIQ原生feature_extractor参数复用同一模型，其计算函数和模型不变。[一致性记录](FID_CACHE_PARITY.json)：真实GT与预测各64帧，两批特征最大差0。仅这些批次测得的时间不能推广为整批加速比例。重启只终止本次低效FID子进程；其他计算结果和进程保留。重测五项与既有AF评分的差值见[验收记录](FINAL_AUDIT.json)。FID从39.1527885932变为39.1409943196，差−0.0117942736（约−0.0301%），不声称跨运行完全一致。历史封装的batch为256，本轮为64并由gpu_guard关闭TF32、启用确定性cuDNN；这几项执行差异没有逐项消融，因此不将差值单独归因于缓存或任何一个设置。两批cache特征一致性仅在相同当前配置下成立。

项目根目录的恢复命令（已完成时图像阶段直接复用回执）：

```bash
CUDA_VISIBLE_DEVICES=7 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
  compare/.envs/vico_eval/bin/python experiments/p2_vico_avatarforcing_added_20261111/image_metrics.py
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
  compare/.envs/vico_more/bin/python experiments/p2_vico_avatarforcing_added_20261111/evaluate.py score
python experiments/p2_vico_avatarforcing_added_20261111/finalize.py
```

换权重、输入、特征或评分协议时使用新实验目录，不能直接复用本目录完成回执。历史评测文件未改写；[输入核验](INPUT_AUDIT.json)和[完成清单](COMPLETE.json)记录来源。

[Avatar Forcing独立交接文档](../../related/method_handoffs/avatarforcing.md) · [全部21个方法文档](../../related/method_handoffs/README.md)。
