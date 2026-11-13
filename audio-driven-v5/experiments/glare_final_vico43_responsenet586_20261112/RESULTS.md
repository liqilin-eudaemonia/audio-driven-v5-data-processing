# GLARE final32940 · ViCo43＋ResponseNet586

全部完成，629条生成及评分，1,258个原生／评分视频完整解码验收通过；执行错误0。仅node11物理GPU4。

最终权重：`/private/liqi_workspace/project_code/audio-driven-v5/experiments/glare_paper_settings_node11_gpu7_20261111/latest.pt`，SHA256 `4f42e95fbe07b2c391a587da7e6cce3a2c1c8731a602ce0b5c86af29b2b40bb5`；未使用best。

| 数据集 | PSNR ↑ | SSIM ↑ | FID ↓ |
|---|---:|---:|---:|
| ViCo test43 | 18.737042 | 0.634337 | 30.455398 |
| ResponseNet full586 | 18.542848 | 0.681975 | 6.659255 |

ViCo：CSIM 0.511535，CPBD 0.183138；全部16,288评分帧。

| ResponseNet其他视频指标 | 值 |
|---|---:|
| lpips | 0.253220 |
| fvd | 109.207025 |
| emo_acc | 0.639468 |
| emo_div_generated | 2.105802 |
| emo_div_gt | 3.382253 |
| delta_resp | 0.290718 |

ResponseNet各图像／分布指标覆盖586条，PSNR／SSIM／LPIPS共128,038帧；情绪共同有效127,581帧。没有生成listener音轨，音频指标N/A；speaker背景音不作listener输出。

| DIM EMOCA | Exp | Pose |
|---|---:|---:|
| FD | 26.818755 | 0.083810 |
| P_FD | 28.219331 | 0.085737 |
| MSE | 0.696666 | 0.016380 |
| SID | 4.702481 | 3.509615 |
| Var | 0.833202 | 0.015616 |
| rPCC | 0.018046 | 0.058090 |

DIM共同有效16,238帧，公开代码16项数值复核最大误差0；Var为用户指定逐系数总体方差均值，原公式另存。

| BFM公开定义运动指标 | 分组 | 原展示倍率数值 | 覆盖条数 |
|---|---|---:|---:|
| FD | exp | 13.581185 | 43 |
| FD | angle | 11.283835 | 43 |
| FD | trans | 6.862198 | 43 |
| RTLCC | exp | 0.223161 | 42 |
| RTLCC | pose | 0.268108 | 42 |
| RWTLCC | exp | 0.292156 | 41 |
| RWTLCC | pose | 0.326765 | 41 |
| FID_fm | exp | 2.780692 | 43 |
| FID_fm | pose | 0.125838 | 43 |
| FID_delta_fm | exp | 3.508514 | 43 |
| FID_delta_fm | pose | 0.061533 | 43 |
| SND | exp | 2.815777 | 43 |
| SND | pose | 0.126453 | 43 |
| V_D | exp | 0.007225 | 43 |
| V_D | pose | 0.003839 | 43 |
| V_D_GT | exp | 0.021512 | 43 |
| V_D_GT | pose | 0.008256 | 43 |

BFM的FD／FID_delta_fm乘100，其余保持原值。ViCo使用既有公开实现口径；不是对论文私有协议完全等价的声明。ResponseNet为完整586条，不能与历史精选／随机200条直接混排。

[完整JSON](RESULTS.json) · [验收](FINAL_AUDIT.json) · [ViCo43视频](vico/VIDEOS.md) · [ResponseNet586视频](responsenet/VIDEOS.md) · [协议与恢复](README.md)
