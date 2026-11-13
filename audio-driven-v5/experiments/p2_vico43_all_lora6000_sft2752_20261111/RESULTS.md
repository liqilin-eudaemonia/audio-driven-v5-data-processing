# 全attention／FFN LoRA6000：ViCo正式test43

43条已生成并完成官方画质／身份5项、CustomListener风格运动补测、DIM表情／姿态12项。
SFT2752、seed137、Euler30、CFG3／7；原生512×512／25fps，抗混叠缩小到256×256并按最近时间戳转30fps评分。

| 官方画质／身份 | 实测 |
|---|---:|
| fid | 31.207690 |
| csim | 0.593484 |
| psnr | 19.086009 |
| ssim | 0.647273 |
| cpbd | 0.142201 |

| 运动指标 | 分组 | 实测（原展示倍率） | 覆盖条数 |
|---|---|---:|---:|
| FD | exp | 13.995444 | 43 |
| FD | angle | 9.662795 | 43 |
| FD | trans | 6.626089 | 43 |
| RTLCC | exp | 0.230091 | 42 |
| RTLCC | pose | 0.260064 | 42 |
| RWTLCC | exp | 0.296290 | 41 |
| RWTLCC | pose | 0.332332 | 41 |
| FID_delta_fm | exp | 3.296515 | 43 |
| FID_delta_fm | pose | 0.057649 | 43 |
| SND | exp | 2.923722 | 43 |
| SND | pose | 0.099776 | 43 |
| V-D | exp | 0.011354 | 43 |
| V-D | pose | 0.003917 | 43 |

运动表的FD及FID_delta_fm展示值乘100，其余保持原尺度。BFM为Exp64、角度3及平移3；DIM为EMOCA Exp50／Pose6，同名FD不能混排。

| DIM口径 | Exp | Pose |
|---|---:|---:|
| FD | 27.345738 | 0.064315 |
| P-FD | 29.300260 | 0.065525 |
| MSE | 0.774121 | 0.012971 |
| SID | 4.685235 | 3.459982 |
| Var | 0.905801 | 0.010235 |
| rPCC | 0.022930 | 0.029981 |

DIM主Var沿用用户指定的逐系数总体方差平均，原发布代码拉平方差另存。所有指标使用单次seed137；DIM逐帧共同有效范围和旧SFT2752基线同覆盖重算结果见JSON。

运动补测属于已确认公开定义复现，DIM属于公开代码口径；这些结果不等同于严格复现论文私有执行配置，AF的SPECTRE口径不在本轮范围。

[完整JSON](ALL_RESULTS.json) · [图像结果](IMAGE_RESULTS.json) · [运动结果](bfm/RESULTS.json) · [DIM结果及有效帧](DIM_RESULTS.json)

[43条推理视频与音轨](VIDEOS.md) · [交付复核](FINAL_USER_REVIEW.json)
