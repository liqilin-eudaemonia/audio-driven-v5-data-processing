# DyStream · ResponseNet分层随机200条

使用冻结200条清单；512×512/25fps原生生成，最近时间戳映射到每条GT原始FPS及帧数后评分。
仅speaker音频及listener首帧；自身音频置零，关闭二次裁脸；无生成listener音轨。
使用既有ResponseNet视频评分器，FID/FVD全200条特征汇总；Learning2Listen按用户要求暂不启动。

| 指标 | 值 |
|---|---:|
| psnr | 16.410668765162484 |
| ssim | 0.6406914972448015 |
| lpips | 0.34168296319477004 |
| fid | 17.530999408931564 |
| fvd | 184.48084533595784 |
| emo_acc | 0.5551914060721926 |
| emo_div_generated | 3.08 |
| emo_div_gt | 3.345 |
| delta_resp | 0.23861316441473035 |

状态：complete；有错误的pair：0。完整覆盖见RESULTS.json。
