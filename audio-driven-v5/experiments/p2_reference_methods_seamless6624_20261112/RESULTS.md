# Seamless 完整test6624：五方法结果

本页仅统计完整6624条；历史双200结果独立保留。L2L四套权重分别报告。ListenFormer使用本地ViCo重训；L2L公开GPAvatar渲染；DualTalk尚无已验证RGB映射。音频指标N/A。

更新：2026-11-13T06:57:07.054936+00:00；状态：running。产物数量不代表全量分数已汇总。

| 方法 | 视频 / 6624 | 原生动作 / 6624 | 视频N/A | 已评分请求 / 6624 |
|---|---:|---:|---:|---:|
| DyStream 官方 | 631 | N/A | 0 | 300 |
| Avatar Forcing 官方，audio-only | 1283 | N/A | 0 | 1200 |
| ListenFormer 本地ViCo重训 | 985 | 987 | 0 | 200 |
| L2L conan＋GPAvatar | 386 | 975 | 2 | 200 |
| L2L fallon＋GPAvatar | 386 | 975 | 2 | 200 |
| L2L stephen＋GPAvatar | 386 | 975 | 2 | 200 |
| L2L trevor＋GPAvatar | 386 | 975 | 2 | 200 |
| DualTalk 原生动作 | N/A | 988 | N/A | N/A |

## 完整6624视频八项

| 方法 | 状态 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | FID ↓ | FVD ↓ | Emo-Acc ↑ | Emo-Div 生成/GT | ΔResp ↓ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DyStream 官方 | 进行中 | — | — | — | — | — | — | — | — |
| Avatar Forcing 官方，audio-only | 进行中 | — | — | — | — | — | — | — | — |
| ListenFormer 本地ViCo重训 | 进行中 | — | — | — | — | — | — | — | — |
| L2L conan＋GPAvatar | 进行中 | — | — | — | — | — | — | — | — |
| L2L fallon＋GPAvatar | 进行中 | — | — | — | — | — | — | — | — |
| L2L stephen＋GPAvatar | 进行中 | — | — | — | — | — | — | — | — |
| L2L trevor＋GPAvatar | 进行中 | — | — | — | — | — | — | — | — |
| DualTalk 原生动作 | N/A：无已验证RGB映射 | — | — | — | — | — | — | — | — |

完整FID/FVD按6624条有效特征重新汇总。partial及N/A按原评分器支持情况保留，不补零。

## DECA原生动作诊断

这些动作诊断与统一RGB八项分开；DualTalk neck/global head代理映射仍有限制。

| 方法 | FD Exp | FD Pose | P-FD Exp | P-FD Pose | MSE Exp | MSE Pose |
|---|---:|---:|---:|---:|---:|---:|

[运行与Monitor说明](README.md) · [实时状态](status.json) · [GPU监控](MONITOR.json) · [CSV](image_metrics.csv)。只有根目录COMPLETE.json存在才表示全部6624范围完成。
