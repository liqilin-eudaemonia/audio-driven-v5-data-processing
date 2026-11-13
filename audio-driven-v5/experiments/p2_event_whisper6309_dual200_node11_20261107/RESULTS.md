# Event＋Whisper 6309步：固定双200条

仅node11 GPU7；固定ResponseNet分层随机200＋Seamless固定200，seed137／Euler30／CFG3、7。
6309 updates: first4000 batch31 then2309 batch30 =193270 sample exposures; historical32x6000=192000 (+0.66%)

| 数据集 | PSNR↑ | SSIM↑ | FID↓ | FVD↓ | LPIPS↓ | Emo-Acc↑ | Emo-Div | ΔResp↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| seamless | 16.1694 | 0.672093 | 16.6092 | 176.399 | 0.345494 | 0.385586 | 3.72864 | 0.534292 |
| responsenet | 16.9971 | 0.656301 | 13.2933 | 160.223 | 0.321524 | 0.61049 | 3.17 | 0.217495 |

完整覆盖、音频活动、UTMOS有效性和逐条结果见RESULTS.json与metrics/；不把静音／短回应的N/A填零。
