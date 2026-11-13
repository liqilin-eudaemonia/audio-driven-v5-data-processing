# 自主规划3条样本：与真实标注指导同样本配对

两种指导×AV/AV＋Event，共4条件，每条件3条×2seed＝6个输出。新增自主规划12个视频；真实标注指导复用原相同样本的特征与检测结果重新汇总。

仅原test24中3条规划通过的样本，不代表完整测试集或自主规划成功率。模型均为guided、冻结Qwen的最终step1249；同GT、首帧、speaker、seed和采样设置。计分使用独立listener.wav，不使用叠加speaker的音轨。

每项AV指标在4条件×2seed之间取共同支持；GLARE严格版取相同共同有效ID，去门槛版使用全部3条。每seed先计算后等权平均，不能把6输出当作6个独立源样本。

## 去额外门槛版

共同支持（每seed）：{'fvd': 3, 'utmos': 0, 'sync': 0, 'glare': 3}

| 条件 | FVD-32 | UTMOSv2 | LSE-D | LSE-C | R-F1 | R-tIoU | R-ATD | R-FID |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| AV · 真实标注指导 | 934.9025 | N/A | N/A | N/A | 0.0542 | 0.6176 | 68.0000 | 90.3271 |
| AV · Qwen自主规划 | 827.6170 | N/A | N/A | N/A | 0.1309 | 0.7662 | 40.5745 | 89.2762 |
| AV＋Event · 真实标注指导 | 887.9599 | N/A | N/A | N/A | 0.0819 | 0.6356 | 96.2500 | 83.9277 |
| AV＋Event · Qwen自主规划 | 962.3465 | N/A | N/A | N/A | 0.0794 | 0.6100 | 100.0000 | 82.9751 |

## 严格逐帧版

共同支持（每seed）：{'fvd': 3, 'utmos': 0, 'sync': 0, 'glare': 1}

| 条件 | FVD-32 | UTMOSv2 | LSE-D | LSE-C | R-F1 | R-tIoU | R-ATD | R-FID |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| AV · 真实标注指导 | 934.9025 | N/A | N/A | N/A | 0.0333 | 0.3676 | N/A | 92.9044 |
| AV · Qwen自主规划 | 827.6170 | N/A | N/A | N/A | 0.0694 | 0.8214 | 46.2500 | 75.2351 |
| AV＋Event · 真实标注指导 | 887.9599 | N/A | N/A | N/A | 0.0750 | 0.6378 | 113.7500 | 81.3344 |
| AV＋Event · Qwen自主规划 | 962.3465 | N/A | N/A | N/A | 0.0417 | 0.3600 | N/A | 78.5603 |

## 指标覆盖与音轨诊断

N/A表示不适用或没有共同有效样本，不表示0分；逐条原因保存在comparison.json的audio_diagnostics。FVD和R-FID在3条样本上方差很大，只用于诊断。R-F1等取决于自动反应检测，不能直接视为人工判断的回应合理性。

| 条件 | seed | 样本 | GT RMS | 生成RMS | 生成UTMOS状态 | 生成Sync状态 |
|---|---:|---|---:|---:|---|---|
| AV · 真实标注指导 | 137 | naturalistic_V03_S0921_I00000314_local_a5b50e764dc3 | 0.0 | 1.889602267090638e-05 | success | success |
| AV · 真实标注指导 | 251 | naturalistic_V03_S0921_I00000314_local_a5b50e764dc3 | 0.0 | 2.6316582282669363e-06 | not_applicable | not_applicable |
| AV · 真实标注指导 | 137 | naturalistic_V03_S0921_I00000314_local_b69cf369a34c | 0.0 | 2.0408988222015953e-05 | success | success |
| AV · 真实标注指导 | 251 | naturalistic_V03_S0921_I00000314_local_b69cf369a34c | 0.0 | 2.356359626373533e-05 | success | success |
| AV · 真实标注指导 | 137 | naturalistic_V03_S1493_I00000481_local_135b540c056d | 0.015503811105024787 | 0.135117212055664 | success | success |
| AV · 真实标注指导 | 251 | naturalistic_V03_S1493_I00000481_local_135b540c056d | 0.015503811105024787 | 0.1184625225783418 | success | success |
| AV · Qwen自主规划 | 137 | naturalistic_V03_S0921_I00000314_local_a5b50e764dc3 | 0.0 | 2.0960530439356988e-05 | success | success |
| AV · Qwen自主规划 | 251 | naturalistic_V03_S0921_I00000314_local_a5b50e764dc3 | 0.0 | 2.6592308070887952e-06 | not_applicable | not_applicable |
| AV · Qwen自主规划 | 137 | naturalistic_V03_S0921_I00000314_local_b69cf369a34c | 0.0 | 3.091238428338904e-05 | success | success |
| AV · Qwen自主规划 | 251 | naturalistic_V03_S0921_I00000314_local_b69cf369a34c | 0.0 | 0.00017460703250259764 | success | success |
| AV · Qwen自主规划 | 137 | naturalistic_V03_S1493_I00000481_local_135b540c056d | 0.015503811105024787 | 2.7440927217020525e-06 | not_applicable | not_applicable |
| AV · Qwen自主规划 | 251 | naturalistic_V03_S1493_I00000481_local_135b540c056d | 0.015503811105024787 | 2.8138510940191215e-06 | not_applicable | not_applicable |
| AV＋Event · 真实标注指导 | 137 | naturalistic_V03_S0921_I00000314_local_a5b50e764dc3 | 0.0 | 3.209017593720545e-05 | success | success |
| AV＋Event · 真实标注指导 | 251 | naturalistic_V03_S0921_I00000314_local_a5b50e764dc3 | 0.0 | 2.5947221201598712e-06 | not_applicable | not_applicable |
| AV＋Event · 真实标注指导 | 137 | naturalistic_V03_S0921_I00000314_local_b69cf369a34c | 0.0 | 2.5097673870106153e-05 | success | success |
| AV＋Event · 真实标注指导 | 251 | naturalistic_V03_S0921_I00000314_local_b69cf369a34c | 0.0 | 8.220899898212479e-05 | success | success |
| AV＋Event · 真实标注指导 | 137 | naturalistic_V03_S1493_I00000481_local_135b540c056d | 0.015503811105024787 | 0.17058201098696796 | success | success |
| AV＋Event · 真实标注指导 | 251 | naturalistic_V03_S1493_I00000481_local_135b540c056d | 0.015503811105024787 | 0.13092760108281246 | success | success |
| AV＋Event · Qwen自主规划 | 137 | naturalistic_V03_S0921_I00000314_local_a5b50e764dc3 | 0.0 | 1.765174925177503e-05 | success | success |
| AV＋Event · Qwen自主规划 | 251 | naturalistic_V03_S0921_I00000314_local_a5b50e764dc3 | 0.0 | 0.0002086792032335285 | success | success |
| AV＋Event · Qwen自主规划 | 137 | naturalistic_V03_S0921_I00000314_local_b69cf369a34c | 0.0 | 0.0001042763138521551 | success | success |
| AV＋Event · Qwen自主规划 | 251 | naturalistic_V03_S0921_I00000314_local_b69cf369a34c | 0.0 | 0.00017955570899896652 | success | success |
| AV＋Event · Qwen自主规划 | 137 | naturalistic_V03_S1493_I00000481_local_135b540c056d | 0.015503811105024787 | 7.779916019949372e-05 | success | success |
| AV＋Event · Qwen自主规划 | 251 | naturalistic_V03_S1493_I00000481_local_135b540c056d | 0.015503811105024787 | 2.020641304288705e-05 | success | success |
