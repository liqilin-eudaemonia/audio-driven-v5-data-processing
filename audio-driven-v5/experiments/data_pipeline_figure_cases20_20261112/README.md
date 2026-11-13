# 数据处理流程图：20个真实有声候选

入口：[浏览器复核页，Cursor转发8876](http://localhost:8876/listen_benchmark_review/data_pipeline_cases20_20261112/)。

已完成20条素材导出和逐条媒体验收；浏览器20条加载、5种音频模式、播放、反馈导出／重载、HTTP Range及移动端布局检查通过。没有人工判断每条动作／词句是否正确，最终由用户选择主case和备用。

## 范围与选择

- 全部来自最终LTX500h与最新Qwen SFT train的交集，没有使用测试集。
- 20个不同source pair，每个listener参与者最多2条；5–8秒，原图人脸短边p10≥224px，实际本批均大于320px。
- 要求净化音轨非零、原caption含短引用发声、真实Event缓存的speech为启用正例且至少有一个启用视觉正例，原生成检查无提示。
- 在上述条件下优先大脸、动作可辨认及较简洁caption，并平衡yeah／okay／right／oh／非词汇发声及其他短回应。不是总体随机抽样，也不是人工真值。
- 同时保存实际屏蔽项、停用类别及全部原文；没有为符合草图改写caption或Event标签。

## 如何选

1. 默认听净化后的listener，确认确实有清楚发声；再切speaker或双方同时听核对倾听角色。
2. 切换净化前listener与被置零部分，判断是否适合展示音轨门控的前后差异。被移除部分不是已经确认的纯噪声。
3. 对照视频、4张原生关键帧、caption、检测线索和Event表。适合主case的片段应容易说明动作顺序和发声。
4. 页面可选择“选作主case／备用case／暂不选”，意见保存在本浏览器并可导出JSON；也可以直接反馈C01–C20编号。

## 每条材料

页面左侧为最终裁框listener视频和同步float32音轨播放器；右侧为4张原生源画面裁剪PNG、原caption、真实19维Event数值、辅助线索与官方活动区间。
每条media目录提供：`listener.mp4`、`reference.png`、`source_frame.jpg`、`frame_01.png`至`frame_04.png`、`listener.wav`、`original_listener.wav`、`speaker.wav`、`removed_listener.wav`、原caption JSON、清单行、音频配方及`event_targets.json`。
MP4／M4A是H264／AAC试听副本；网页同步试听使用与处理配方一致的float32 PCM。PNG关键帧从实际源视频重新读取并采用同一最终裁框，不从生成视频或示意图取得。
Event表保持真实y／known／weight，另列冻结的类别启用标记；有效样本权重为known×enabled×weight，训练的类别平衡项未在这里重新计算。类别是整段presence，不是时间标签。

## 候选列表

| 编号 | caption中的引用发声 | 秒 | 人脸p10原像素 | 净化前区间外能量占比 | 样本ID |
|---|---|---:|---:|---:|---|
| C01 | Mm-hmm | 6.46 | 460.1 | 0.04% | naturalistic_V03_S0134_I00000140_local_e4ebcfd986af |
| C02 | Yeah. | 6.35 | 549.9 | 8.62% | naturalistic_V03_S1374_I00000340_local_e788913b625b |
| C03 | Oh | 5.82 | 432.8 | 0.05% | naturalistic_V03_S2069_I00000094_local_81843f8261e1 |
| C04 | There. | 5.25 | 419.6 | 0.00% | naturalistic_V03_S2072_I00000098_local_f0ce9187c13c |
| C05 | Okay | 6.38 | 386.4 | 46.82% | naturalistic_V03_S0935_I00000164_local_778164e5b5a8 |
| C06 | Right? | 6.47 | 414.0 | 2.60% | naturalistic_V03_S0557_I00000077_local_c4e0cb6f423c |
| C07 | Yeah, | 5.74 | 432.4 | 0.00% | naturalistic_V03_S0335_I00000034_local_0526845ed335 |
| C08 | just pushing | 6.74 | 387.3 | 20.32% | naturalistic_V03_S0115_I00000126_local_6633bd00fd8a |
| C09 | Okay. | 5.89 | 383.0 | 5.43% | naturalistic_V01_S1600_I00000134_local_1645d9b78249 |
| C10 | Uh-huh. | 6.87 | 371.8 | 12.98% | naturalistic_V03_S2139_I00000181_local_ee92f6ab4c95 |
| C11 | Oh. | 6.60 | 411.5 | 55.89% | naturalistic_V03_S0218_I00000461_local_524280db7a2c |
| C12 | Right | 7.49 | 402.9 | 0.01% | naturalistic_V03_S2041_I00000333_local_db531d896d7a |
| C13 | Yeah | 7.46 | 409.4 | 1.67% | naturalistic_V03_S0099_I00000309_local_8c1cd96b3787 |
| C14 | AI | 7.82 | 398.4 | 7.05% | naturalistic_V03_S1213_I00000461_local_0cac1861dcb7 |
| C15 | uh | 5.44 | 377.3 | 35.22% | naturalistic_V03_S0936_I00000112_local_2b76e410435b |
| C16 | Okay, | 6.31 | 353.8 | 0.00% | naturalistic_V03_S1371_I00000205_local_08ad393f3e19 |
| C17 | Right | 6.71 | 337.7 | 0.02% | naturalistic_V01_S0156_I00000127_local_7d53e83482be |
| C18 | Oh. | 7.83 | 387.8 | 95.19% | naturalistic_V03_S0161_I00000486_local_5e0040db8431 |
| C19 | Yeah. | 7.37 | 399.9 | 0.11% | naturalistic_V03_S0262_I00000139_local_09f0cb6f22c6 |
| C20 | Exactly. | 6.12 | 419.0 | 0.47% | naturalistic_V03_S0124_I00000499_local_8659917a89dc |

上述发声文字来自原caption，未经本次人工逐条听辨；区间外能量占比不是去噪准确率或声源归属判断。

[候选清单CSV](candidates.csv) · [完整选样及原行](selection.json) · [媒体验收](MEDIA_AUDIT.json) · [浏览器验收](BROWSER_AUDIT.json) · [完成记录](COMPLETE.json)。

复现入口：`scripts/prepare_data_pipeline_cases20_20261112.py`，两路CPU，无GPU／无重打标。原始训练数据、缓存和草图未修改。
