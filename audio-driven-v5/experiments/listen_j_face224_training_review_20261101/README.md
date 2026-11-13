[此前展示的24条样本清单CSV](review_sample_list.csv) · [总筛选记录内的逐条网页直达清单](../../related/P2_J数据筛选与500h有声补充方案记录_20261101.md#7-已展示的24条人工复核样本清单)。

# train＋dev 人脸 p10≥224：24条最终裁框训练素材复核（2026-11-01）

[打开复核页面](http://localhost:8876/listen_benchmark_review/j_face224_training_20261101/)。Cursor转发当前机器8876端口。

从最新最终保留集的267,070段／468.1782504小时中，在train内均匀随机抽20条、dev内抽4条。固定种子`20261101:<split>:face224`，按排序后的sample ID无放回抽样；dev为了检查而超额抽样，24条不用于估计总体类别比例。没有按画面效果、标签数量、有声与否筛换样本。

## 使用范围

这是已处理数据池的源素材重放。该J池尚未导出新的训练manifest／VAE latent缓存，本页不宣称复核了已投入训练的batch。原split、training_admitted及候选标记保持不变，不将dev加入train。视频25fps／512×512参照现有新50h训练配置；8n+1补齐仅列出参考参数，预览不拉长真实片段。原Omni整段描述是当前文本候选，未虚构固定文本或标注指导组的最终system prompt。

## 实际读取和校验

- 最终裁框：统一调用`j_omni_crop.effective_row`，校验画质retain、原来源指纹、测量SHA256，应用`quality-bounded-crop-v1-20261030`的`crop.new`。绝不直接把历史caption框当作最终框。
- 视频：按TAR偏移有限范围读取原listener视频，25Hz时间网格选最近源PTS，复用RGB裁剪＋PIL bicubic到512；首帧与参考PNG一致。保存每张源帧PTS及编码前RGB哈希。网页MP4为CRF15压缩预览，不是逐像素无损训练缓存。
- listener：复用caption内官方词／句／VAD并集配方；使用现有`read_audio`重放，与原模型输入的float32波形SHA256逐条比对。源时间不删除、不拼接。
- speaker：从同一pair的另一参与者规范化音轨读取相同起止时间；核对角色、源文件SHA256与音频采样数。不对speaker套listener静音掩码。
- 网页同步播放器使用little-endian float32原波形，48路（24×2）浏览器fetch校验；设备输出采样率由浏览器正常重采样。下载MP4/M4A是AAC试听副本。无新增持久WAV。
- 标注：整段原文与完整caption JSON保持逐字节一致；保留原行为统计、点头候选、官方音频时间支持，未重打标或把开放类别硬映射回七类。

24条中：8条listener有声、16条全静音。14条历史旧框标注，其中11条的标注框与最终视频框不同；页面明确提示，遵循此前保留旧标签并标明来源的约定。旧框点头证据不冒充新框检测。

## 建议复核顺序

1. 看默认listener视频：头部是否完整，动作与整段原文是否一致。
2. 切换speaker音轨：核对对方说话内容与listener反应的对应关系；双方同时试听只用于上下文复核，不是训练目标混音。
3. 对listener有声样本核对回应、串音与静音位置；全静音样本不因没声音而视为播放器故障。
4. 查看黄色裁框来源提示、点头候选和官方词／句／VAD；有问题记录时刻和现象，可导出反馈JSON。

## 文件与复现

- [抽样清单](sampling.json)、[构建校验](build_audit.json)、[媒体与真实浏览器验收](review_audit.json)。
- [页面数据](../../data_previews/listen_benchmark_review/j_face224_training_20261101/data.json)：每条源文件位置、双路音频哈希、实际框／历史标注框、视频源PTS、原文与统计。媒体子目录还包含原始caption、统计行、净化配方和裁框来源。
- [构建脚本](../../scripts/build_j_face224_training_review.py)、[页面模板](../../scripts/j_face224_training_review.html)、[验收脚本](../../scripts/check_j_face224_training_review.py)。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/build_j_face224_training_review.py --workers 2
/tmp/native-fps-review-qa/bin/python scripts/check_j_face224_training_review.py
```

CPU最多2路解码，每路编解码限线程；不占GPU、不扫描全量媒体。构建续跑复用本次已完成的样本文件，修改媒体逻辑时应使用新输出目录或明确重建本次预览；不会改动生产源文件。

## 抽样样本

|序号|划分|sample_id|人脸p10(px)|秒|listener|标注框与最终框不同|
|---|---|---|---:|---:|---|---|
|1|train|`naturalistic_V03_S0204_I00000137_local_951b2fcefdda`|350.27|4.402|全静音|否|
|2|train|`naturalistic_V03_S0555_I00000166_local_f16a6d24808b`|237.52|2.870|全静音|是|
|3|train|`naturalistic_V01_S1558_I00000309_local_27725e5e8382`|319.00|6.810|有声|否|
|4|train|`naturalistic_V03_S1582_I00000000_local_5a573df554ac`|289.63|4.543|全静音|否|
|5|train|`naturalistic_V03_S0242_I00000371_local_3e5477274b4d`|249.39|6.486|全静音|否|
|6|train|`naturalistic_V03_S2026_I00000094_local_c89850b2679d`|288.68|8.808|全静音|是|
|7|train|`naturalistic_V03_S2050_I00000179_local_9e8e7d3d4f6b`|295.35|10.385|全静音|否|
|8|train|`naturalistic_V03_S1022_I00000167_local_d9c5e69f4766`|269.14|2.881|有声|否|
|9|train|`naturalistic_V03_S0862_I00000108_local_55863a0f73d2`|252.62|4.977|全静音|否|
|10|train|`improvised_V03_S1174_I00000372_local_0b26de7f88f3`|334.65|8.785|全静音|是|
|11|train|`naturalistic_V03_S0555_I00000170_local_31300c1d682e`|234.34|2.674|有声|否|
|12|train|`improvised_V01_S0227_I00001226_local_2262c9e23394`|276.48|2.441|有声|否|
|13|train|`naturalistic_V03_S1764_I00000415_local_2248f12b97af`|280.16|5.739|有声|否|
|14|train|`naturalistic_V03_S1877_I00000202_local_9a894c7408a3`|229.15|8.260|有声|否|
|15|train|`naturalistic_V03_S0887_I00000089_local_b044895829dd`|243.05|6.626|全静音|是|
|16|train|`improvised_V03_S0286_I00000135_local_20ee2a00acbd`|229.12|5.182|全静音|是|
|17|train|`naturalistic_V03_S0845_I00000396_local_664d6dda36eb`|368.26|4.901|全静音|是|
|18|train|`naturalistic_V03_S1553_I00000254_local_61813ed9f110`|237.97|7.031|全静音|是|
|19|train|`naturalistic_V03_S2100_I00000181_local_0c6cd218cc1c`|241.73|7.503|全静音|否|
|20|train|`naturalistic_V03_S1805_I00000045_local_cb1763d3f0c3`|263.73|4.724|全静音|是|
|21|dev|`naturalistic_V03_S1096_I00000005_local_7b992fb1bd42`|268.80|8.174|全静音|是|
|22|dev|`naturalistic_V03_S1662_I00000115_local_72814e892167`|269.10|9.470|有声|是|
|23|dev|`improvised_V03_S0209_I00000582_local_5ffe6778447f`|238.53|7.405|有声|否|
|24|dev|`naturalistic_V03_S0120_I00000125_local_d78fd06302b9`|225.48|4.268|全静音|是|

### S1877 音频冲突核查（2026-11-01）

网页“listener有声”原先只表示波形非零，现改为“listener非零音频”，不作为人声或回应的判定；每次切换样本重置为仅播放listener。S1877的官方词／句记录yeah.（7.222–7.962s），VAD为7.660–7.976s，而原caption否认发声。净化音频重新计算与网页float32逐样本一致，7.122–8.076s以外全部为零。保留区间内仍可能含杂音，未凭caption直接改动原音频或标签。核查记录：[S1877_audio_check.json](S1877_audio_check.json)。
