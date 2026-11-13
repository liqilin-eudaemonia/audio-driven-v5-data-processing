# 四组权重、固定与标注指导的完整配对评测

范围：原训练集24条和独立test24条，各4组权重 × 2个生成seed（137、251）。**每组每个数据集48个输出，指导版共384个**；与原固定prompt的384个输出配对。先前9个工程验证输出经完整身份及文件哈希核验后复用，不缩小评测范围。

唯一推理干预是Gemma正向文本换为整合后的原始标注。speaker音频、listener首帧、权重、30步采样、video CFG=3、audio CFG=7、负向文本、seed、分辨率和长度保持原设置。每次生成均核验负向embedding完全相同；生成完成后再逐输出核对实际输入哈希、checkpoint、配置、几何参数和seed。

- `jobs.json`：冻结的384条生成任务；`pairing_audit.json`：输入/权重/采样参数。
- `status.json`：自动流程状态；`workers/`、`logs/`：逐分片进度和错误。
- `generation_pairing_verified.json`：全部生成完成后的逐输出配对核验。
- `generate_cluster.py`：3机24卡完整生成，独立进程与完成哈希校验。
- `evaluate_cluster.py`：完整Omni AV评测、GLARE独立检测、8条件共同支持聚合。
- `finish.py`：生成完成后自动接续指标、报告和HTML，不需要人工二次启动。

完整输出和评测目录：

- `../p2_guided_train24_fourway_eval_20261027/`
- `../p2_guided_test24_fourway_eval_20261027/`

每个目录中，四组的`omni/results.json`保留全部48个计划配对的FVD-32、UTMOSv2与SyncNet结果；`glare/results.json`保留完整48配对检测覆盖及失败，`detected_pairs.json`保留成功检测。公开四项为R-F1、R-tIoU、R-ATD、R-FID。完整集合不足时不虚报全量分数。

`comparison.json`、`table_values.json`、`report.md`使用固定/指导共8条件、两个seed均有效的相同片段集合，分别按每项指标取交集。FVD按每seed样本池计算；SyncNet按参考实现有效轨迹等权；两seed等权平均。GLARE匹配、ATD无匹配时空值传播沿用既有协议。原固定prompt结果文件不覆盖。

HTML入口：http://127.0.0.1:8876/p2_av_review/ 。完成前显示真实已收录输出数/48及进行中状态；完成后各行48个输出，指标采用8条件同口径比较，原始独立listener音轨用于评分，speaker混音仅供展示。

指导版访问了目标标注，属于给定回应文本的生成诊断；不将其解释为只听speaker便能自主决定如何回应的证据。train与test分开统计。

## 完成状态与检测修正版

384个指导输出及384个固定输出的评测已完成，`status.json`为`complete`；生成配对身份核验384/384通过。音视频指标各条件保留48条完整记录，支持不足的指标仍明确报告有效范围。

GLARE随后统一修复了FER内部MTCNN输入缺少BGR→RGB转换的问题，参见[修复记录](../p2_glare_detector_fix_20261027/README.md)。原结果不覆盖，最新报告为两个评测目录中的`comparison_rgb_v2.json`、`report_rgb_v2.md`；HTML优先读取修正版。test八条件共同GLARE片段由0/24增至10/24，train修正版为1/24。FVD、UTMOS及SyncNet不受此检测修复影响。

页面支持八条件共同范围与同prompt内部四组范围切换。两套数据、两种范围共四种表格状态均已用浏览器逐数值核对，每行48个输出；GT有声播放、混音切换、连续滚动及移动端检查全部通过。验证记录见`../p2_av_review_20261027/browser_rgb_validation.json`和`browser_full_guided_validation.json`。
