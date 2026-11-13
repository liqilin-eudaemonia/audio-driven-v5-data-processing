# GLARE人脸检测颜色通道修复

问题来自两个因素叠加：

1. 原适配层调用FER的BGR接口，符合FER公开函数的入参说明；但本机FER 22.5.1内部MTCNN路径直接把这个BGR数组传给人脸检测，缺少RGB转换。FER情绪网络的灰度转换又必须使用BGR，因此不能简单将整个`detect_emotions`入参改成RGB。
2. 本项目额外的质量门槛要求每帧MediaPipe、FER及单人脸关联全部有效，任何一帧失败即整段无效；八个条件、两个seed再取共同片段交集。此100%门槛是本项目防止漏检被当成负反应的质检设置，并非四个指标公式本身要求所有视频双检测器全帧通过。

原test GT只有11/24条满足该门槛，MediaPipe部分全部有效，主要失败在FER人脸检测。原test生成384条中180条整段通过，13条存在MediaPipe缺失，其他大量失败来自FER；因此交集为空不等于所有视频都失败。

## 验证与修复

- `failure_analysis.json`记录原GT、生成视频的逐类失败；`probe_inputs.json`和`rgb_probe.json`为修复试验。
- 在8条失败视频的39个“MediaPipe有效、MTCNN漏检”帧上，改成RGB检测框后39/39恢复了单人脸和情绪输出。
- 新适配层：`src/p2_glare_rgb/detector.py`。MTCNN用`find_faces(BGR转RGB)`，FER用`detect_emotions(原BGR, face_rectangles=新检测框)`。
- 3项测试验证颜色通道只作用于人脸检测、情绪分支仍用原BGR、空框不回退错误路径、质检阈值和全部反应规则不变。
- 仍保留100%逐帧门槛、IoU阈值、唯一FER人脸要求；不插值、不补0，不把没检测到当成没反应。
- 模型权重与GLARE四项指标公式不变。检测模型在worker内常驻；MediaPipe逐视频重新初始化。

对48个GT及768个生成视频共816个独立视频统一重跑，固定版和指导版使用同一新协议。原目录和结果完整保留，RGBv2有独立身份、质量统计和文件哈希。

## 结果与任务

- `protocol.json`冻结全部视频、配对、参数与源码；`status.json`记录检测、评分、等待AV汇总及发布状态。
- `detections/<video_sha>/`保留每帧FER、MediaPipe、quality和有效/无效标记。无效视频不产出伪造的反应CSV。
- `pairs/`保留每条件全部48配对的成功/失败范围。
- `scope.json`记录八条件共同范围、固定四组内部范围、指导四组内部范围。
- 原文标注指导评测目录中的`comparison_rgb_v2.json`和`report_rgb_v2.md`为修正版；原`comparison.json`、`report.md`保留原BGR版本。
- HTML优先展示RGBv2，允许切换严格八条件比较与各prompt内部四组比较，不混用两个检测版本。

这修复了可确认的实现问题，不承诺消除所有检测盲点；仍然漏检的视频保留失败记录。不同有效样本集合的表不能直接跨表比较。

## 全量检测结果

816个独立视频全部执行完成，无执行错误；643个严格通过，173个因检测不完整而无效。

| 数据与范围 | 原BGR版 | RGB修正版 |
|---|---:|---:|
| train GT | 12/24 | 17/24 |
| train生成视频 | 146/384 | 252/384 |
| test GT | 11/24 | 21/24 |
| test生成视频 | 180/384 | 353/384 |
| test八条件、双seed共同片段 | 0/24 | 10/24 |

修复后，train八条件共同片段为1/24；固定prompt四组内部为5/24，指导四组内部为1/24。test固定四组内部为15/24，指导四组内部为14/24。共同范围同时要求对应GT有效。

剩余test生成失败中，13个视频有MediaPipe缺帧、28个有FER缺帧，两类有重叠。颜色修复不能解决全部姿态、画质或人脸检测困难。train共同范围仍只有1条，不足以据此作稳健的四组优劣结论。

28组评分任务全部完成，修正版报告及HTML已发布，`status.json`为`complete`。浏览器已逐项核对train/test的两种比较范围、每条件48输出与全部表格数值，并复验有声播放和手机布局；记录位于`../p2_av_review_20261027/browser_rgb_validation.json`及`browser_full_guided_validation.json`。
