# OmniResponse 评测工程验收

本目录只验证实现正确性，不是 E0/E1 的正式效果评测。两段真实媒体来自现有训练数据的工程自配对；FD 的 3 维特征和文本是明确标记的已知样例，不冒充实际 3DMM 或模型转写。

- [最终回归](regression.log)：71 项通过：新评测契约/公式/覆盖率/缓存/音轨选择/导出/生成入口，加上原 GLARE 评测回归。
- [真实单模型测试](model_smoke_v2.json)：CPU I3D、UTMOSv2、S3FD/SyncNet、METEOR/ROUGE-L/RoBERTa；SyncNet 保持 50 个真实帧。
- [最终完整流程](verified_end_to_end/report.md)、[完整 JSON](verified_end_to_end/results.json)：两条媒体、自配对、同一 seed，所有已选择指标实际运行。
- [最终验收汇总](validation.json)：最终源代码绑定、数值、自配对两侧一致、缓存复用和训练不受影响的核对结果。
- [当前参考模型与代码锁](../../configs/p2_omni_runtime.lock.json)：97 个源文件、174 个模型/资源文件，两个 Python 环境及 ffmpeg/ffprobe 固定。
- [metrics 环境](metrics_python.freeze.txt)、[SyncNet 环境](sync_python.freeze.txt)：原环境版本清单，本次未安装或升级这些环境。

修复记录：初始真实测试发现上游 AVI 默认音频编码的延迟使 50 帧增加为 52 帧；`model_smoke.json` 是修复前诊断记录，不能作为最终验收。改为 PCM、显式映射 listener 音频并复核时间轴后，`model_smoke_v2.json` 回到 50 帧。

小样本 I3D 自配对的上游 sqrtm 产生约 -0.0001508 的舍入误差，最终报告保留 raw_reference_value，并按明确相对容差将该小负距离归零；大负值不接受。工程自配对的 FVD-32 为 0，FD 为浮点舍入量级，不能用这两个样本评估生成质量。

`end_to_end/`、`final_end_to_end/` 为开发过程的旧冻结协议，保留诊断；最终入口是 `verified_end_to_end/`。代码变更后不伪造旧协议身份以复用旧结果。

使用说明：[P2_OmniResponse评测实现与运行](../../related/P2_OmniResponse评测实现与运行.md)。

最终完整链路及缓存复用已通过。复用耗时约 12.5 秒，成功记录 mtime/哈希不变，结果逐项一致；E0 在验收时仍正常训练至第 526 步，原训练 release 校验通过。
