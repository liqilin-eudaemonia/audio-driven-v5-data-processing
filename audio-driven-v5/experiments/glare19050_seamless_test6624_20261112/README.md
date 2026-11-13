# GLARE step19050：完整Seamless test6624

**最终已完成（2026-11-12 11:33 UTC／北京时间19:33）：** 6624条生成、209个评分块、全局特征合并及全部生成媒体指纹验收通过。图像／分布6619条、表情6576条、ΔResp6569段；保留5条原素材缺帧的N/A及其他检测支持不足，最终状态partial。见[COMPLETE.json](COMPLETE.json)、[FINAL_AUDIT.json](FINAL_AUDIT.json)、[全量原始成绩](test6624_RESULTS.json)和[用户指定结果页](http://localhost:8876/listen_benchmark_review/vico_tables_20261112/)。32940步全集继续暂停。以下为执行与修复记录。

**2026-11-12收尾：** 6624条生成、209个评分块均已结束。首次汇总被读取器版本差异校验拦住：全程评分模型／公式／权重一致，过程中共享读取器新增了upright分支和EOF处理。已恢复出SHA完全一致的原读取器快照，核对全部5144条新版读取器成功行的1229个来源；4条触发EOF差异的样本以原读取器实际执行得到的缺帧N/A保留，其余RGB读取路径一致。原分片及特征不改写，汇总按可追溯记录移除这4条的指标支持。证据见[READER_COMPATIBILITY.json](READER_COMPATIBILITY.json)、[原读取器快照](reader_before_upright.py)和[检查脚本](verify_reader_compatibility.py)。另有此前042块记录的1条原始GT缺帧，均保留请求ID。

全局FID／FVD与最终媒体指纹验收由`finish_merge.py`接续，进度见`merge_recovery.log`；以`COMPLETE.json`和`FINAL_AUDIT.json`落盘为最终完成依据。旧汇总异常保存在`FAILURE_before_merge_recovery_*.json`，不是重新生成视频。完成后自动更新评测对比表。

2026-11-12用户明确要求Seamless测试完整6624条，范围以此为准。固定200是其子集，不能代替完整成绩。此前19050步的[ViCo43、ResponseNet586和固定200](../glare19050_vico43_responsenet586_seamless200_20261112/README.md)已完成。

权重为`before_extension30epochs.pt`，实际step19050，SHA256 `3df923977903bffa4710eb2b13685cf28bb2a6d22f710dfda2ed2f3625f87f55`。本目录固定同一完整test6624条／11.05h，参考清单、seal、源行与裁框身份同final32940，见[protocol.json](protocol.json)。

已复用全部6624条speaker WAV／listener首帧和官方韵律缓存，输入逐文件SHA通过。已完成的200条原生／评分视频及8个评分块直接复用，权重、输入、seed、nfe、CFG及视频SHA逐条核对；复用视频和评分文件以软链接引用原完成结果，溯源写入新完成收据。继续处理其余6424条，不重复计算200条，不重提韵律。

仅使用node11物理GPU4。用户最新指令为暂停32940步、只测19050步；32940步的控制器及评分进程均已停止，结果和共享输入缓存保留。本队列现为4个生成worker，每进程PyTorch显存限22GiB，LIA解码batch4；生成期间最多2个评分worker，生成结束后最多8个。

原生模型／精度／Euler nfe10／speakerCFG2／seed137／首帧参考／零历史／空动作统计均沿用先前配置。512×512／25fps；视频8项、face-padding0、batch64、GPU检脸／表情、4线程像素计算、GT特征复用和CPU预取均保持。只复用现有固定200的RGB缓存，不保存另一个全量RGB缓存。无listener生成音频，speaker背景音不参与音频评分。

- 入口：[eval_glare19050_seamless_test_20261112.py](../../scripts/eval_glare19050_seamless_test_20261112.py)，复用现有完整队列和原生推理。
- 状态：[status.json](status.json)；冻结与复用验收：[PREPARED.json](PREPARED.json)；启动／进程：LAUNCH.json、processes/、REQUEST_SCOPE.json。
- 日志：controller.log、logs/generate_*.log、metrics/NNN/video.log；异常见FAILURE.json。
- 视频：`seamless/generated/<id>/generated.mp4`，评分副本：`seamless/scoring/glare_step19050/<id>.listener.mp4`。
- 最终完整结果：`test6624_RESULTS.json`，固定200另存`fixed200_RESULTS.json`。FID／FVD各自合并全部特征，不平均分片分数。
- 全部结束以`COMPLETE.json`和`FINAL_AUDIT.json`为准，完整成绩生成前不拿200条代替。GT缺帧时沿用原评分器N/A并保留请求ID及覆盖率，不补造帧，见source_issues/。

恢复前先核实本目录控制器／子进程状态；不要重复启动。设置CUDA_VISIBLE_DEVICES=4及原glare-deps优先PYTHONPATH，用avatarforcing环境运行同脚本的`run`。INPUT／PROSODY／已完成视频／评分分片继续复用；不得删除被软链接引用的原输入、韵律、固定200结果目录。
