# GLARE step19050：与final32940同范围测评

**2026-11-12恢复说明：** 829条视频已全部生成，ViCo五项／BFM／DIM全部完成。生成接管对象缺少returncode属性造成调度器退出，现已补齐Popen语义并验证存活／成功／失败／重复poll；已完成的10个ResponseNet分片直接复用。剩余评分恢复为最多8路；与final32940的4路生成共享GPU4，原全量评分在本轮完成后自动恢复。原始异常保留在FAILURE.previous.*及恢复日志中，见RECOVERY_20261112.json。完成文件仍以COMPLETE.json为准。

2026-11-12用户要求测试`before_extension30epochs.pt`。已实际读取检查点：**step19050**，SHA256 `3df923977903bffa4710eb2b13685cf28bb2a6d22f710dfda2ed2f3625f87f55`。使用node11物理GPU4，与仍在运行的final32940 Seamless全量队列共享；没有停止原任务。

先执行ViCo正式43条、ResponseNet完整586条、Seamless固定200条，合计829条。ViCo优先生成和评分；完整Seamless6624不在这个目录的当前队列中。每套数据完成后与[最终32940步结果](../glare_final_vico43_responsenet586_20261112/RESULTS.md)及[固定200结果](../glare_final_seamless_test6624_20261112/fixed200_RESULTS.json)对照。

829条speaker WAV、listener首帧及官方韵律缓存全部复用，逐文件SHA核对，见PRECHECK.json和REUSED_CONDITIONS.json。缓存不含未来listener条件；不重新提取韵律、不加载另一份Qwen2-Audio。测试ID、GT、seed137、原生Euler／nfe10、speakerCFG2、512／25fps、空动作统计、首帧参考与零历史均沿用最终版。

仅新增1个常驻生成worker，PyTorch显存限22GiB，LIA解码batch4；final32940为batch8，FP32批处理可能有通常的微小舍入差异，不宣称逐像素严格单变量。沿用原生LIA解码和模型，不改官方源码、精度或采样参数。图像指标、BFM、EMOCA及视频8项均复用已验证入口、权重、定义和GT特征。ViCo256／30fps，ResponseNet512／原FPS／face-padding64，Seamless512／25fps／face-padding0。

按显存余量接续：生成期间最多1个额外动作特征worker，之后最多2个；原视频评分2路，恢复后按上述GPU调度分配增至8路。评分只领取对应生成视频已全部完成的分片。主Seamless全量生成继续；GPU4整体仍约99%利用率，新增实验会分享其吞吐，原全量ETA可能顺延。

- 入口：[eval_glare19050_20261112.py](../../scripts/eval_glare19050_20261112.py)。原生推理复用[现有模块](../../scripts/eval_glare_final_20261112.py)，只参数化step／模型名并加入分片就绪检查。
- 进度：[status.json](status.json)；协议与权重：[protocol.json](protocol.json)；进程：LAUNCH.json／processes/。
- 日志：logs/controller_resume.log、logs/generate.log及各评分日志；FAILURE.json存在时核对对应日志。
- 视频：`<dataset>/generated/<id>/generated.mp4`；评分副本：`<dataset>/scoring/glare_step19050/<id>.listener.mp4`。
- 分阶段分数：`vico/IMAGE_RESULTS.json`、`vico/bfm/RESULTS.json`、`vico/DIM_RESULTS.json`、`responsenet/RESULTS.json`、`seamless/RESULTS.json`。
- 全部完成后：RESULTS.json／COMPLETE.json／MEDIA_AUDIT_COMPLETE.json；这些完成文件出现前不表示829条全部测完。

恢复前核实原控制器及子进程状态。使用avatarforcing环境、glare-deps优先PYTHONPATH和`CUDA_VISIBLE_DEVICES=4`执行同脚本的`run`。可接管仍活着的本目录生成进程；已有完整视频／评分分片复用。其他阶段若有遗留进程，应先核查并处理，不能重复叠加启动。EXECUTION_ADAPTER.json记录调度器接续及实际源码指纹；本轮原生模型和输入条件未改。
