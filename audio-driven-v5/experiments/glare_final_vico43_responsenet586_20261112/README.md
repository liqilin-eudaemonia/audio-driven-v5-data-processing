# GLARE 最终检查点：ViCo43＋ResponseNet586

**全部完成（2026-11-12 01:26 UTC）：** 629条生成、全部评分及1258份媒体完整解码验收通过，执行错误0。ViCo DIM共同有效16238帧，公开代码16项数值误差0；ResponseNet各图像／分布项覆盖586条。共约37分钟，仅GPU4。后续已接续[Seamless完整test6624](../glare_final_seamless_test6624_20261112/README.md)。[结果表](RESULTS.md) · [汇总JSON](RESULTS.json) · [验收](FINAL_AUDIT.json) · [交付核查](DELIVERY_AUDIT.json)。

2026-11-12 用户指定直接评测最后检查点，仅 node11（172.29.189.119）物理 GPU4。其他任务保持。

权重：`../glare_paper_settings_node11_gpu7_20261111/latest.pt`，第32940步；不用best。完整SHA、629条冻结输入、输入身份和协议见 [protocol.json](protocol.json)。

- ViCo 正式test43：相同speaker音频和listener首帧；512／25fps原生生成，256／30fps评分。官方画质／身份5项、现有BFM运动补测、DIM EMOCA12；GT／speaker特征复用，保留共同有效帧与覆盖率。这是既有公开实现口径，不能冒称作者私有论文协议完全复现。
- ResponseNet 全586：沿用冻结J切分，包含514条官方词时间、72条本地词时间来源。512／GT原fps及精确帧数，既有8项视频指标，FID／FVD由全586特征汇总。GLARE不生成listener音频；原生MP4的speaker背景音只供试听，不评分为listener输出。
- 原生FLOAT模型／ODE、nfe10、speaker CFG2、seed137；全片段单窗口、参考frame0、零历史，使用学习到的空动作统计；不读取listener未来或GT运动条件。沿用已训练的数据接口，不改官方模型。
- 加速：官方离线韵律跨来源batch64、常驻模型共享缓存、4路生成、LIA解码batch8、每路有界CPU编码重叠；评分batch64、GPU检脸／表情、4线程像素计算、CPU预取、GT特征缓存，按实测显存逐步扩至16个独立ResponseNet评分分片，与4路ViCo动作提取重叠。生成阶段约98%–99%利用率，评分重叠阶段约99%–100%；属于生产观测，不宣称严格同样本加速比。批处理可能有通常的浮点差异，不声称逐位相同。

运行入口：[eval_glare_final_20261112.py](../../scripts/eval_glare_final_20261112.py)。状态见 [status.json](status.json)、`controller.log`、`logs/`；进程见 `LAUNCH.json`／`processes/`。原生视频：`<dataset>/generated/<id>/generated.mp4`；评分副本：`<dataset>/scoring/glare_final32940/<id>.listener.mp4`。

阶段结果：`vico/IMAGE_RESULTS.json`、`vico/bfm/RESULTS.json`、`vico/DIM_RESULTS.json`、`responsenet/RESULTS.json`。最终汇总 `RESULTS.json`、逐视频完整解码验收 `FINAL_AUDIT.json`、`COMPLETE.json`。COMPLETE出现前不表示全部完成；执行错误见FAILURE.json及对应日志。

[ViCo43视频](vico/VIDEOS.md) · [ResponseNet586视频](responsenet/VIDEOS.md)。ViCo系数评分使用与既有基线相同的vico_more环境（numpy1.26.4／scipy1.15.3／sklearn1.3.2），不是GLARE推理环境。BFM沿用此前已确认的诊断验收：五点及对齐RGB完全相同、系数差小于原1e-3容差；未参与对齐的68点差异保留披露，不改真实网络输入。

恢复前核实node11的控制器已退出；不要重复启动。GPU4和UUID固定校验。使用avatarforcing环境、glare-deps优先PYTHONPATH，运行同一脚本的`run --workers 4 --decode-batch 8 --metric-workers 16`。完成收据复用，过期进程的claim自动回收，已有韵律、完整视频及完成的动作分片不重算；评分分片锁避免与仍活着的原进程重复计算。不要修改既有韵律缓存关联源码后直接混用。评分环境接续见CONTROLLER_RESUMED.json，原GPU工作进程保留。
