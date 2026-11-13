> 后续状态：4路功能预检通过，但独立进程ROI／音频严格一致性检查失败，批量未启动；已在[固定前端开关版本](../p2_avatarforcing_table4_audio_preparation_stable_20261110_v5b/README.md)接续，旧输出仅保留诊断。

# 补齐AF论文中的目标说话人音源分离

目标仍是复现AF Table4自身8项，并核清其他行来源；本目录准备输入音频，不代表已完成新的生成或Table4评分。

## 依据与当前进度

AF论文5.1明确写了视觉辅助的说话人音源分离；[官方README](https://github.com/TaekyungKi/AvatarForcing#preprocessing)也指定IIANet和ClearVoice。此前冻结输入明确是混合对话，没有执行分离。因此先补这个有论文依据的工程步骤，不通过扫描种子／CFG去追表格数值。

- 已取得[IIANet官方源码](https://github.com/JusperLee/IIANet)，固定提交`86f35381cca8cb02a16d0405f3f8cf46b27d61c5`，42个文件逐个核对Git blob及SHA。[来源清单](IIANET_SOURCE_MANIFEST.json)。
- 使用官方单视频推理指定的Vox2模型。GitHub Release音频权重与作者[issue6](https://github.com/JusperLee/IIANet/issues/6)压缩包中的音频权重逐字节相同；视觉权重取该包中与配置同名的`lrw_resnet18_mstcn_adamw_s3.pth.tar`。[包及源码一致性](IIANET_BUNDLE_PARITY.json)。
- 音频网络277个状态、视觉网络137个状态逐项相等，没有未加载后留随机值的模型状态。[实际模型组装验收](IIANET_MODEL_ASSEMBLY_AUDIT.json)。
- 额外依赖只安装在本目录`python_deps`，不修改现有Torch2.0.1／CUDA11.8环境。[原版音频I/O预检](AUDIO_IO_PREFLIGHT.json)确认默认WAV是FLOAT，不能误写成PCM16。
- 已在GPU7启动两条公开演示对应片段的预检，分别提取speaker和listener两路。此时没有启动新的43条AF生成。[启动回执](CANARY_LAUNCH.json) · [进程状态](RUNNER_STATUS.json) · [原始日志](canary.log) · [预检完成后产生的结果](CANARY_COMPLETE.json)。

## 固定处理方式

[协议](PROTOCOL.json)在分离输出和新Table4分数出现前固定：每路输入是对应人物的512／25原始视野固定裁剪，配同一段16k混合音轨；一条人物视频对应一个提取目标。使用官方MTCNN检测／FAN4关键点／嘴部平滑裁剪／LRW归一化，以及完整序列Vox2网络，不按评分挑片段或改声音增益。

保留官方`detectface`和`crop_mouth`函数。适配仅包括：以同样的OpenCV顺序解码替代mmcv视频包装；NumPy1.26对含None关键点的保存显式指定object类型，之后仍走官方插值；绝对路径及GPU7，跳过原脚本硬编码GPU9的main。检测结果另存供核查，不改变MTCNN选择。保留官方224 FMP4人脸中间视频、96灰度嘴部ROI、88中心裁剪及输出FLOAT WAV，另存原始浮点估计。

只在node11物理GPU7运行，allocator5%、至少32768MiB空闲才启动，保留其他作业。两个预检片段及最长片段均需核验长度、有限性、加载指纹、目标人物和重复网络前向一致性；分离可靠后再覆盖正式43条并适配相同生成配置。

listener stem用于检查当前“自身音频为零”假设，不会自动喂入GT listener音频。后续默认仍保留speaker-only比较条件；AF私有测试如何处理自身音频仍未公开。论文只明确视觉辅助分离，README另提ClearVoice但未给串接模型／配置，本轮不擅自增加一条不明的增强链。

前序结果：参考图预处理对照完整43条／129路独立验收为[1/8](../p2_avatarforcing_table4_reference_mean_20261110_v3/CONTROLLED_REFERENCE_RESULTS.md)；相同生成改官方256 GT／speaker后两套冻结评分均[0/8](../p2_avatarforcing_table4_canonical_gt_20261110_v4/RESULTS.md)，Exp FD20.00→42.82，没有修复差距。两条公开演示的本地表情时序方差仍偏弱。所有旧结果保留，不重写为已对齐。
