# 相同AF生成，改用官方ViCo256真值／speaker评分

目标仍为复现AF Table4自身8项并核清其他行来源。本对照没有重生成视频，也没有选择样本／随机种子／单元格公式；只改变评测GT／speaker画面的来源。**本轮86路提取、43路生成特征复用、完整43条／129路与8项独立重算已完成；主口径和完整DIM候选均0/8，未复现Table4。** [完整结果](RESULTS.md) · [独立验收](FINAL_AUDIT.json)。

## 为什么要核对这一项

对官方256视频首帧与original层进行图像配准：86路中85路成功，77路横纵缩放差异超过10%，两个方向的尺度比中位数1.317。官方256裁脸与512宽视野方形裁剪有**人脸比例差异**，不只是分辨率和背景不同。[全部配准证据](../p2_avatarforcing_table4_reference_mean_20261110_v3/CANONICAL_CROP_GEOMETRY_AUDIT.json)。它解释了为何GT来源需要单独核对，并不证明AF作者用了哪种来源。

此前完整测试采用恢复后的512方形GT／speaker。本轮固定使用ViCo发布的256视频；同样按25fps最近时间戳读取完整区间，保留正式43条。共用的`decode25`会先将256源图用OpenCV cubic上采样到512再跟踪；本轮保留这段既有逻辑，记录源分辨率与跟踪分辨率，未将其称作原生256跟踪。生成仍为V3同一批512／25视频和相同动作特征，**没有另做生成图像256下采样**，以便隔离GT／speaker输入变化。[完整冻结协议](PROTOCOL.json)。全部86路已实际完整解码并核查源256、共用解码器上采样512及时间网格：[解码预检](CANONICAL_FULL_DECODE_PREFLIGHT.json)。

## 执行与验收

1. 等待V3完整43条／129路的评分和独立验收，以及两条作者公开演示的同前端读取结束；不会与其GPU worker叠加。
2. 仅node11物理GPU7，最多4个2.5% allocator worker，另计上下文并保留16GiB余量。提取43路官方GT及43路官方speaker；模型、权重、JPEG和SPECTRE时间上下文函数均与V3相同。
3. 复用43路完全相同的生成特征。核验86个新视频SHA、元数据角色、全部129路形状／有限性／时间网格，并独立重算主8项。
4. 主口径保留FD逐片段平均、拉平rPCC、SID15／9与逐系数时序Var；另完整运行DIM公开代码候选（SID40／20、拉平Var），不逐格拼接。预测相关的两种Var必须与V3逐值相同。

实际入口：`canonical_eval.py prepare`、`run.py`。使用原版环境`compare/.envs/avatarforcing_native201/bin/python`及`CUDA_VISIBLE_DEVICES=7`。[启动回执](LAUNCH.json) · [控制器](STATUS.json) · [提取进度](EXTRACTION_STATUS.json) · [结果](RESULTS.md) · [最终核验](FINAL_AUDIT.json)。阶段文件按实际运行产生。

AF没有公开其实际GT输入、拟合聚类中心及全套评分入口。本对照的任何数值接近都不能单独证明私有设置一致，也不改变目标为全部8项复现。

前序已实际完成：V3完整8项独立验收为1/8；作者两条演示的同前端读取也已完成。本队列已接续实际提取86路，实时状态见上方，不再等待前序任务。

## 最终结论

同一批生成视频改用官方256 GT／speaker后，Exp FD20.002274→42.816567，Pose FD0.068175→0.080033；rPCC也没有复现。生成相关Var逐值保持不变，已由程序断言。当前没有证据支持把此GT切换作为修复；不能反推作者的私有GT来源已被确认。

下一项按论文5.1补齐目标说话人音源分离，见[音频准备与官方资源核查](../p2_avatarforcing_table4_audio_preparation_20261110_v5/README.md)。
