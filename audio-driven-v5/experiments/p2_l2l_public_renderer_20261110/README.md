# L2L＋GPAvatar：公开写实渲染器

用户已授权为L2L接入兼容的公开渲染器。本轮选择[GPAvatar（ICLR2024）](https://github.com/xg-chu/GPAvatar)，使用官方公开checkpoint和50维表情＋6维姿态接口。保留原四套L2L人物权重的全部动作，不重训、不按GT选择人物。

绝对路径：`/private/liqi_workspace/project_code/audio-driven-v5/experiments/p2_l2l_public_renderer_20261110/README.md`。

[设计与兼容性](DESIGN.md) · [资源下载回执](DOWNLOADS.json) · [固定协议](protocol.json) · [实时状态](status.json) · [权重加载核对](WEIGHTS_LOADED.json) · [最短／最长样例](SMOKE_COMPLETE.json) · [全部生成回执](RENDER_COMPLETE.json)。对应文件不存在时，该项尚未完成。

**接入测试已通过，全量流程已启动。** 两个首帧×四套人物权重的8组／24帧样例通过，[画面对照](smoke_review.jpg)已实际检查。546个推理参数与公开checkpoint逐张量相等；追踪和渲染的FLAME基底最大差0；表情／头转／下颌对真实RGB的影响均验证有效。完整1秒／30帧视频已生成，所有副本完整解码通过，新进程完整片段与样例对应帧的原始像素差0：[视频](generated/conan/YFhkBtIUgBM_000106_000107.0/generated.mp4) · [完整片段回执](PILOT_VIDEO_COMPLETE.json)。该条保留在正式172条中。

按当前样例实测，纯渲染估计约7.8小时，另加首帧追踪、编码及评分；共享GPU变化会影响实际耗时。**尚未把172条及其最终评分标成完成。** [后台最近阶段](ACTIVE.json) · [控制器启动](controller_launch.json) · [冻结的自动接续流程](CONTROLLER_PROTOCOL.json)。

## 输入和生成结果

固定原ViCo test43、原L2L预测和原listener首帧。Conan／Fallon／Stephen／Trevor四套公开动作模型各生成43条，共172条。标签统一使用“L2L公开人物权重＋GPAvatar公开渲染器”，与作者未发布的原写实管线区分。

每个listener仅用首帧提取外观、身份形状和静态相机校准；后续只读取L2L预测的exp50、头转3、下颌3。眼球／颈部自由度沿用官方固定值；将首帧68个FLAME脸部关键点的均值保持在初始相机位置／深度，围绕参考脸中心施加预测旋转，避免未预测的平移使脸移出裁框，也避免同一旋转同时加到几何与相机。独立几何投影与等价相机投影最大差约9.54e-7，脸中心误差0；三类动作改变均影响实际RGB。

GPAvatar原生512×512，保持L2L的30fps和完整帧数。首帧统一加64像素黑边后缩回512，追踪成功后渲染；保留原生输出，并用固定scale0.8的中心采样还原公共512画幅，再存256评分副本。此模型只生成视频，不生成listener音频。官方tracker使用首帧抠图／透视规范化；输出保持模型的黑色背景，不把未来GT背景贴回输出。

## 文件和入口

- 接入脚本：`/private/liqi_workspace/project_code/audio-driven-v5/scripts/p2_l2l_gpavatar.py`。
- `references/<UUID>/`：只含该listener首帧的追踪和外观数据。
- `smoke/<人物>/<UUID>/`：最短、最长片段各取首／中／末帧检查，共8组。
- `generated/<人物>/<UUID>/generated.mp4`：正式完整512视频；`complete.json`包含动作、协议和视频哈希。
- `generated/<人物>/<UUID>/raw_padded.mp4`：模型原生512填充画幅视频，保留以便核查静态画幅还原。
- `scoring/l2l_gpavatar_<人物>/<UUID>.listener.mp4`：256评分副本。
- `CUDA_PROBE.json`、`RENDER_CONTROL_PARITY.json`：实际CUDA算子与动作控制检查。

准备使用原官方资源tar、主checkpoint及作者公布的MD5核验，FLAME／EMOCA沿用已有用户许可确认。原项目和已完成L2L动作结果保持封存，新增内容单独存放在本目录。

## 执行

已启动的控制器按`track_smoke → smoke → pilot → track → generate → EMOCA → 统一指标 → 四套官方画质指标`接续。已有正式视频和阶段回执均复用。控制器结束后写`PIPELINE_COMPLETE.json`；目前没有该文件即表示全量尚未完成。失败看`status.json`及对应`*_exit.json`和日志；不恢复暂停的我们模型best10。

在任意有当前集群连接配置的项目节点启动或恢复：

```bash
cd /private/liqi_workspace/project_code/audio-driven-v5
python experiments/p2_l2l_public_renderer_20261110/launch_stage.py controller
```

控制器已有独占锁时不重复启动。仅在node11物理GPU7手动运行单阶段时：

```bash
cd /private/liqi_workspace/project_code/audio-driven-v5
env CUDA_VISIBLE_DEVICES=7 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
  MAX_JOBS=2 TORCH_CUDA_ARCH_LIST=9.0 \
  TORCH_EXTENSIONS_DIR="$PWD/experiments/p2_l2l_public_renderer_20261110/torch_extensions" \
  compare/.envs/vico_more/bin/python scripts/p2_l2l_gpavatar.py track_smoke
```

GPU启动检查UUID及至少32GiB空闲，进程显存上限10%，在上限处仍保留至少约17.6GiB；完整30帧实测峰值4.13GiB。其他作业继续运行。最初容量门槛、旧protobuf兼容、紧裁框检测以及脸中心坐标适配的过程证据保留在日志和`revisions/`；旧诊断24帧与正式方案分开，未用于挑GT分数。隔离依赖版本见[ENVIRONMENT.json](ENVIRONMENT.json)，原环境包未改动。

全量结束后核验所有视频完整解码、原始动作哈希、首帧条件及正式test覆盖。`emoca_eval/RESULTS.md`和`metrics.csv`给出与已完成四行共用有效帧的8方法表；各人物的五项画质指标写入`image_eval/<人物>/RESULTS.json`。FID／ArcFace沿用原官方计算、batch64以限制显存，其余聚合不变。最终分数均采用独立目录和渲染器标签。
