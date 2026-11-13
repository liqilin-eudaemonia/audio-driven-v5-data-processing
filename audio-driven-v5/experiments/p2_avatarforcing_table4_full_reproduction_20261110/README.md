# Avatar Forcing Table4完整复现：进行中

> 2026-11-10后续状态：下文保留早期执行记录。v2已完成43条／129路并独立验收，两套冻结评分均0/8；预热后的并行严格检查误差0已通过。当前继续[参考图均值框整批对照](../p2_avatarforcing_table4_reference_mean_20261110_v3/README.md)，不要恢复本目录旧控制器。

**当前执行已接续到[尾帧修正v2](../p2_avatarforcing_table4_framegrid_20261110_v2/README.md)。** 本目录37条生成已保留；第38条因34.9667秒片段的round874／官方模型ceil875帧计数不同而在断言处退出。v2只修正输出计数和对应评分时间网格，源调用核对证明生成网络／条件／随机数／导出未改；37条只读复用，其余6条接续。不要恢复本目录带错误断言的旧控制器。

本轮新增SPECTRE官方前端已完成6路实际验证：RetinaFace ResNet50＋2DFAN2、置信度筛选及JPEG编解码；旧SFD／FAN4代理6路保留作诊断。源码及1055个跟踪器状态张量已逐项核验，见[官方前端协议](OFFICIAL_SPECTRE_PROTOCOL.json)、[6路实测](OFFICIAL_SPECTRE_PROBE.json)、[跟踪权重核验](official_tracker_audit/TRACKER_WEIGHTS_AUDIT.json)和[两种前端的实际系数差](FRONTEND_COEFFICIENT_COMPARISON.json)。文件名OFFICIAL仅指SPECTRE前端，不能称作AF官方测评程序。

独立媒体验收已通过全部43对：原始角色／CRC／SHA、全部文件、裁剪和解码／音频条件均经过检查，见[输入验收](ORIGINAL_INPUTS_INDEPENDENT_AUDIT.json)。注意ffmpeg与OpenCV的JPEG色度上采样不同，FFV1已对实际ffmpeg输入像素逐路验证无损，解码器差值另列。

并行提取优化未启用：2.5% allocator预检的关键点最大差6.10e-5像素、动作系数2.56e-6，超过预设1e-6门槛。8%串行重复诊断因当时GPU可用21315MiB低于27886MiB保护阈值而未启动。该优化按[暂缓记录](OFFICIAL_PARALLEL_HOLD.json)保留，v2不依赖它，继续单路正式提取。

用户已明确范围：**复现Avatar Forcing自身行8项，并核清其余行来源。** 目标未完成；不以准备环境、修完代码或重建协议替代原表复现，也不根据新评分反向选择公式／样本／种子。验收要求见[REQUIREMENTS.json](REQUIREMENTS.json)。

| 指标 | Exp目标 | Pose目标 |
|---|---:|---:|
| FD | 16.64 | 0.05 |
| rPCC | 0.01 | 0.01 |
| SID | 3.12 | 3.00 |
| Var | 2.80 | 0.03 |

## 已接续的事实

- 上轮43条8项没有复现，成绩独立封存。已确认完整原入口与简化直调会因实际cuDNN状态和min/max导出产生差异；独立固定入口已验证1条。见[上一轮定位](../p2_avatarforcing_table4_diagnosis_20261110/README.md)。
- 用户指出原始数据已下载后，核查发现现有`ViCo-Dataset.zip`同时包含`videos/cropped`和`videos/original`。此前只沿用cropped，遗漏original层。现已恢复test43双方共86路／83个唯一原始视频至`/public/0_DATA/2_Avatar/ViCo/original_test43/`，无需重新获取原始数据。见[归档核验](archive_control/ARCHIVE_CHECK.json)和[完整原始清单](ORIGINAL_TEST43_MANIFEST.json)。
- 已另行取得的两个YouTube全源文件仅保留作核对，不再继续下载其他源；正式恢复以官方ViCo归档original成员为来源。
- 正在通过作者发布的真实speaker面板与original视频进行像素配准，恢复可观察的裁剪窗口和帧映射；不通过生成分数选择裁剪。
- 官方推荐的Torch2.0.1／torchvision0.15.2／CUDA11.8独立环境已完成安装和实际GPU预检。现有环境和旧模型不覆盖。

## 本轮新增实证

- 官方代码公开范围已重新检查：main `507e3046ad539c938e0412ac7a74cf6e3ba2a61c`、page `f9e9dcc658793f6dc317e4fc29590f389d9aac41`、15个issue及全部已有评论、0个release、0个tag。未找到ViCo Table4完整评分入口；详细范围和原始响应见[公开代码核查](PUBLIC_CODE_AUDIT.md)。
- 两条发布示例的真实speaker像素配准，得到原始图坐标下的固定裁剪框约`[116,91,820,795]`、`[296,241,650,595]`。直接运行官方SFD／逐帧均值裁剪后得到`[116,91,820,795]`、`[296,242,650,596]`，分别完全一致和相差1像素。见[图像配准](AUTHOR_CROP_REGISTRATION.json)、[逐帧解码时序配准](AUTHOR_TIMELINE_REGISTRATION.json)、[官方裁剪实测](CROP_PROBE.json)。这验证输入恢复路径，不等于Table4数值已复现。
- 逐帧解码仍观察到发布speaker画面约两帧偏移；未把它推广成全部43条的时间偏移。新输入仍按明示的25fps／原始起点处理。
- 两对original的speaker／listener内嵌AAC包MD5分别相同，说明这里是共同对话音轨，不是已分离的两条人声。现有listening-only输入假设继续显式保留，未冒称已执行论文音源分离。[音轨与时间轴核验](SOURCE_TIMELINE_AUDIT.json)。
- `compare/.envs/avatarforcing_native201`独立环境已安装官方Torch2.0.1／torchvision0.15.2／torchaudio2.0.2／CUDA11.8和指定依赖。H200卷积、bf16 SDPA实际通过；327个生成参数与210个Wav2Vec状态张量逐张量相等，通过原生`weight_g/v`命名加载，无2.10命名适配。[GPU验收](NATIVE_GPU_PREFLIGHT.json) · [权重验收](NATIVE_WEIGHTS_AUDIT.json)。
- 全43对的原始视野预处理正在GPU7运行；原版环境下两条发布示例对应片段已生成275／500帧，完整解码、帧率及官方导出通过。见[两条实际推理验收](NATIVE_GENERATION_PROBE.json)。三路各两条共6路SPECTRE提取及时间上下文分块校验已通过；GT自对照FD约0、rPCC为0，不能记为AF成绩。见[提取预检](NATIVE_SPECTRE_PROBE.json)、[编码器校验](NATIVE_SPECTRE_PARITY.json)、[原版评分环境自检](NATIVE_METRIC_SELF_CHECK.json)。
- 自动接续已启动：预处理86路全部完成后，完成余下41条生成，复用已通过的2条；再提取余下123路系数并评分。只使用GPU7并检查容量，不启动训练、不调整其他作业。控制器PID在[启动记录](NATIVE_PIPELINE_LAUNCH.json)；实时状态见[接续状态](NATIVE_PIPELINE_STATUS.json)、[预处理](PREPROCESS_STATUS.json)、[生成](NATIVE_GENERATION_STATUS.json)、[特征](NATIVE_SPECTRE_STATUS.json)。尚无全43条新成绩。
- [冻结生成口径](NATIVE_GENERATION_PROTOCOL.json)和[冻结统计口径](NATIVE_METRICS_PROTOCOL.json)明确保留未披露项。SID训练聚类中心仍没有公开；当前GT拟合量化器只能标作代理重建，不冒称L2L原始训练拟合。主统计沿用上一轮方便比较，文字对应的Var／rPCC汇总等预声明诊断全部保留，不择最接近原表的一组替换。
- Table4其他四行来源已核清：†的RLHG／L2L／DIM共24个单元格逐项与DIM来源记录相同；INFP*为AF作者在其数据上训练200万步的重实现，不是原始INFP权重。全部32个基线数字从AF原PDF提取，见[来源核验](BASELINE_PROVENANCE.json)。该文件还记录INFP* Pose SID3.20超过K9的最大比特熵`log2(9)=3.169925`这一文字／数值不一致，不能强行视为全表同一套公开K9程序。
- 其中AF的“DIM†”对应DIM原论文`DIM-Listener†`，原论文†表示**CANDOR预训练**；AF的†则表示**转引DIM数字**，两者脚注含义不同。AF的L2L行对应DIM Table1普通L2L行，不是另一条CANDOR预训练L2L。已用本地DIM PDF第11页独立核对。

## 当前执行入口

使用`compare/.envs/avatarforcing_native201/bin/python`，工作目录为项目根目录，`CUDA_VISIBLE_DEVICES=7`。源码均位于本实验目录，旧封存脚本不修改：

1. `native_preprocess.py all`：恢复86路官方宽视野裁剪、512／25无额外有损压缩GT和原始首帧参考。
2. `native_run.py`：等待6路实际SPECTRE预检与完整预处理通过，再依次运行完整生成、提取和评分；控制器运行日志为`native_pipeline.log`。
3. `native_generate.py all`、`native_spectre.py all`、`native_score.py score`分别是三个阶段的可恢复入口。完成标记及所有输入／输出SHA逐阶段核验。

已安装环境版本见[NATIVE_ENVIRONMENT_FINAL.json](NATIVE_ENVIRONMENT_FINAL.json)。`NATIVE_EXECUTION_COMPLETE.json`只表示工程运行完，不自动表示Table4已复现；必须看8项逐格对照与独立最终核验。

## 推进顺序与验收

1. 恢复并核验43条原始双方媒体、实际帧率／时间轴与音轨，核对两条作者示例的裁剪和时序。
2. 按论文及可观察的官方预处理恢复512／25输入和GT；明确来源、角色、裁剪、音频与参考帧。
3. 运行官方模型／权重／默认采样配置；显式记录实际计算状态与导出范围。先验证发布示例对应片段，再完成全部43条。
4. 沿论文引用链继续核查FD、rPCC、SID、Var的具体计算，区分原ViCo的Feature Distance与Table4的Fréchet Distance。未披露细节须保持推断标签，不挑最接近表格的结果。
5. 在冻结配置下提取／评分并逐项对照8项原表。核清其他四行的来源：†三行转引DIM，INFP*为AF作者重训结果。
6. 原表8项、全部视频／特征覆盖、来源及统计设置经过独立检查后，才能标记目标完成。若仍不一致，保留结果并继续用受控实验定位，不改写目标。

资源沿用node11物理GPU7，只做推理／预处理／评测，按实时容量检查并保留其他作业。不恢复旧DIM best10或其他训练。
