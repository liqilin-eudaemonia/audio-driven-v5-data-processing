# Avatar Forcing Table4：参考图均值裁剪框对照

目标仍是复现AF自身8项两位小数，并核清其他四行来源。本轮只改变参考图预处理（整段均值裁剪框及官方预处理JPEG输出）；没有更换权重、音频、随机种子、采样或指标公式。不能把成绩差异进一步单独归因于框位置或JPEG编解码。**本轮43条／129路已完整评分并独立验收，主口径和完整DIM候选均1/8，未复现Table4。** [本轮结果及V2对照](CONTROLLED_REFERENCE_RESULTS.md)。

## 依据和限制

作者公开补充视频的背景能配准回ViCo原始listener。CNN示例的参考区域与整段固定均值框吻合；使用均值框后，生成画面的背景角点误差从8.77降至0.14像素。TED示例没有改善，误差6.14变为6.67像素，仍存在生成形变／参考时刻不明等差异。两个结果均保留在[几何对照](REFERENCE_FRAMING_CONTROL.json)，不能声称已经重建作者全部参考图。

采用同一规则处理全部43条：取官方预处理输出的listener第一张JPEG，直接执行官方图像transform，不再单帧SFD二次裁剪。[冻结生成协议](NATIVE_GENERATION_PROTOCOL.json) · [整批对照决策依据](REFERENCE_POLICY_DECISION.json)。

均值框的位置来自完整listener轨迹，这是使用完整轨迹确定ROI的离线预处理假设。模型条件仍只有首帧像素，没有输入未来listener画面或自身真实音频；不能隐去ROI使用完整轨迹这一点。

## 验收和执行

- 两条示例已实际生成；选择依据是作者公开画面，而非本地指标高低。[生成预检](NATIVE_GENERATION_PROBE.json)。
- 官方SPECTRE前端函数体和权重未变，源视频不变的86路GT／speaker特征只读复用，并保留父回执、特征及视频SHA。[复用清单](REUSED_GT_SPEAKER_FEATURES.json) · [86路独立来源核验](REUSED_FEATURES_INDEPENDENT_AUDIT.json)。
- 新目录内已重新运行完整275帧严格一致性检查：动作／关键点最大误差均为0，没有放宽1e-6门槛。[严格检查](official_tracker_audit/PARALLEL_PARITY.json)。控制器因此采用实际严格检查及源代码／86路来源核验作为启动门槛，不等待未运行的六路串行probe标记。
- 仅node11物理GPU7，原版Torch2.0.1环境；生成allocator15%，保留16GiB；生成后最多4个2.5%独立SPECTRE worker。
- `native_run.py`先补齐另外41条，后提取43路新生成系数，再运行冻结评分。`audit_when_ready.py`对全部43条／129路及8项独立重算，最后单独执行事先冻结的DIM完整代码候选。
- 已实际启动：控制器PID3492136、生成PID3493331、审计等待PID3492155，均已核对远端进程。进度按[启动回执](NATIVE_PIPELINE_LAUNCH.json)、[实际控制器状态](NATIVE_PIPELINE_STATUS.json)和[生成状态](NATIVE_GENERATION_STATUS.json)读取；[最终验收](FINAL_AUDIT.json)在评分后产生。

前一轮的[完整结果](../p2_avatarforcing_table4_framegrid_20261110_v2/OFFICIAL_RESULTS.md)已验收，但主口径和完整DIM候选均0/8。官方没有公开完整Table4评分入口，SID训练聚类中心和跨片段汇总等细节仍需明确标记为推断。其他四行的32项来源已核清，见[基线出处](../p2_avatarforcing_table4_full_reproduction_20261110/BASELINE_PROVENANCE.json)。

另已启动[作者两条公开演示的同前端动作读取](published_motion_control_v2/PROTOCOL.json)，包含两路真实speaker作编码／时序误差对照及两路AF输出；只用于定位，不将两条定性展示当作Table4整批成绩。

作者演示读取的第一次尝试在回执字段构造处退出，保留原数组及错误；修正后独立v2目录的尝试在加载模型前因可用显存10183MiB低于32768MiB而安全退出。已明确排队：`published_motion_control_runner.py`等待完整43条评分／审计完成，再检查容量、提取4路并自动输出[同前端对照](published_motion_control_v2/COMPARISON.md)。不在主生成旁反复申请显存。

新增[官方权重来源核验](checkpoint_source_audit/README.md)：重新下载的两个官方文件与当前加载字节相同。本地文件名虽标反，但参数集合无交集、按参数名加载已逐张量验收，未造成覆盖；运行和封存权重路径保持不变。未获得Table4专属DPO／训练步数绑定信息。

后续[官方256评测输入对照](../p2_avatarforcing_table4_canonical_gt_20261110_v4/README.md)已冻结并排队：85/86路首帧可配准，77路存在>10%的横纵缩放差异。本轮生成及评分不改；接续对照只替换GT／speaker图像来源，完整比较同样的8项。
