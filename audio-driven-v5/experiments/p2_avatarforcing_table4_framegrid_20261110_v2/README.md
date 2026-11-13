# Avatar Forcing Table4：尾帧修正后的正式接续

目标保持不变：复现AF自身8项两位小数，核清其余四行来源。**本轮43条生成、129路提取、8项评分及独立验收全部完成；主口径和完整DIM代码候选均为0/8，未复现Table4。** [原始需求](REQUIREMENTS.json)和[前轮完整记录](../p2_avatarforcing_table4_full_reproduction_20261110/README.md)均保留。

## 为什么另开v2

前轮已完成37条生成，随后在`yTInFHb_o1Q_000516_000551.1`的帧数断言退出。两条同源片段的实际时长都是34.9666667秒：视频按25fps四舍五入有874帧，官方模型按559467个音频采样、每帧640采样向上取整，输出875帧。适配器把这两种计数混为一谈。

v2按官方`AvatarForcing.sample.T`检查输出帧数。模型仍接收原874帧speaker序列，末端运动条件由原代码内部复制；没有修改网络、权重、音频、随机数、CFG、NFE或导出算法。SPECTRE的GT／speaker时间网格随实际生成帧数走，最后的34.96秒仍在原片段内，按最近可用帧取值。没有删除最后一帧来通过断言。

[源调用核对](GENERATION_MATH_CALL_AUDIT.json)确认初始化、参考图／音频预处理、RNG恢复、推理实参和官方导出调用的AST均未改变。37条实际视频只读复用，保留父协议和原始回执SHA；其余6条在本目录续跑。[复用清单](REUSED_GENERATION_MANIFEST.json) · [冻结生成协议](NATIVE_GENERATION_PROTOCOL.json)。

## 本轮最终结论

[主结果](OFFICIAL_RESULTS.md)：FD Exp22.979871／Pose0.089712，rPCC Exp0.057621／Pose0.019046，SID Exp3.412787／Pose2.652574，Var Exp0.523453／Pose0.016406。[完整DIM代码候选](DIM_STYLE_RESULTS.md)也为0/8；原始作者打印函数的一致性检查通过。

[最终独立验收](FINAL_AUDIT.json)确认43条／13575帧完整解码、129路有限系数、全部输入／权重／代码SHA及8项独立重算。工程执行通过，数值复现未通过。下文保留执行过程。

下一项有画面证据的对照是[整段均值框参考图](../p2_avatarforcing_table4_reference_mean_20261110_v3/README.md)，不修改本轮结果。

## 已完成及执行过程记录

- 原始媒体：43对／86路／83个唯一original文件全部恢复。独立核查了canonical speaker／listener角色、归档CRC／大小和文件SHA、全部裁脸文件、代表帧裁剪重编码、视频头、首帧及音频长度。[输入独立验收](ORIGINAL_INPUTS_INDEPENDENT_AUDIT.json)。
- 原始25fps条件视频共每角色13573帧；模型音频时间网格共13575帧，差的2帧正是上述两个片段的合法尾部时间点。原始条件数据不重写。
- 官方Torch2.0.1／CUDA11.8、327个核心参数和210个Wav2Vec状态已核验。新进程必须与原37条保持相同初始化CPU／CUDA RNG摘要，最终控制器逐条核对。
- 生成43条全部完成：37条只读复用，6条实际新生成；两条小数时长片段均成功输出875帧。独立核验全部视频SHA并逐帧解码，43条／13575帧、512×512／25fps和初始化CPU／CUDA RNG一致。[生成完整验收](GENERATION_INDEPENDENT_AUDIT.json)。实时状态：[控制器](NATIVE_PIPELINE_STATUS.json)、[生成](NATIVE_GENERATION_STATUS.json)。
- 系数提取采用SPECTRE正式示例的RetinaFace ResNet50＋2DFAN2、最大可信人脸筛选、JPEG编解码、1.6倍裁剪及五帧时间上下文。前轮6路实测验证及函数体复用证明在[前端验收](OFFICIAL_SPECTRE_PROBE.json)和[源代码核对](FRONTEND_SOURCE_REUSE_AUDIT.json)。本轮129路全部重新提取，不混用父轮旧特征；本轮前32路串行完成后保留，剩余独立任务并行。
- 本轮最先6路已与父轮同输入的8% allocator串行结果对照：关键点和动作系数逐元素最大误差全部为0，见[独立串行重复比较](SERIAL_REPEAT_COMPARISON.json)。它不替代尚未完成的129路覆盖和全量评分。
- 独立视频并行已通过原1e-6门槛并启用。原8%串行重复也只在首次执行的第0帧出现6.10e-5像素差异，第二遍完全相同；4次推理预热后，完整275帧关键点／动作系数与已封存参考最大误差均为0。[原始重复证据](../p2_avatarforcing_table4_full_reproduction_20261110/official_tracker_audit/NUMERIC_REPEAT.json) · [预热后严格验收](official_tracker_audit/PARALLEL_PARITY.json)。没有放宽阈值，也未改变模型权重或数学运算。
- 当前最多4个独立视频worker，仅GPU7，每个allocator上限2.5%、另计1GiB上下文并保留16GiB余量；实测峰值1501MiB。初始化元数据有全局锁，视频独占锁和原子提交避免重复写。已实际通过跨进程文件锁测试。[冻结并行协议](OFFICIAL_PARALLEL_PROTOCOL.json) · [切换时保留32路的记录](EXECUTOR_HANDOFF.json) · [实时提取状态](OFFICIAL_PARALLEL_STATUS.json)。
- 评分冻结FD／rPCC／SID／Var各Exp／Pose两项，保留预声明的汇总诊断。SID训练聚类中心及作者实际汇总仍未公开，GT拟合只标作代理重建。[统计口径](OFFICIAL_METRICS_PROTOCOL.json)。**文件名OFFICIAL仅指SPECTRE前端，不代表AF官方测评脚本。**

输入FFV1容器保持的是ffmpeg解码JPEG后的BGR像素，独立验收逐路确认；OpenCV JPEG解码因色度上采样不同，可有少量像素差异，已量化记录，没有把两种解码器谎称为逐字节相同。

## 运行及验收

仅node11物理GPU7，环境`compare/.envs/avatarforcing_native201/bin/python`，`CUDA_VISIBLE_DEVICES=7`。当前控制器在[NATIVE_PIPELINE_LAUNCH.json](NATIVE_PIPELINE_LAUNCH.json)记录PID。

1. `native_run.py`接续`native_generate.py all`，原37条不重算。
2. 43条生成全部完成后，运行`official_spectre.py all`提取129路，全部时间戳／帧数／权重及输入输出SHA记录在回执。
3. `official_score.py score`输出`OFFICIAL_RESULTS.json`及Markdown，逐项对照8个目标数字。
4. 最后独立核验43条／129路覆盖、尾帧、有限性、源代码／权重／视频SHA、8项数值和基线出处；工程完成标记不自动证明Table4复现。

独立CPU审计已挂接：`audit_when_ready.py`只在真实评分完成后运行`final_audit.py`，重算全部8项、逐帧解码43条并核验129路特征和尾部时间点。[审计等待状态](AUDIT_WATCH_STATUS.json)。只有`FINAL_AUDIT.json`中的8项对照才能支持数值复现结论。

另在全量成绩产生前冻结了一套完整公开代码候选：**直接使用DIM发布的FD／rPCC／SID40／20／拉平方差实现**，在同一组SPECTRE系数上运行。依据是AF转引DIM及附录K9与INFP* PoseSID3.20的上界矛盾；不逐格拼公式，不替换主结果。[事先冻结的候选](DIM_STYLE_PROTOCOL.json)。最终审计后会额外输出`DIM_STYLE_RESULTS.json`和原始作者打印函数的16标量一致性检验；它不等于DIM的EMOCA十二项复现。

基线来源已核清：[32个单元格与出处](../p2_avatarforcing_table4_full_reproduction_20261110/BASELINE_PROVENANCE.json)。AF的DIM行对应原论文的CANDOR预训练DIM-Listener，L2L对应普通L2L行，INFP*为AF作者自行训练200万步版本。官方仍未提供完整AF Table4评分入口，见[公开代码核查](../p2_avatarforcing_table4_full_reproduction_20261110/PUBLIC_CODE_AUDIT.md)。
