**已按用户要求暂停（北京时间2026-11-13 14:57:18）：** node11 GPU7任务、自动恢复和ResponseNet后续守护均已停止，已有产物保留。资源策略当前不允许任何GPU，需用户重新授权才能恢复。[暂停进度与恢复说明](../p2_joint_qwen_ltx_test64_16gpu_20261113/NODE11_PAUSE_README.md)。以下运行安排为暂停前历史记录。

# Seamless完整test6624：L2L / ListenFormer / DyStream / Avatar Forcing / DualTalk

**最新资源安排（2026-11-13）：仅使用node11物理GPU7，GPU4禁止用于本轮及后续队列。** 以下旧GPU4/7记录为历史安排，当前以共享`RESOURCE_POLICY.json`、运行时设备检查和最新Monitor为准。

用户要求将参考方法评测扩展到**完整Seamless test6624条**。本目录为独立全量运行；历史ResponseNet200＋Seamless200已完成，结果保留在[双200结果](../p2_reference_methods_dualtest_20261112/RESULTS.md)。完整ResponseNet586已作为本轮结束后的后续队列注册。

## 模型和指标

沿用双200已验收的模型、必要适配和seed137。DyStream、Avatar Forcing、DualTalk使用官方权重；ListenFormer使用本地ViCo完整200轮重训best.pt；L2L使用Conan/Fallon/Stephen/Trevor四套官方人物模型，分别报告，不按GT选优。资产位置与SHA见[模型清单](MODEL_MANIFEST.json)。各方法原始推理入口见[方法交接索引](../../related/method_handoffs/README.md)。

DyStream/AF输入speaker音频和listener首帧，listener音频为零，AF user视觉为零。L2L、ListenFormer、DualTalk从实际speaker视频提取必要动作。`features/`不读listener未来；`evaluation_gt/`仅用于评分。ListenFormer原生256视频显式升512；L2L视频标注公开GPAvatar渲染；DualTalk仍仅报告原生/代理动作，RGB为N/A。

统一视频八项为PSNR、SSIM、LPIPS、FID、FVD、Emo-Acc、Emo-Div、ΔResp。Seamless face-padding0、25fps、512评分；FID/FVD在全6624范围的有效特征上汇总，不平均分片分数。缺脸或官方首帧追踪无身份沿用N/A并记录覆盖，技术执行异常重试/报错。没有生成listener音轨，音频指标N/A。原生DECA动作诊断与统一RGB八项分别列出。

## 固定清单与复用

- `protocol.json`只包含Seamless6624，`RUN_SCOPE.json`强制执行范围，`BATCHES.json`保存265个固定批次：264×25＋最后24条。
- 前200条保留双200的原8个批次及顺序，L2L复用这些采样结果；其余6424条按固定ID顺序分批，每人物每批seed137。
- 复用200条的输入、speaker/GT特征、首帧身份和模型输出。源文件逐条哈希核验，旧目录通过只读复用链接接入，新完成回执记录原出处；见`REUSE_COMPLETE.json`。
- 仅复用代码指纹完全匹配的评分分片。评测代码在`pinned_metrics/`按字节固定，使用原数据目录；不修改共享评测源码。全量新/复用分片仍需代码与资产一致才合并。
- `MODEL_VERIFIED.json`、`PREFLIGHT.json`记录启动前的6624清单、批次、reader与权重核验。清单来源于已完成输入审计的完整测试集，生成时再次核验实际音频/首帧。

## Monitor与GPU调度

**只使用node11（172.29.189.119）物理GPU7，GPU4禁止使用。** 每20秒核验GPU UUID、显存和其他会话GPU进程；连续没有其他作业180秒后接入。本轮改为模型常驻：每个方法／阶段保留一个进程，加载一次后连续处理全部未完成输入；依赖未就绪时在同一进程内等待。删除正常调度命令中的条数限制，关闭15分钟定时退出。

出现前台GPU任务时，对本目录登记的进程组发SIGSTOP暂停计算，保留CUDA模型与Python状态；前台结束、空闲稳定后对同一进程组发SIGCONT继续。暂停期间仍占用模型显存。启动按实际显存准入，GPU7预留40GiB，最多12个不同阶段worker；只清理未使用的CUDA allocator缓存，保留活跃权重。进程技术故障会报错并有界重试，恢复进程时才需要重新加载。

评分改为单个`scoring_service.py`进程：跨方法／25条分片复用原评分器的Networks和Faces实例，包括LPIPS、FID、FVD、人脸和情绪模型。固定评测源码、权重、参数、批次和结果合并方式保持原协议；缓存适配器指纹额外写入运行回执。真实单片段双次评分会对照已有分数与特征，通过后在同一进程继续正式分片。检查结果写入`SCORER_CACHE_PARITY.json`，模型实例创建次数写入`SCORER_RESIDENCY.json`。

Monitor为`monitor.py`，负责资源采样和恢复意外退出的协调器；`controller.py`负责待办阶段准入和断点续跑。只有本目录登记的进程会被接管。技术失败最多自动重试3次，达到上限后状态变为`needs_attention`并保留错误；其他可执行阶段继续。

```bash
cd /private/liqi_workspace/project_code/audio-driven-v5
PYTHONUTF8=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
  compare/.envs/vico_more/bin/python \
  experiments/p2_reference_methods_seamless6624_20261112/monitor.py
```

`monitor.lock`及`controller.lock`防止重复启动。Monitor启动记录在：
`../p2_vico_additional_baselines_20261107/jobs/reference_seamless6624_20261113_monitor_resident_v1/`。

## 看进度、结果和日志

- [全量结果表](RESULTS.md)、[CSV](image_metrics.csv)：仅全范围汇总完成后填最终数值。
- [实时状态](status.json)：动作、视频、N/A、评分覆盖、活跃worker、GPU分配与错误。
- [Monitor状态](MONITOR.json)、`monitor_resources.jsonl`：资源采样与协调器PID。
- `monitor/controller_XXX.log`：协调器日志；`processes/<任务>/run_XXX.log`和`exit_XXX.json`：真实worker日志和退出码。
- `metrics/<方法>/seamless/<批次>/attempt_XXXX/`保存原评分器输出。重试使用新目录，保留失败现场。
- `results/*_seamless.json`为全6624视频结果，`*_seamless_native.json`为原生动作诊断。
- **只有根目录`COMPLETE.json`生成后，才代表所有计划全量结果完成**。Monitor已启动或缓存已准备，不等于6624推理完成。

当前阶段及实际启动验收以状态和回执为准。全量运行时间将根据GPU7实际资源与吞吐更新，双200耗时不能直接作为本轮完成承诺。

实际启动验收见[STARTED.json](STARTED.json)。已核验31个唯一模型资产、全部6624条引用及265个批次；Monitor已实际接上空余GPU7，且AF、DyStream、ListenFormer及共用特征已有历史200之外的新产出。L2L原有200条采样批次保留，新增批次待对应25条特征就绪后自动生成。早期曾扩至6个GPAvatar渲染worker，后改为2路短任务；当前改为每阶段常驻一个进程，避免频繁冷启动。DyStream沿用已验证的加速方式，省去未用于评分的speaker音轨合成副本；原模型输出帧及正式视频导出保持。

## 资源让位修正

初始策略仅按显存准入，虽然没有停止其他会话任务，仍造成GPU4原有生成显著变慢并使解码等显存。已释放18个本轮进程组，保存所有完成产物和调度记录（`RESOURCE_YIELD.json`）；没有停止LoRA/RNA。

当前策略见[RESOURCE_POLICY.json](RESOURCE_POLICY.json)，以上方常驻策略为准。`RESOURCE_YIELD.json`和[资源协调验收](RESOURCE_COORDINATION.json)保留早期释放18个本轮进程组的历史记录；旧的GPU4／GPU7分配、2路小批次和15分钟退出规则已被最新用户指令覆盖。

## ResponseNet586后续队列

用户已授权继续测试完整ResponseNet586，仅使用GPU7。已由独立CPU守护`followup_watch.py`注册在`FOLLOWUPS.json`；本轮Seamless6624的COMPLETE.json生成后，自动接续ResponseNet的Monitor，沿用同一份空闲资源策略。当前不启动后续GPU工作。[后续队列状态](FOLLOWUPS_STATUS.json) · [ResponseNet586说明](../p2_reference_methods_responsenet586_20261112/README.md)。

## 2026-11-13 Monitor计时缺陷修复

Monitor进程并未退出；此前大部分时间因前台任务占卡而让位。另发现GPU7旧租期未在“仅前台作业运行/长期无后台任务”期间清除，01:44:52恢复的任务在01:45:36被旧时间片误判超时，只有约44秒。这是调度实现缺陷。现已在前台期间和长空闲后清理旧租期，短间隔仍保留正常让位窗口，新任务使用新租期；5项针对性回归通过，见`LEASE_REGRESSION.json`。

最新用户资源要求已同步至Seamless6624和后续ResponseNet586：只准入GPU7，设备层在初始化CUDA前拒绝GPU4。GPU4上的本轮任务已释放，已完成产物保留。

## 2026-11-13 模型常驻改造

`RESIDENT_CPU_CHECK.json`记录四项通过的检查：同一PID暂停／恢复、相同配置复用模型实例、恢复原独立评分进程的精度设置、控制器无条数退出和定时拆卸。该检查不分配CUDA。实际GPU评分一致性和长期运行证据分别由`SCORER_CACHE_PARITY.json`、`RESIDENT_LIVE_ACCEPTANCE.json`记录；文件未生成时不视为已经通过。

本轮改造只调整进程生命周期和缓存复用；已完成的推理／评分回执继续复用，不因改造重跑。

调度进度读取使用`progress_scan.py`，只发现已有样本目录中的完成回执并缓存已完成项，避免每轮访问数万个未来文件路径；仍限制在冻结样本清单中，新目录与稍后补齐的回执可继续发现。检查见`RECEIPT_SCAN_CHECK.json`。接入时仅恢复CPU协调器，保留所有GPU worker的PID及模型；前后PID证据见`CONTROLLER_RELOAD.json`。

首次常驻评分检查发现生成端设备初始化覆盖了原独立评分进程的cuDNN精度设置，已用独立评分器原设置的上下文修复；原评分代码和权重不改。失败进程未提交正式评分分片，其他GPU worker继续运行。记录见`SCORER_PRECISION_FIX.json`，修复后的实际一致性以`SCORER_CACHE_PARITY.json`为准。

实际GPU验收已通过：`RESIDENT_LIVE_ACCEPTANCE.json`记录node11 GPU7现场、11个阶段进程在CPU协调器接管前后的相同PID、超过旧条数上限的连续产出，以及评分模型跨两次真实计算仍只构造一次。`SCORER_CACHE_PARITY.json`的两次原分数／FID-FVD特征一致性检查均通过，评分服务已继续正式分片。完整6624尚在运行。

最新耗时复核见[ETA_RECHECK_20261113.md](ETA_RECHECK_20261113.md)：使用更长的实际计数窗口和12个正式评分分片，当前配置下Seamless约9–15天、ResponseNet接续额外1–2天；这是条件外推，替代早先较窄的完成日期估计。

最新范围与资源测算：用户确认L2L四套均测，当前四套队列继续；[单卡／假设8卡的进度与耗时表](GPU_SCALING_ESTIMATE_20261113.md)。8卡仅为规划估计，实际仍只使用node11 GPU7。

[当前已测方法、数据集与实际产物统计](EVALUATION_INVENTORY_20261113.md)：ViCo43及双200已完成，两项全量暂停，固定200缓存不重复计数。
