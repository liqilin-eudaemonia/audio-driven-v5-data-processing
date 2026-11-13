# GLARE final32940 · Seamless完整test6624

**最新状态：用户要求暂停32940步，只继续19050步（2026-11-12）。** 控制器及全部在途评分进程已停止，6624条生成视频、79个已完成评分块（2472条）及共享输入／韵律缓存保留。见[暂停记录](PAUSED.json)。启动入口检测到此记录会直接保持暂停；恢复需先明确撤销暂停标记，再运行原入口。以下恢复说明为暂停前历史记录。

**2026-11-12恢复说明：** GT `naturalistic_V03_S0208_I00000490_local_4f4afb744d0a`按冻结网格需51帧，原读取器只能读到50帧；三个评分尝试一致。已保留原生allow-partial结果及该ID的N/A，不补帧、不静默删除请求行；有效覆盖会少于请求数，证据见`source_issues/042.json`。其余数据恢复处理。暂以GPU4优先完成19050步剩余评分，同时本队列4路生成继续；19050的COMPLETE出现后自动恢复本队列评分，见GPU4_PRIORITY.json及RECOVERY_20261112.json。此前已完成视频与评分块全部复用。

2026-11-12用户要求在ViCo43和ResponseNet586完成后接续。前两套已[全部完成](../glare_final_vico43_responsenet586_20261112/RESULTS.md)，此任务使用同一最终检查点，仅node11（172.29.189.119）物理GPU4；其他GPU任务保持。

固定完整test：6624条、11.047173小时；参考清单为`OmniResponse/evaluation/seamless/prepared_20261102/references.jsonl`，seal／源行／裁框／读取器身份均校验。前200条是原固定诊断集，单独汇总，其余6424条也全部评测。全量和子集不得直接混排。身份、原始行、参数、209个互斥评分块、输入引用和SHA见 [protocol.json](protocol.json)。

最终权重：`../glare_paper_settings_node11_gpu7_20261111/latest.pt`，step32940，SHA256 `4f42e95fbe07b2c391a587da7e6cce3a2c1c8731a602ce0b5c86af29b2b40bb5`。使用speaker波形＋listener首帧，官方学习到的空动作统计；不使用listener未来、音轨或GT描述。原生FLOAT／512／25fps、seed137、nfe10、speaker CFG2、零历史；反应监督等沿用已训练配置。

提速复用前一轮真实验证的路径：16路有界CPU输入准备，200条直接复用已核验的PNG／speaker WAV；官方离线韵律跨输入batch64；4个常驻GLARE进程、LIA解码batch8、CPU编码重叠。就绪缓存使用追加队列和游标，避免每取一条都扫描6624个文件。先完成韵律后允许最多8路评分与生成重叠，生成释放显存后最多16路评分。每卡范围及UUID强校验，容量保留至少20GiB准入余量。

已实际验证断点恢复：按需释放空闲CUDA分配器缓存后接续，保留此前542条完整视频及已有韵律／输入缓存；原生模型、数值、参数与prompt不变，见`MEMORY_MIGRATION.json`。低于24GiB空闲时才清空未使用的显存缓存，避免多个进程各自长期占用峰值空闲池。源码初始指纹在protocol，运行中这次显存管理适配的实际指纹另存迁移记录。`elapsed_s`为当前控制器运行时间，整体起点以protocol.created为准。

首个25条评分块提前作为真实接入检查，产物直接计入全量，进程见`SCORE_SMOKE_LAUNCH.json`，日志在`metrics/000/video.log`。它不额外生成样本；自动队列稍后通过分片锁及SUCCESS收据接续。

评分沿用Seamless视频8项：PSNR／SSIM／LPIPS／FID／FVD／Emo-Acc／Emo-Div／ΔResp；batch64、GPU检脸／表情、像素线程4、CPU预取和GT特征缓存，**face-padding0**。200条复用原无损GT RGB缓存，其他6424条按需解码，不新增约780GB全量RGB缓存。全量／固定200的FID和FVD均从各自全部特征汇总，不平均分片分数。原检测支持不足保留N/A／覆盖率，不伪造负例。GLARE没有生成listener音轨，音频评分不适用；原生MP4的speaker背景音只供试听。

- 入口：[eval_glare_seamless_test_20261112.py](../../scripts/eval_glare_seamless_test_20261112.py)，复用[已验证的生成模块](../../scripts/eval_glare_final_20261112.py)及常驻原生推理。
- 进度：[status.json](status.json)、`INPUT_STATUS.json`、`PROSODY_STATUS.json`；启动／进程：`LAUNCH.json`、`processes/`。
- 日志：`logs/controller.log`、`logs/inputs.log`、`logs/prosody.log`、`logs/generate_*.log`，各评分块`metrics/NNN/video.log`。
- 视频：`seamless/generated/<id>/generated.mp4`；评分副本：`seamless/scoring/glare_final32940/<id>.listener.mp4`。
- 完成结果：`test6624_RESULTS.json`、`fixed200_RESULTS.json`、对应逐条`*_pairs.jsonl`、`FINAL_AUDIT.json`及`COMPLETE.json`。COMPLETE出现前全量尚未结束。
- 故障：`FAILURE.json`、`INPUT_FAILURE.json`及对应日志。执行错误重试；不改变官方NULL回退或静默缩减6624条。

恢复前先核实node11的LAUNCH／processes中原任务已退出，防止重复启动和显存叠加。设置`CUDA_VISIBLE_DEVICES=4`，使用avatarforcing环境，并把`runtime_deps:glare-deps:src:scripts`放入PYTHONPATH；在项目根目录执行`python scripts/eval_glare_seamless_test_20261112.py run`。`prepare`只冻结输入清单；`run`会恢复READY／韵律／视频／SUCCESS收据，游标从头安全跳过已完成项。已有评分进程由分片锁保护，但失联控制器的其他存活进程必须先核查。不得在运行中更改protocol或原生推理源码。
