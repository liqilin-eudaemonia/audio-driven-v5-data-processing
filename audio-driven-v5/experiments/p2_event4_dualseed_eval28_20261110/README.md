**2026-11-11已全部完成：** 当前训练完整保存Feature=0.10组step80并退出，GPU7重复模型显存检查已修复。Event=1双seed使用既有四机28卡入口补齐，8轮评分和四组对照汇总全部完成，28卡已释放。[指标、覆盖率及验收](repair_20261111/RESULTS.md)。

**当前已按用户要求改为 node11 GPU7 单卡接续（2026-11-10）：** 复用已完成的512条生成／解码、8轮常规指标和5轮完整评分，仅补余下3轮事件评分及最终双seed汇总。已完成实际模型加载并写出新检测结果；沿用原检测权重／提示／采样／类别和冻结GT，显存采用0.36预算＋28GiB CPU offload以保留同卡既有服务，实际执行覆盖单独留档。训练保持暂停，不使用其他节点GPU；完成后等待实验方向决定。[启动验收](GPU7_RESUME_STARTED_ACCEPTED.json) · [执行计划](GPU7_RESUME_PLAN.json)。入口 `gpu7_resume.py dispatch` 已启动，勿重复；进度 `STATUS.json`，错误 `GPU7_RESUME_FAILURE.json`，日志 `gpu7_resume_worker.log`。本目录 `STOP` 可安全停止本轮；完成产物继续复用。

**用户要求安全退出：** 本轮评测控制器及28卡进程已退出、GPU释放核验通过；STOP保留，训练继续暂停。512条生成／解码及8轮常规指标全部完成，5轮完整评分完成，剩余3轮事件评分与最终汇总。后续按用户指令校验来源和收据后接续，不重跑完成产物。见[安全退出验收](SAFE_EXIT_ACCEPTED.json)。

**端口冲突修复与接续：** 先前rank13的vLLM随机TCP端口发生EADDRINUSE，导致保护退出；已为28个worker及3次启动尝试划分独立低端口段，只有该启动冲突做有限重试。按任务日志PID、GPU和进程启动时间确认后清理了3个本轮残留EngineCore；节点取消时跟踪并清理自己的后代进程，失败状态立即写入STATUS。11项检查通过。故障记录保存在failures_before_port_isolation，完成结果保留，滚动队列从现有产物接续。

**跨组滚动调度已启用：** 当前入口`rolling_driver.py run`。生成／解码／常规评分共用跨8组任务池，按卡释放继续接其他权重／seed；最多2组事件评分同时推进，与GPU视频任务重叠CPU准备。每张物理GPU由共享文件锁互斥，事件任务有优先准入，快卡完成本组即可接其他就绪组，不再等待整组统一释放；保留原生采样、指标、缓存、所有已完成结果。10项测试与node4跨机GPU锁互斥实测通过；启动已观察到28个视频任务同时来自5组。状态`ROLLING_VIDEO_STATUS.json`／`STATUS.json`，失败`ROLLING_FAILURE.json`／`ROLLING_VIDEO_FAILURE.json`，完成仍写COMPARISON.json。训练保持暂停。

**CPU阶段提速修正：** 首组64条生成／解码／常规指标已完成，发现独立事件检测的媒体准备和候选提取原来只在单机串行，导致28卡等待；现改为四机28个CPU worker，每片段仍使用同一原生reader及候选规则，按唯一media去重并核验完整64条，随后28卡检测。保留所有已完成生成／常规指标；6项检查与两条真实样本RGB／音频指纹一致性通过。CPU阶段不会占GPU，短暂空卡仍可能出现，不能以显存为空推断任务结束。

# 四组Event × 双seed · 四机28卡评测

范围：Event 0.05／0.01／0.20／1.00，Feature两路各0.025；统一500步检查点、同64条、生成seed137／238，共512条。原seed137已有latent继续复用，SFT2752／Qwen H／Gemma／speaker Whisper缓存共享；Euler30／视频CFG3／音频CFG7及512分辨率保持。

资源：node12／node8／node4 GPU0–7＋node11 GPU0–3，共28卡。node11 GPU4–7不参与；旧GPU7评分队列已停并保留产物，其余服务未停止。

已完整保存并暂停Feature弱组step484，模型／Adam／28份RNG及SHA通过，28卡释放通过。见TRAINING_PAUSED_ACCEPTED.json。训练STOP保留；评测结束不自动续训。

沿用evaluate28.py共享领取、逐卡释放、最多4路VAE解码和2路评分、就绪分片评分、GT无损RGB／特征缓存、CPU预取2／像素4线程、GPU人脸／表情及batch64，最终原生全量bind逐行核验并合并全部特征。

事件评分沿用原生prepare／candidates／Qwen3-Omni infer／score及参考标签冻结逻辑，只将独立匿名media分配给28卡：每个media恰好一个worker，实际host／GPU UUID另行记录并验证，采样／提示／类别定义保持。空闲卡采用80%显存预算、关闭CPU权重offload；其完整GPU验收待实际阶段运行。模型身份／库版本必须在28卡一致，原生检测函数的generate语法树一致性检查通过。

测试：5项新增检查及8份真实评测配置检查通过；原40项双seed／评分回归保持。CPU检查不替代实际GPU验收。配置和源码指纹见PLAN.json／TESTS_ACCEPTED.json。

入口：driver.py run（已启动，勿重复启动）。进度STATUS.json；进程CONTROLLER.json；错误FAILURE.json；各任务evaluations/<组>_500_seed<seed>/eval28.log及status.json。停止：本目录STOP；只停止本评测，训练继续保持暂停。

全部结束写COMPARISON.json／COMPLETE.json：每seed指标和均值、来源组配对bootstrap、两个seed是否方向一致；FID／FVD各在64条上计算后平均，不合并成128条独立样本。不自动晋级、不声称全局最优。
