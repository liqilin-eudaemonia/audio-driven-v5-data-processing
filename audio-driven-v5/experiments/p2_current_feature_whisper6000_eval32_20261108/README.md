# 当前Feature6000＋Whisper6000：四机32卡双200评测接续

用户授权：AV＋Whisper第6000步完整保存后，用四机32卡测评正在生成的AV＋Feature6000和新Whisper6000，并沿用既有加速方案。总计两组×ResponseNet200＋Seamless200＝800条；Feature已有结果全部保留。

## 模型与控制变量

- Feature：`../p2_current500h_av_feature31_20261107/step_006000.pt`，SHA9049cd752edb21069476bc69d68bcac77c64c4e5cb93ad196e2dd5af7f6a6e84。AV＋双路各0.025 Feature，无Event／Whisper。
- Whisper：`../p2_current500h_av_whisper31_20261107/step_006000.pt`，当前等待完整保存；SHA以实际complete标记及CPU全量核验为准。仅AV损失＋speaker Whisper直连，无Event／Feature。
- 两组保持最新SFT2752、相同400条规划文本／H／Gemma／首帧、seed137、Euler30、视频CFG3／音频CFG7、原FPS和512裁框。训练同500h选择、31卡batch及样本／noise顺序，均可与已完成的当前AV-only6000作单组件参考。
- Whisper特征复用同400条speaker缓存，逐条验证编码器、代码及源音频审计；不重跑ASR、Qwen或GT缓存，不读取未来listener作为推理条件。

## 自动接续顺序

1. `queue_p2_current_dual32_20261108.py`已后台挂好。观察训练达到6000后写本轮训练STOP，原生训练完成正在保存的完整模型／Adam／31份RNG后退出；不kill训练进程、不改训练源码，不停止其他任务。STOP保留，可后续显式断点恢复。
2. 等待完整6000标记及原31卡控制器退出，确认实际权重SHA／有限性／身份。Feature推理版已先行导出并逐张量一致；Whisper完整保存后同样导出。完整训练文件保留，推理worker采用已验证模型文件的只读mmap及size／mtime／ctime前后绑定，避免32卡重复扫描Adam大文件。导出不改变精度或张量。
3. 接管Feature旧CPU协调器，保留其GPU7当前生成进程及样本锁。其余卡共享领取未完成样本；同一次事件不会重复提交。每张卡做完Feature后可直接转Whisper，不必等待其他卡长尾；Whisper CPU导出与Feature生成重叠。若Feature在切换前已进入原解码／评分，则保留该原流程，复用其最终结果。
4. 每卡做完生成即可进入解码，不等待其他卡或node11整机；最多4解码进程／GPU，让GPU VAE与CPU编码交错；每节点启动并发限2，显存不足等待，遇OOM将该卡解码并发降至2后有限重试。保持原VAE、tile、FFmpeg和音频裁时。
5. 每模型32个固定评分分片，某分片的媒体齐全即可提前评分，两组交错推进；每GPU每队列最多2路，实际按任务数量及显存余量启动。沿用后台2条CPU预取、4线程PSNR／SSIM、batch64、GPU人脸／表情、无损GT和特征缓存，末尾合并全200特征重算FID/FVD。音频规则与N/A覆盖保持。

## 四机与资源约束

node12 172.29.252.37、node8 172.29.104.237、node4 172.29.225.51、node11 172.29.189.119，各物理GPU0–7。cluster.json给出key／主机及映射。推理每卡一个样本，跨卡不做DDP，保持seed与采样数值，只改变任务领取顺序。每节点最多同时启动2个新进程，减少共享盘／CPU突发。

渲染新任务需该卡至少80GiB空闲；原GPU7 Feature进程作为已运行任务接管。不会杀掉node11其他任务。SSH认证前瞬断会重试；所有GPU进程与样本都有独立锁和记录。协议变更仅为32分片与四机资源，原Feature协议在其distributed32/original_protocol.json备份。

## 入口、状态、验收

- 入口：scripts/p2_current_dual32_eval_20261108.py prepare|run|worker；模型、导出及原生适配位于src/p2_current_dual_eval_20261108.py。
- 当前门槛：GATE_STATUS.json；实际四机阶段：status.json、RESOURCES.json、processes/和logs/。
- Feature结果仍位于`../p2_current500h_feature6000_sft2752_dual200_20261108/`；Whisper新结果位于`../p2_current500h_whisper6000_sft2752_dual200_20261108/`。
- 原生权重导出／逐张量校验：exports/*/RECEIPT.json；CPU源码、编码器、同训练控制检查：CPU_ACCEPTANCE.json；四机共享文件绑定实测：FOUR_NODE_EXPORT_PIN.json。
- 4项回归测试覆盖32分片完整性、接管活动样本不重复、成功样本不重跑、模型文件变化拒绝。源码快照code_snapshot/、CODE_SHA256.json。
- 真实32卡模型加载及产出要等6000落盘、切换后查看各模型distributed32/model_acceptance和progress，不能将CPU准备当作32卡已启动。
- 完成以各模型FINAL_AUDIT.json与本目录FINAL_AUDIT.json为准。COMPARISON.json对照当前AV-only和联合6000。

本目录STOP可取消后续接续；如已进入GPU阶段，各样本可先完成当前提交。没有恢复Goal、聊天定时巡检或无关训练。


## 2026-11-08：去除整机／全量阶段等待

- 每卡独立完成Feature→Whisper生成后，立即领取已有latent的解码工作；不等待node11其他卡完成。
- 四节点显存探测和旧GPU7进程存活探测均为异步。某节点SSH变慢不阻塞其他节点调度；缺少新鲜显存数据的卡暂不接新任务。
- 分片按32个固定逻辑分片保留；哪个分片的真实媒体已经齐全，就将其送到任意已释放且显存足够的GPU评分。解码最多4路／卡，遇OOM降为2路；评分最多2路／卡，避免与本卡生成模型争资源。
- 提前评分的GT字段直接复用同双200、已验收的AV-only原生绑定；只替换本次生成文件路径。最终会运行原生全量bind，逐行核对提前评分清单完全一致，再从各200条特征重算FID/FVD。不会将小分片FID/FVD直接平均，也不会漏掉慢样本。
- Feature与Whisper各自齐全即可独立汇总验收，不必等待另一模型评分结束。最终800条总验收仍要求两组完整。
- 代码：`src/p2_progressive_eval_20261108.py`（资源探测、分片绑定及后处理）与`scripts/p2_current_dual32_eval_20261108.py`（跨卡领取与阶段衔接）。状态：`PROGRESSIVE_STATUS.json`、`PROGRESSIVE_FAILURE.json`、各模型`PROGRESSIVE_BIND_AUDIT.json`和`FINAL_AUDIT.json`。
- 本次11项CPU测试通过：包括慢节点不阻塞、半成品准备不被读取、未齐分片不发布、真实评分行与原生结果相同、最后样本缺失拒绝汇总，以及既有模型绑定／共享领取回归。实际GPU推进以运行状态为准。
- 不停止其他vLLM，不改变400条输入／权重／SFT2752条件／推理采样／评分数学，不恢复Goal或聊天巡检。


## 完成与可视化

2026-11-08 08:32:47 UTC，两组各400条最终验收通过；完整接续耗时22.92分钟。最新总页已收录两组，两个测试集各18行、30项差值对照。840项数值／差值及浏览器验收通过，记录和改动前备份见`dashboard_update/`。

[打开Seamless总结果](http://localhost:8876/listen_benchmark_review/p2_responsenet_random200_20261105/?dataset=seamless&ablation=joint_whisper#ablation-joint_whisper) · [打开ResponseNet总结果](http://localhost:8876/listen_benchmark_review/p2_responsenet_random200_20261105/?dataset=responsenet&ablation=joint_whisper#ablation-joint_whisper)。
