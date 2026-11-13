# GLARE：论文学习率／有效batch／混合精度对齐试训

**约30轮训练已完成，北京时间2026-11-12 07:50。** 总步数32940，训练启动器正常退出（exit0）。已实际加载验证best.pt与latest.pt，分别为32000和32940步。最佳完整dev loss为0.13933026，最终为0.13938570。最近几次验证变化已很小。当前推理优先使用本目录best.pt，latest.pt保留最终训练状态供恢复。


**用户再次授权增加10轮，当前总预算约30轮。** 总步数32940，node11 GPU7运行v9入口。原前21960步调度继续执行，之后的新增10980步保持末端学习率5e-6。此末段恒定学习率为本次续训实现选择。模型、AdamW、GradScaler、随机数及epoch_order/游标接续，每2000步保存和完整dev验证，早停保持关闭。

当前恢复命令见launch.json，入口为scripts/train_glare_paper_settings_v9.py／run_glare_paper_settings_v9.sh。恢复保留`--steps 32940 --previous-steps 21960 --initial-steps 10980 --extension-start-lr 5e-5 --extension-end-lr 5e-6`等原命令参数。延长前权重保留为before_extension30epochs.pt，旧协议/命令为protocol_initial20epochs.json及launch_initial20epochs.json，接续身份记录为continuation30_config.json。已检查调度边界和计数恢复，并确认真实续训推进到19055步，loss/梯度正常。


**推理入口已准备并通过真实短片验收。** 支持当前自训练检查点，复用官方latent推理、ODE及LIA解码，输出带音轨MP4。官方统计CFG的空历史batch维度兼容在推理实例中完成。[命令、示例视频与验收记录](../glare_inference_20261111/README.md)。


**用户已授权再训练10轮，当前总预算约20轮（2026-11-11）。** 总步数21,960，正在node11 GPU7接续运行，当前入口`train_glare_paper_settings_v8.py`／`run_glare_paper_settings_v8.sh`。模型、AdamW、GradScaler、RNG和采样游标从安全保存的检查点接续。

续训学习率由本次实现确定。原前10,980步的warmup和余弦计划继续执行，之后用新增10,980步将学习率从5e-5余弦下降至5e-6。初始峰值5e-4保留于原计划中。用户指定新增10轮后，原自动早停已关闭，验证及最佳权重选择每2000步执行，latest也每2000步保存。总预算最后再保存和验证。

原10轮配置与启动命令保留在`protocol_initial10epochs.json`、`launch_initial10epochs.json`，延长前检查点保留为`before_extension20epochs.pt`。新旧协议及恢复步数记录在`continuation_config.json`。恢复使用launch.json中的v8命令，保持`--steps 21960 --initial-steps 10980 --extension-start-lr 5e-5 --extension-end-lr 5e-6 --early-stop-patience 0`。Checkpoint中的scheduler_step保存两段调度的统一计数。

已检查调度边界、尾段单调下降、最终学习率及恢复计数。实际冷恢复后已推进到10625步，loss/梯度正常，学习率平滑接续。


**最新采用v7、microbatch128。** 用户释放部分GPU7资源后，128/192训练测试通过，256在当前显存余量下触及限额。生产128约2.43秒/更新，192约2.49秒，最终采用128。effective batch256、FP16、reaction0.05，保存/完整dev验证每2000步。[实测及恢复](../glare_micro128_20261111/README.md)。


**最新运行入口为v6，microbatch32。** 64/128及CUDA传输预取已实测，64未带来生产端到端收益，已恢复32。模型与优化器进度接续，保存/验证每2000步。[试验结果](../glare_training_pipeline_20261111/README.md)。


**最新：反应权重0.05＋真实microbatch32已启用。** 从第181步起调整，前180步旧权重保留；按长度随机分桶、effective batch256、FP16不变。当前入口v4使用打包IPC和4个读取worker，保存／验证仍每2000步。[变更、测试和恢复](../glare_batch_training_20261111/README.md)。下方v2及“反应权重0.2”为历史配置。


**已启动并通过64步试训：** 固定256条dev loss由2.15420降至2.00327（约7.0%）；无FP16溢出。第20–63步约5.77秒/更新，44.3pair/s，约1.76小时/epoch（不含完整dev及保存）。初步结果支持继续，尚不能判断收敛；完整3158条dev在2000步后首次评估。[实测](PILOT_RESULT.json)。


2026-11-11 用户要求对齐这三项，再实测吞吐、按验证表现决定预算。仅node11物理GPU7；保留同卡vLLM与其他卡任务。

## 配置及边界

- 学习率峰值5e-4、AdamW、有效batch256、FP16 autocast＋动态GradScaler。参数权重及AdamW状态保持FP32，累计梯度后unscale、clip1、step；发生溢出时跳过参数更新并记录，调度也不前进。
- 单卡逐pair计算、累计256次；有效样本数对齐论文，但不是论文四卡每卡真实batch64，也不保证两种采样方式下随机条件dropout相关性相同。
- 保留原官方FLOAT前向与既定数据适配：完整pair、首帧参考、无历史，生产六类reaction＋mask，冻结音频缓存。反应loss权重仍为此前0.2，官方README示例0.05；因此只是用户指定三项对齐，不称全配置或完整论文复现。
- 新模型从相同预训练冻结资源及原生FMT随机初始化开始，没有将旧batch8权重当作论文配置起点。旧训练已完成当前更新并保存，原目录全部保留。
- 暂定最多10,980更新（281,060训练pair下约10epoch）；10%warmup、余弦至峰值的1/10。warmup随这次上限固定，不冒称论文650epoch调度。达到上限仍有改善，需要另行明确延长及调度策略，不能悄悄改变恢复身份。

## 验证与训练预算

- 第0和64步：相同、均匀选取的256条dev，固定验证RNG，比较原生flow loss，作为初步稳定性检查，不据此判定收敛。
- 每2000步：完整3158条dev、固定RNG验证；保存latest和改善时的best。
- 连续3次完整dev评估没有相对上次最佳值至少0.5%的改善时，停止并保存；或者达到10epoch上限。单次短测不承诺最终预算。
- 原train/dev成员不变，测试集不参与预算或选模。验证为训练目标loss，不等同生成视频质量指标。

## 入口、结果与恢复

当前入口：`scripts/train_glare_paper_settings_v2.py`，启动封装`run_glare_paper_settings_v2.sh`。准确命令/PID在launch.json，运行配置及指纹在protocol.json。train.log／metrics.jsonl为逐步证据；initial_validation.json是256条初始验证；最新权重latest.pt、最佳权重best.pt（达到保存节点后）。

V2在一次256pair更新内保留autocast的权重转换缓存，backward显式禁用autocast，更新结束退出上下文，避免下次使用旧权重转换值；LRU位置编码／mask复用仍开启。第一版FP16前10步保留在amp_initial_checkpoint.pt，优化器／scaler／RNG／游标全部接续，源码变更有execution_migration.json记录。

恢复前检查现有任务已退出。复用launch.json命令，--resume指向latest.pt，保持所有数据／steps／累积／精度参数。GradScaler、成功更新数及早停计数随checkpoint保存，不能拿旧FP32训练的checkpoint直接恢复这条训练。SIGTERM发给训练子进程会在当前更新后保存退出；不要直接杀进程组造成无谓重跑。
