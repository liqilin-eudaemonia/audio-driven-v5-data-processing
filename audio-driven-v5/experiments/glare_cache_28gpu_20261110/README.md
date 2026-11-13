# GLARE 500h 缓存：四机 28 卡准备

**当前执行：用户已授权并启动四机28卡全量缓存提取。** 284,218pair／946,304韵律块；沿用已验收GPU FER配置，复用已有结果。四机看护及完成后自动汇总进程已启动，node11 GPU4–7保留。[实际运行、输出和恢复](production_gpu_fer/README.md)。本条覆盖下方测试阶段“未全量启动”记录。

**最新实测完成：** GPU批量MTCNN接入后，同批640pair＋4,130韵律块端到端202.11秒→164.11秒（再减18.8%；较最初共减28.6%），失败0。有效反应掩码一致，2条共23个有效smile/laughter值有微小数值差异（最大0.0018572），不能称逐位等价。全部测试已结束，正式全量尚未启动。[完整结果与可复用配置](gpu_fer_fixed_640/README.md)。覆盖下方“正在测试”历史记录。

**最新状态（2026-11-10，覆盖下方历史准备／停止记录）：** 用户已重新授权四机28卡测试。旧四组对照已按后续指令停止，不再等待其全部完成。相同640 pair＋4,130韵律块的端到端实测为229.72秒→202.11秒，耗时减少12.0%、吞吐增加13.7%；LIA、标准化音频、动作统计、反应及mask逐张量一致，全部韵律原答及官方归一化结果一致。[实测证据](cpu_tuned_6/OUTPUT_COMPARISON.json)。当前正在运行[2,560 pair＋8,202韵律块测试](cpu_tuned_large/config.json)，使用CPU检测服务及解码池跨批复用、仅加载当前任务索引、每卡1个LIA worker、6检测服务／4解码进程与有界异步写盘。目标为缩短全部缓存端到端耗时，GPU利用率仅用于定位瓶颈；短测收益不能直接当作全量收益。正式全量生产仍未启动，原生产STOP保留，node11 GPU4–7禁用。[最新结果和任务指针](LATEST_ACCELERATION.json)。

**新增可切换提速代码：LIA每卡1／2 worker，以及LIA／韵律同步／异步写盘。** 生产默认仍为1 worker＋同步写盘；[四组固定样本测试入口](benchmark_ready/TEST_PLAN.json)准备完成后可逐组启动。CPU验证通过不代表已证明GPU提速；用户STOP保留，没有启动测试。

## 立即测试的四组配置

| 配置文件（`benchmark_ready/`） | LIA每卡进程 | LIA／韵律写盘 |
|---|---:|---|
| `1_sync.json` | 1 | 同步，基线 |
| `2_sync.json` | 2 | 同步，单独检查多进程收益 |
| `1_async.json` | 1 | 异步，单独检查写盘重叠收益 |
| `2_async.json` | 2 | 异步，检查组合收益 |

生成命令只准备，不启动：

```bash
python scripts/prepare_glare_cache_benchmark.py
```

默认固定约640个真实pair、至少4096个官方韵律源块，精确成员见`cohort.json`。四组使用相同来源、模型、提示、输入、精度与批量；独立输出目录，强制重算LIA，禁止借用旧LIA结果造成虚假提速。Wav2Vec2/SER沿用已完成的只读缓存；本测试不修改生产缓存，不允许发布为生产训练清单。

**重新收到测试指令后**，依次执行下列命令，等一组worker退出再进入下一组；实际命令亦保存在`TEST_PLAN.json`：

```bash
python scripts/run_glare_cache_cluster.py --config experiments/glare_cache_28gpu_20261110/benchmark_ready/1_sync.json start
python scripts/run_glare_cache_cluster.py --config experiments/glare_cache_28gpu_20261110/benchmark_ready/2_sync.json start
python scripts/run_glare_cache_cluster.py --config experiments/glare_cache_28gpu_20261110/benchmark_ready/1_async.json start
python scripts/run_glare_cache_cluster.py --config experiments/glare_cache_28gpu_20261110/benchmark_ready/2_async.json start
```

将末尾`start`换成`status`或`stop`可查看／停止该组。需要反序复测时，用`prepare_glare_cache_benchmark.py --out <新的目录>`生成同样本的新输出目录，不把已完成目录的缓存命中速度当作重新提取速度。计时应同时报告含加载／JIT的端到端耗时和各阶段吞吐；两个阶段的负载比例可能导致部分卡提前完成，不能只看瞬时GPU利用率。

两进程模式仍只用原28张卡：最初8个韵律进程＋40个LIA进程，共48个GPU进程；全部转为LIA时最多56个进程。每卡两进程平分原CPU解码／检测／在途组额度，日志和worker锁隔离；不会同时在同一卡混跑本流程的Qwen与LIA阶段。

异步队列有上限：LIA默认16个写操作、韵律128个块提交；原子写入、SHA和原生回读校验完成后才记为完成。LIA统计可读待落盘的不可变CPU动作张量，写线程自己的回读仍读取实际磁盘文件。写入错误阻止后续完成回执；停止时等待队列排空，已提交的数据可恢复。

可观察字段：韵律`new`为已落盘数，`scheduled_new`为已调度数；LIA`completed`为已提交完成回执数，`scheduled_completed`为已排队数。`async_writes`包含队列容量、在途数、写入耗时和背压等待。原官方提取和原模型数值代码未改。

**2026-11-10 最新：用户已停止联合测试；STOP 保留，GPU 提取未启动。** 首次启动被其他任务占用卡位的检查拦住，四机核查均无本次看护或worker。随后按用户要求补齐“全部训练前缓存”的代码闭环，只做CPU验证，不恢复GPU测试或训练。旧 node11 GPU7 GLARE 韵律提取已正常停止并保存进度；不恢复该卡，不停止其他任务。`4-77` 按 `4–7` 理解。

## 完整缓存范围

| 训练输入 | 当前代码负责的工作 |
|---|---|
| listener LIA逐帧motion | 复用旧结果，补齐缺失 |
| speaker标准化音频 | 复用已提交输入，必要时补齐 |
| 六类反应与有效掩码 | 保留原生产配置，复用并补齐缺失信号 |
| 动作均值／速度／加速度等统计 | 使用原计算方法，与LIA配套提交 |
| Wav2Vec2投影前特征 `[N-1,9216]` | 旧任务记录284,218条完成；现已加入逐任务复用检查，缺失时调用旧编码器补齐 |
| SER语音情绪概率 `[7]` | 与Wav2Vec2同一缓存一并复用／补齐 |
| 官方韵律强度与有效掩码 | 官方完整提取、NULL零强度，映射到pair并汇总 |

队列中的内部名称 `lia` 代表整组非韵律缓存，**不是只提LIA**。该任务现在必须连同Wav2Vec2/SER检查通过才能提交完成回执。原 `shared_cache/protocol.json` 的 `wav2vec_ser=online...` 是旧协议身份字段，保留原文件以复用回执；新增运行清单明确声明全部缓存范围。

原音频特征绑定已验证的“首帧参考、无历史、N-1目标帧”窗口，不能宣称支持任意随机子窗口。可训练的音频投影、韵律嵌入和生成网络继续参与训练，不缓存其输出。

复用默认核对来源、协议、窗口、形状、提交回执与文件存在，不为复用再遍历全部大张量；回执保留创建时的payload SHA。最终汇总支持显式 `--verify-payloads` 做完整重读校验。此次抽核3个真实音频特征文件的SHA通过；旧完成计数不是本次重新全量校验。

| 机器 | 地址 / SSH | 外部入口端口 | 本任务物理 GPU |
|---|---|---|---|
| node8 | root@172.29.104.237:22 | 20491 | 0–7 |
| node12 | root@172.29.252.37:22 | 20356 | 0–7 |
| node4 | root@172.29.225.51:22 | 20359 | 0–7 |
| node11 | root@172.29.189.119:22 | 20476 | **0–3；4–7 禁用** |

## 入口与任务

- [配置](../../configs/glare_cache_28gpu_20261110.json)
- [运行命令](../../scripts/run_glare_cache_cluster.py)：prepare / commands / preflight / start / stop / status / retry-errors / assemble。
- [节点看护](../../scripts/glare_cache_cluster_node.py)：后台持有自己子进程；失败重启、阶段切换；不使用 pkill。
- [单卡执行](../../scripts/glare_cache_cluster_worker.py)：原韵律和 LIA 流程的执行封装。
- [冻结音频复用与补提取](../../scripts/glare_cluster_frozen_audio.py)：已有缓存不加载模型；仅缺失项启用原Wav2Vec2/SER编码器。
- [全部训练缓存汇总](../../scripts/assemble_glare_training_caches.py)：逐pair列全七类输入、原始来源和划分；缺项不发布最终入口。
- [队列与分片](../../src/glare_cache_cluster.py)：不可变 SQLite 路由，文件锁领取任务，原子完成回执。

冻结成员仍为 284,218 pair、约 500.001h，原 train/dev 划分保留。韵律包含 41,433 条 speaker 长音轨、946,304 个必需块；LIA 沿用 245,159 个既有相邻 pair 组。两阶段分别 224 个任务，整条音轨或原 pair 组不跨任务。已完成特征按原回执复用，已完成音频不再提取。

最初每机 2 卡韵律，共 8 卡；其他 20 卡 LIA。一个阶段无可领取任务时，进程退出释放模型和显存，节点看护把空卡转到另一阶段。实际推理使用一进程一卡，不使用 DDP 屏障。CPU 解码进程与反应服务不占 GPU。

## 提速措施与保持的规则

1. 韵律使用官方 Qwen2-Audio 分析器，初始 batch64、源音轨预取2、至多128块排队；OOM 自动拆分并降低后续 batch。H200 上按实际余量分配显存，上限96GiB，至少预留4GiB。沿用官方 FP16、原提示、贪心、64新token，无提前截断或自加时长提示。历史缓存合同里的 batch_cap16 保留作身份记录，新的操作 batch 单独记录。
2. 一条长音轨只交给一个任务，在该次运行中解码和官方响度归一化一次；跨来源拼 batch。LIA 保留原相邻 pair 共享解码、CPU 进程预取和有界 `/dev/shm`，无需保存全量 RGB。
3. 模型跨多个任务复用。Qwen 权重在节点上校验一次，单卡以文件属性核对已验证回执，避免每张卡都重复哈希大权重；LIA 模型和既有 fused2 图跨任务复用，保持 batch4 / FP32。在 SM90 上采用原已验收 fused2，其他架构回退原生执行。当前四机均为 H200。
4. CPU 按整机预算分配，考虑 MP / FER 子进程及其线程；初始每个 LIA worker 4 个解码进程、3 个检测服务、最多7组在途。node8/12/4 CPU预算各128，node11预算64；再按实际配额和共享内存下调。编译缓存放各节点 `/tmp`，减少共享盘竞争。
5. 动态领取224组任务，按完整来源／组的时长均衡，避免固定七等分或四等分的长尾。任务完成必须发生在原生结果提交、覆盖数校验之后；进程崩溃由操作系统释放任务锁，恢复时只补缺失阶段。

GPU worker 和最终条件汇总显式沿用原来的 `runtime_deps + glare-deps + src + scripts` 依赖覆盖路径（Transformers 4.46.3）。基础 avatarforcing 环境本身的 4.30.2 不含 Qwen2-Audio，不能单独拿它启动。预检实际导入官方所需类，不能只检查 `transformers` 这个包存在；不安装或升级共享环境。

**官方优先原则不变：** 官方归一化为 NULL 时使用零强度，原始拒答、格式问题、截断统计只作诊断，不据此丢 pair。真实 IO / 推理执行错误重试；任务连续3次返回此类错误后单列，继续其他任务，不伪造 NULL，也不把失败任务记作完成。修复输入后可显式重新开放失败任务。

## 操作

在项目根目录，使用现有 GPU 环境 Python 或能够读取本项目的 Python 调用：

```bash
python scripts/run_glare_cache_cluster.py prepare
python scripts/run_glare_cache_cluster.py commands
python scripts/run_glare_cache_cluster.py preflight --full-assets
```

以上命令不启动 GPU 任务。`preflight --full-assets` 只做 CPU 校验、文件读取和 nvidia-smi 查询；模型权重校验回执保存在各 node 目录。启动时会重新核查分配卡、共享路径、输入指纹与权重回执；其他任务占用卡时拒绝新启动，不终止其他任务。

**收到启动指令后才执行：**

```bash
python scripts/run_glare_cache_cluster.py start
python scripts/run_glare_cache_cluster.py status
```

`start` 把四个节点看护放到后台，不依赖 SSH 会话持续连接。同一命令可接续该 run；已有节点看护不会重复启动。旧 GPU7 控制器、旧四卡控制器不能与本流程同时写缓存；共享文件锁阻止其重叠，即使启动端退出，节点及单卡仍保留此约束。

```bash
python scripts/run_glare_cache_cluster.py stop
python scripts/run_glare_cache_cluster.py retry-errors
python scripts/run_glare_cache_cluster.py start
```

`stop` 仅停止本 run 的子进程，等待当前有界工作退出；原缓存保留。`retry-errors` 在修复输入后恢复失败任务，不清理已完成缓存。模型／接口／指纹等系统性错误会让相应 worker 报错并最多恢复3次，不能以 NULL 掩盖。

全部队列及看护完成后，用官方规范化和逐帧编码器汇总 pair 条件，再自动导出包含全部缓存的train/dev清单：

```bash
python scripts/run_glare_cache_cluster.py assemble
```

此操作准备训练缓存，不启动训练；全量训练读取器、训练超参与短训验收属于后续接入工作。未实跑28卡，本次不给未经实测的提速倍数或完工时间。

需要额外完整重读payload时，在已有官方条件汇总基础上显式执行：

```bash
python scripts/assemble_glare_training_caches.py --verify-payloads
```

## 保存位置

- 准备清单、路由、任务回执和运行状态：`run/manifest.json`、`run/routing.sqlite`、`run/tasks/`、`run/status.json`。
- 各机器与单卡日志：`run/nodes/<node>/supervisor.log`、`gpu<id>/worker.log`；实时节点状态为 `run/nodes/<node>/status.json`。总状态由 `status` 命令刷新。
- 原始韵律回答继续写入：`../glare_current500h_prosody_subset_20261110/cache/<原合同ID>/chunks/`。
- LIA / reaction / statistics / audio 继续使用：`../glare_current500h_shared_cache_20261110/cache/`；旧父缓存仍可引用。
- 汇总后韵律条件：`../glare_current500h_prosody_subset_20261110/official_conditions/<policy_id>/`。
- 汇总清单入口：`../glare_current500h_prosody_subset_20261110/full/LATEST.json`。
- 全部缓存训练清单入口：`run/TRAINING_CACHES_LATEST.json`，指向`run/training_cache_exports/<版本>/train.jsonl`、`dev.jsonl`；每条的`glare_caches`列全输入路径。
- 验证证据：`validation/`；四机环境结果 `run/preflight.json`。既有全量音频及原始数据不搬移。

## 已验证与限制

CPU 测试覆盖：精确28卡分配、旧 node8 地址拒绝、整音轨路由、LIA原查询接入、模型跨任务复用、每次任务启用进程解码、多进程互斥、崩溃释放锁、原子完成、失败重试、停止后不误记完成、官方 NULL 与原始质量诊断不触发淘汰。还进行四机共享文件锁互斥／释放检查及只读环境预检。

本次没有加载 GPU 模型，没有做28卡实测。因此代码准备和 CPU 验证通过不等于已完成集群吞吐或模型前向验收。启动后应先核对各卡实际产出、OOM拆分、CPU解码等待和滚动速度，再据实调整并行比例或 batch。

全缓存接续新增9项CPU检查通过：旧缓存不加载模型、缺失项转交原编码器、缺失payload保留旧回执再恢复、窗口／来源不匹配拒绝混用、停止标记阻止加载、缺缓存不记任务完成、最终清单缺项不发布、权重验证增量复用。另有原15项调度检查；GPU补提取路径尚未实测，须在用户重新允许测试后验收。
