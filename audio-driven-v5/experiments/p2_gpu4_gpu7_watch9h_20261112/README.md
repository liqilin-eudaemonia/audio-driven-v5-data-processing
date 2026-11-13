# node11 GPU4／GPU7：九小时巡检、恢复与空卡接续

用户明确要求未来九小时每小时检查两卡，任务退出或明显变慢及时处理；GPU7空闲后可分担GPU4的工作。

**固定窗口：2026-11-12 14:52:14—23:52:14 UTC，即北京时间11月12日22:52:14至11月13日07:52:14。** 到期只结束本次巡检，不停止仍在运行的测评。未到窗口结束且完成最后核查前，不能把Goal标记完成。

**九小时巡检已完成，测评继续运行。** 最后现场核验为2026-11-12 23:52:54 UTC：原GPU4／GPU7控制器及生成进程均存活，两卡利用率100%，没有未处理的退出或降速告警。截止时Seamless已导出538／6624；ResponseNet6000完整586条已验收，6500已导出256／586。GPU7仍在运行6500，未满足空卡接续条件，未启动辅助模型。期间处理一次约5倍的算力争用降速，同为289帧的生成已由约540秒恢复至94秒；末次Seamless样本约90秒／273帧。[逐项完成核验](FINAL_AUDIT.json) · [最后现场进程与GPU核验](operator/09_live.json) · [九小时到期快照](WINDOW_FINISHED.json)。此处完成仅指巡检，不表示全集测评完成。

## 当前范围与证据

- GPU4：[LoRA6000＋SFT2752／当前Seamless完整6624](../p2_lora6000_seamless_full_gpu4_20261112/README.md)。
- GPU7：[LoRA6000及6500／ResponseNet完整586](../p2_responsenet_full586_all_lora6000_6500_20261111/README.md)。
- [配置与期限](CONFIG.json)、[后台提交PID](LAUNCH.json)、[实际守护PID](CONTROLLER.json)、[分钟存活检查](LIVE.json)、[守护状态](STATE.json)。
- 每小时证据：`hourly/00.json`至`08.json`，包含GPU UUID／利用率／显存／计算PID、真实/proc进程身份、工作进程、任务状态、输出数和最近20条逐片段耗时。`operator/`为交互代理复核。
- 首次现场两卡均100%，GPU4已导出234条、39条新latent，GPU7的6000已导出487／586条；6500仍由原队列自动接续。完整缓存已就绪。

## 故障和速度检查

后台每分钟轻量检查实际进程，每小时记录完整吞吐证据。只对精确登记脚本／PID／启动ticks匹配的本任务进程操作；从不按卡或模糊名字杀任务，不因为SSH超时或一条旧状态字段判断任务退出。

控制器确认退出、所有登记子进程退出、控制锁释放、冻结执行源码核对通过且没有用户STOP时，后台按原入口断点恢复；每小时最多3次，间隔至少3分钟。仍在产生输出的存活模型保留，不启动第二份。若控制器已退出但工作进程仍存活，记录其状态，等待其完成或由交互代理核查，避免重复模型。

连续15分钟输出未变、至少5次GPU空闲检查、没有外国计算进程、渲染器CPU也没有推进时，才停止精确匹配的本任务渲染进程，让仍存活的原控制器断点重试。CPU合并／绑定或显存准入等待不按此规则重启。

速度对比记录各条`seconds/native_frames`；只在中位片长相近（相差≤16帧）时比较两组各10条样本。两组都超过初始基线1.5倍时写`NEEDS_OPERATOR_REVIEW.json`，交互代理每小时进一步检查并针对实际原因修复。不能因为长片段本来更慢就重启健康计算。

`repairs/`保留所有自动恢复／停滞处理证据，`errors/`保留异常，`watcher.log`为日志。所有已有缓存、生成、latent与评分块保留。7项执行检查覆盖逐样本互斥及崩溃释放、两权重优先级、截止时间、PID复用、忙卡排除和片长匹配的降速判断。

## GPU7释放后接续Seamless

ResponseNet的**6000和6500两份权重均完成且最终验收成功**，其GPU工作进程全部退出，GPU7无其他CUDA进程且至少90GiB空闲后，才允许启用[GPU7辅助生成入口](../../scripts/p2_seamless_gpu7_assist_20261112.py)。不会把6000结束、6500待启动的短暂空闲当作整卡释放。

为避免两卡重复生成，Seamless新增逐样本`flock`，锁覆盖原生生成到atomic提交，进程崩溃自动释放。当前已存活的GPU4模型仍沿用加载时的代码，保持计算；只有需要真正加入GPU7时，在下一条latent完成后短暂替换GPU4渲染进程以加载共享锁。控制器会继续原队列；写出新进程与共享锁协议证据后才允许GPU7加入。

辅助卡读取同一冻结清单／条件／checkpoint／seed／采样，原生模型不改，完成产物直接提交原目录，GPU4自动跳过已完成项。GPU4继续负责解码／评分调度。`HANDOFF_GPU4_ROTATION.json`、`GPU7_ASSIST_LAUNCH.json`及任务目录的`ASSIST_GPU7.json`记录交接。未满足GPU7释放条件时，不启动辅助模型。

## 代码、停止与恢复

- [守护脚本](../../scripts/p2_gpu4_gpu7_watch9h_20261112.py)
- [GPU7辅助原生生成](../../scripts/p2_seamless_gpu7_assist_20261112.py)
- [执行检查](../../tests/test_gpu4_gpu7_watch9h_20261112.py)
- 本目录`STOP`停止巡检，不停止评测；具体任务目录的用户STOP优先，守护不删除。
- `WINDOW_FINISHED.json`表示期限结束并完成最后核查，不等同于两份全集测评结束。

若守护进程本身退出，核实没有同脚本存活进程后，在node11执行：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
 /private/liqi_workspace/project_code/Omni-LTX-stage2/.venv/bin/python -u \
 scripts/p2_gpu4_gpu7_watch9h_20261112.py
```

## 第1小时阶段衔接

GPU4已导出264条，片长校正后的最近20条中位耗时0.345秒／native frame，初始0.368；无持续降速。GPU7的6000已生成并导出全部586条，正在原生8分片评分；6500仍由原控制器接续。GPU7同时出现参考方法全集任务（feature／Avatar Forcing／DyStream／ListenFormer），现场总体利用率约97%，已核对实际PID及命令，保持其计算。GPU7尚未释放，不能加入Seamless辅助模型。见`operator/01.json`及`operator/01_phase_scoring.json`。

## 第2小时：同片长5倍降速处理

GPU4在参考方法队列新加入6个GPAvatar及2个feature进程后，同为289 native frames的生成由94–115秒增至538–553秒，且不足40GiB余量使原解码暂缓。已通过现有Codex任务协调其维护会话收敛自己队列的GPU4并发并保留原产物，同时为RNA6500预留生成余量；不泛杀其他任务，不修改其数值实现。证据及后续复测在`operator/02_contention.json`和后续修复回执。

检测器已补上严重降速的快速判定：3条相近片长的完成样本均超过初始基线2倍即可报警，每5分钟检查；中等1.5倍降速仍需两组各10条。9项执行检查通过，只重启守护使规则生效，九小时截止时间保持，GPU计算未重启。
