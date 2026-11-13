# node11全部本项目测评暂停记录

北京时间2026-11-13 14:57:18：按用户明确要求暂停node11上的Qwen/LTX权重比较及GPU7参考方法测试。GPU7共11个worker、Monitor、controller、ResponseNet followup均退出，退出后GPU7显存7MiB、利用率0%。原Qwen/LTX本轮任务此前已结束，未向node11启动新的联合测评。其他任务后续可能占卡，不把整机持续空闲作为承诺。

- [Qwen/LTX进程核查](NODE11_RELEASED.json)
- [GPU7退出及进度快照](NODE11_GPU7_PAUSED.json)
- 暂停前配置与调度快照：`../p2_reference_methods_seamless6624_20261112/pause_20261113_user_node11_release/`
- 已完成媒体、动作、特征、评分及原25条批次均保留；未提交完成收据的在途样本恢复后重算。
- GPU7共享RESOURCE_POLICY的allowed_gpus置空，FOLLOWUPS.entries置空，两个任务均写PAUSED_BY_USER.json。暂停属于用户操作，已写planned_defer，未算技术失败。
- 两个参考任务尚未完成全量，不能把暂停状态称为完成。它们不会自动转移到node8/node12；这两机16卡授权用于Qwen/LTX联合筛选。

暂停前最后一次调度快照（可能比停止时实际完成数少几条，恢复以原子收据为准）：

```json
{
  "features": 988,
  "evaluation_gt": 1311,
  "gpavatar_references": 863,
  "motion": {
    "listenformer": 987,
    "dualtalk": 988,
    "l2l_conan": 975,
    "l2l_fallon": 975,
    "l2l_stephen": 975,
    "l2l_trevor": 975
  },
  "video": {
    "dystream": 631,
    "avatarforcing": 1283,
    "listenformer": 985,
    "l2l_conan": 386,
    "l2l_fallon": 386,
    "l2l_stephen": 386,
    "l2l_trevor": 386
  },
  "scored": {
    "dystream": 300,
    "avatarforcing": 1200,
    "listenformer": 200,
    "l2l_conan": 200,
    "l2l_fallon": 200,
    "l2l_stephen": 200,
    "l2l_trevor": 200
  }
}
```

## 恢复条件与入口

必须收到用户重新分配资源的指令后才能恢复。若仍授权node11 GPU7：从快照恢复RESOURCE_POLICY（只允许7，禁止4），恢复FOLLOWUPS任务清单并把其gpu_ids改为[7]，移除两目录PAUSED_BY_USER.json并更新暂停状态。CONTROLLER_STATE已经清空活跃PID而保留历史失败计数，planned_defer回执支持下一attempt恢复。

随后在node11使用原vico_more环境启动Seamless6624/monitor.py，并启动同目录followup_watch.py恢复ResponseNet586接续；每项只运行一个控制器，原生锁防重。不要再次执行prepare_full或删除已完成目录。若改到其他机器，先适配并验收硬编码设备UUID和调度资源再启动。

```bash
cd /private/liqi_workspace/project_code/audio-driven-v5
PYTHONUTF8=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 compare/.envs/vico_more/bin/python experiments/p2_reference_methods_seamless6624_20261112/monitor.py
# 另一个守护进程（恢复授权与配置后）
PYTHONUTF8=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 compare/.envs/vico_more/bin/python experiments/p2_reference_methods_seamless6624_20261112/followup_watch.py
```

## 下一轮联合测评资源

仅node8(172.29.104.237) GPU0–7和node12(172.29.252.37) GPU0–7，共16卡。node11整机禁止用于该轮任务，node4也不在本轮范围。原计划16组×64＝1024条，复用320条，新增704条；本记录生成时尚未启动。
