# Omni 尾部跨卡分担：实现与验收

规则、代码入口、配置和回退：[交接说明](../../../related/P2_Omni尾部跨卡协作调度_20261031.md)。仅修改执行调度，模型与输入语义保持原样。

- `cpu_tests.json`：63项测试、2项子测试通过。含读线程停止后GPU仍在保存的锁生命周期，以及原来源拥有者退出、上游回补时辅助包不能被新来源领取覆盖的回归。
- `simulation_32.json`：32路worker线程模拟，512条全部且仅处理一次，31路实际参与产出；仅1个来源完成回执，零重复、零遗漏。它验证调度，不是GPU吞吐测速。
- `cross_host_lock_probe.json`：四台实际主机上的排他锁与释放验证通过。
- `configuration_parity.json`：除 `tail_sharing` 和代码哈希外，原配置全部一致。
- `rollout.json`：最终版分批部署记录；`initial_rollout_before_final_guard.json` 保留首次灰度记录，未开启过跨卡辅助。
- `deployment_acceptance.json`：32个工作进程恢复产出并通过远程身份检查、每卡1条输出来源检查后才生成。
- `input_and_runtime_acceptance.json`：32条输出的修正版裁框、点头来源与时间、音频配方及运行版本核对。
- `activated.json`：完成部署验收后开启协作的时间与参数；开启前工作目录不存在 `tail_sharing/ENABLED.json`。

`before/` 保存修改前代码，`before_deploy_config.json` 保存部署前配置。生产结果和训练缓存没有复制或覆盖。发布脚本是本次有限部署流程，不是定时巡检。

复现CPU模拟：

```bash
python experiments/listen_j_omni_20261030/tail_sharing_20261031/simulate_32.py
```

部署后只读核验：

```bash
python experiments/listen_j_omni_20261030/tail_sharing_20261031/audit.py
```

完成量仍按原来源回执统计；helper领取数只代表领取，不能当作已完成数。实际辅助产出可根据新结果中的 `scheduling.role=helper` 追踪。本目录没有声称实际尾部耗时已减少多少。
