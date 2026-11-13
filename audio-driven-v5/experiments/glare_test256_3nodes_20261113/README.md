# GLARE step19050 → step10618：同256条、三机测评

用户于2026-11-13要求先测19050，再自动接10618。使用 Qwen/LTX 双seed实验完全相同的固定Seamless256 ID、当前GT来源、时间范围与裁剪，清单SHA `8dcc50ac17612fdac1b6030ff10b6d17ebe1411bfa723c9a6ddfb4ecc6863105`。本轮GLARE沿用原生 seed137／NFE10／speaker CFG2；不把它写成双seed结果。

- [19050实时状态](step19050/status.json) · [结果](step19050/RESULTS.md)
- [10618实时状态](step10618/status.json) · [结果](step10618/RESULTS.md)
- [三机配置](cluster.json) · [运行入口](run.py) · [CPU核对](CPU_ACCEPTANCE.json)
- [Qwen/LTX双seed均值](../p2_joint_qwen_ltx_test256_two_seeds_20261113/RESULTS.md)

**最新进度：19050的256条生成、24分片评分及全局汇总已完成；10618因原4路生成使其他卡等待，已改为三机24路，保留117条完成结果、分配139条剩余生成。** 运行证据见 `expansion24_generation/`；19050原执行入口已在该目录归档，其结果和协议保持原样。

## 输入、复用与指标

19050已核验复用248条旧生成，8条因输入来源/首帧变化重新生成。10618已核验复用89条，补生成167条。全部256条speaker音频和官方韵律缓存复用，逐条核对当前listener首帧与Qwen/LTX输入RGB相同，核对GT行SHA、裁框、时间、帧数和倾听区间。权重内部step和SHA实际核验：19050 `3df923977903bffa4710eb2b13685cf28bb2a6d22f710dfda2ed2f3625f87f55`；10618 `937a2cec588aee5e743dbd6f9d3c036acef86d6af898afa5c6ce8a381be03690`。

生成复用已有 `eval_glare_final_20261112.generate` / ResidentGLARE；仅适配设备核验与互斥任务分片。评分复用 `eval_glare_seamless_test_20261112.score` / 原paired metrics，全部256重新评分，24个互斥分片，最后全量特征重算FID/FVD，禁止平均分片FID/FVD。GPU人脸与情绪、batch64、像素4线程、GT缓存保持。音频指标N/A：GLARE不生成listener波形，speaker条件音轨不作预测评分。保留原生缺脸等N/A与有效覆盖。

## 三机与资源

node12 `172.29.252.37`、node8 `172.29.104.237`、node11 `172.29.189.119`，各GPU0–7。10618已扩为三机最多24路常驻生成，每卡一进程；各卡完成本分片生成后直接评分。每路解码batch4、PyTorch显存上限22GiB。评分每卡最多一个进程，启动时要求至少48GiB空闲，PyTorch分配上限24GiB。node11此前不可见的外部占用并未停止，本次利用其足够的剩余显存参与评分；不得声称node11已清空。不同阶段实际并发以status为准，待补生成的评分分片会接续启动。

第一次启动漏接旧生成环境的runtime_deps/glare-deps路径，导致loguru导入失败；已恢复原env_for配置并重启，首轮日志保留。没有改动原生模型或生成参数。

## 运行与恢复

控制器：`compare/.envs/avatarforcing/bin/python -u experiments/glare_test256_3nodes_20261113/run.py run`（仓库根目录执行）。先19050完成256生成、24评分分片和全局汇总，再接10618。controller.lock防重复启动；共享完成收据用于恢复，最多3次执行重试。controller.log、各step的logs/metrics/FAILURE文件记录错误。BASE下STOP表示暂停；恢复前确认是用户要求恢复，再移走STOP并执行上述入口。只停止本控制器时，它会关闭自己启动的远端进程组，不终止其他任务。

源码/输入指纹记录在各step/protocol.json；更改入口后必须先停止控制器并更新执行指纹，不改变实验参数。COMPLETE.json为完成依据，不能根据已启动PID认定测评完成。

## 实际保留的GLARE权重

正式训练目录 `../glare_paper_settings_node11_gpu7_20261111/`：before_extension20epochs.pt=10618，before_extension30epochs.pt=19050，best.pt=32000，latest.pt=32940。另有amp_initial_checkpoint.pt=10、before_reaction005.pt=180。旧配置目录 `../glare_500h_training_node11_gpu7_20261111/`：best.pt及original_step2000.pt=2000，latest.pt=3935；旧配置不与正式后续训练混称。以上步数均实际加载检查点读取。
