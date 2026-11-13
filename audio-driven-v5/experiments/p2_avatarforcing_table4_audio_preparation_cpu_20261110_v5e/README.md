# IIANet音源分离：CPU整批执行，完成后停止

**已完成并停止：43对／86路全部分离完成，独立验收通过，全部相关计算进程已退出。** [音轨清单](separated_audio_manifest.jsonl) · [完成报告](SEPARATION_COMPLETE.md) · [独立验收](SEPARATION_INDEPENDENT_AUDIT.json) · [停止记录](COMPLETED_AND_STOPPED.json)。没有启动后续AF生成或评分。

最终验收第一次把`by_split`角色别名的文件名直接当作person ID，导致身份断言失败。已核验86条别名解析后均与官方metadata一致；修正版[验收程序](finish_separation_v2.py)先解析符号链接，再逐一比对，全部通过。原程序、失败日志及[修正记录](FINAL_AUDIT_REVISION.json)保留，音轨、权重和样本没有修改。最初控制器的失败状态另存，最新完成状态指向独立验收修订2。

按用户最新要求，只完成ViCo正式test43的43对／86路音源分离与独立文件验收，不启动AF生成或评分。Table4仍未复现。

GPU7被其他任务占用后仅剩约4GiB，旧GPU批次无法按原资源余量启动后续分片。CPU同权重试验中，11秒speaker音轨的完整分离前向耗时16.88秒，重复误差0，与GPU输出最大差8.34e-7；这是单段后端对照，不能推广为全部音轨逐字节等价。

本批统一在CPU重算全部86路，不把旧GPU部分输出混入最终清单。每进程8个PyTorch线程、最多8进程；节点实测160核配额，启动检查还要求至少128GiB剩余内存。权重、官方预处理函数、音视频输入、FP32、整段前向、采样率和音频保存方式保持一致，CPU执行规则详见[设备变更](CPU_DEVICE_CHANGE.json)。先运行四个独立CPU进程，嘴部ROI须完全相同，原始音频最大差须≤1e-6；通过才自动运行全43对。

[启动回执](CPU_LAUNCH.json) · [阶段状态](RUN_STATUS.json) · [四进程一致性](PARALLEL_PARITY.json) · [CPU／GPU单段对照](CPU_GPU_CANARY_COMPARISON.json) · [全量进度](PARALLEL_STATUS.json)。各阶段文件按真实完成时间产生。

完成后保存[音轨清单](separated_audio_manifest.jsonl)、[独立验收](SEPARATION_INDEPENDENT_AUDIT.json)、[完成与停止记录](COMPLETED_AND_STOPPED.json)及[简要报告](SEPARATION_COMPLETE.md)。验收覆盖正式test43身份映射、输入和权重哈希、所有波形采样数及有限值、嘴部帧数、FLOAT WAV序列化误差和处理进程退出。

独立干净人声真值没有提供，因此文件与执行验收通过不代表已测得有监督分离质量，也不能宣称取得了AF作者的私有分离配置。

前序GPU输出与失败记录仍保留在[V5d](../p2_avatarforcing_table4_audio_preparation_cpu_frontend_20261110_v5d/README.md)，不覆盖旧成绩。CPU严格预检通过后，才退役本会话自己的旧GPU分离调度器及工作进程；不触碰其他任务。
