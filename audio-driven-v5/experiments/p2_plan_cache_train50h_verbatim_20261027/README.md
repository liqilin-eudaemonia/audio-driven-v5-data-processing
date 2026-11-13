# 新50h逐样本条件缓存与16组代码验证

**最终状态：全部完成，2026-10-27 07:19 UTC。** Gemma、输入＋计划Qwen hidden、静音后teacher各29,975条；[条件验收](../p2_plan_conditions_train50h_verbatim_20261027/COMPLETE.json)、[teacher验收](../p2_matrix16_teacher50h_20261027/COMPLETE.json)、[最终汇总](progress_snapshot.json)。14份条件校验报告、48份teacher报告全部成功。teacher已核对分片完整覆盖、全部完成标记合同与payload存在性；提取/复用时已记录或核对payload SHA，本次未重复读取全部大文件。最后两个teacher分片在准备迁移前自行完成，未实际迁移。所有本轮缓存任务正常结束，未启动正式训练。


2026-10-27。仅新29,975段；保留原始标注，包括单字中文。没有启动16组正式训练，没有处理722,706段全量新文本。

- [16组训练入口与说明](../../related/P2_十六组训练实现与运行.md)
- [正式条件缓存](../p2_plan_conditions_train50h_verbatim_20261027/)，Gemma及Qwen input_plan
- [三机共享配置](../../configs/p2_plan_cache_3nodes_shared50h.json)，共14卡
- [当前作业身份](full50h_job/job.json)、[各卡实时状态](full50h_job/workers/)、[日志](full50h_job/logs/)
- [静音后Feature teacher作业](../p2_matrix16_teacher50h_20261027/README.md)，已扩为三机24卡、每卡2进程（48分片），与Qwen条件缓存叠加执行，生成同一29,975段的teacher

## 查看与续跑

```bash
P2_PYTHON=/private/liqi_workspace/project_code/Omni-LTX-stage2/.venv/bin/python
"$P2_PYTHON" scripts/p2_plan_cache_cluster.py status \
  --job experiments/p2_plan_cache_train50h_verbatim_20261027/full50h_job
```

每分片依次 text → input_plan → verify。全部worker和节点成功，才算完整完成；text全部完成不代表Qwen也完成。独立监督进程不依赖当前SSH会话。失败有限重试一次，持续失败会明确标记，不能自动算作跳过成功。详情见[集群说明](../../related/P2_显式规划缓存三机联合运行.md)。

首次启动时各卡至少80GiB空闲，缓存进程限额64GiB；允许与已有任务叠加。本次node17卡1/2/3/4，node12卡1/2/3/4，node4卡0/1/2/3/6/7。全局14分片覆盖29,975段，无重复成员。

## 验证证据

- [完整文本审计](prompt_audit.json)：29,975条；2,761条含汉字；6条超过4,096 token，最长4,981，新增8,192档完整编码。
- [逐条长度与prompt摘要](prompt_lengths.json)。不改时间、不翻译原词、不截断。
- [真实GPU小批](gpu_smoke/)：9条含最长6条、单字中文及短文本，3个rank两类缓存和verify全部成功；[作业记录](gpu_smoke_64g_retry_job/job.json)。Gemma峰值分配约51.7GiB，保留约56.6GiB；Qwen分配约10GiB。
- 历史`gpu_smoke_job`的48GiB限制触发OOM，后来的64GiB重跑已通过。`errors/`是历史诊断，当前结果以完成标记和最新成功report为准。没有通过截断绕过长文本。
- [88项相关CPU测试](matrix16_pytest.xml)，覆盖16组真实小模型梯度、恢复、缓存及监控等。
- [真实Qwen在线梯度](qwen_gradient_gpu.json)：140处LoRA，AV替代目标／Event／Feature均可回传，冻结主干不变；这个报告明确`real_ltx=false`。
- [新50h事件预检查](../p2_matrix16_training_20261027/fixed_av_s1/preflight.json)：29,975段全部完成，七类均有可用正负支持。只读事件与身份，不重新加载全部AV/hidden或重算源视频hash。

[真实联合GPU验证](joint_ltx_qwen_gpu.json)已通过：预训练LTX＋Qwen、真实后处理teacher，5.767秒／512×512样本的三路loss梯度和一次优化更新，峰值保留54.3GiB。为了验证上游梯度，使用非零初始化的Connector输出层/decoder零层模拟父阶段状态，不是真实s1训练权重。工程验证不构成生成质量结论，也不等于16组训练已完成。
