# GLARE 四项评测工程验收（2026-10-26）

**这是代码与检测/评分链路验收，不是 E1 效果评测。** 用户尚未提供正式 dev/test 清单；没有新启动 GPU 生成。本次使用既有 50 段 debug 数据中的前三段，做同视频自配对及漏检失败测试。

入口：[实现与运行](../../related/P2_GLARE四项反应评测实现与运行.md)、[评测 CLI](../../scripts/p2_glare_eval.py)。

## 测试

[pytest.log](pytest.log)：**86 passed, 4 subtests passed**，其中新增评测文件 34 项。命令：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  /private/liqi_workspace/project_code/Omni-LTX-stage2/.venv/bin/python -m pytest \
  tests/test_p2_glare_eval.py tests/test_p2_listen_core.py \
  tests/test_p2_listen_av_only.py tests/test_p2_listen_evaluation.py \
  tests/test_p2_listen_integration.py -q
```

覆盖四项公式约定、分数/IoU 边界、缺席类与无匹配、ATD 非对称和 ×100、FID 协方差与自身距离、逐侧未匹配反应帧、多 seed null、漏检字段、缓存身份/内容、PTS 和视频/CSV 帧数、失败不丢样本、检查点绑定、dev/test 身份隔离和 source-only 生成调度。

生成测试将原生 GPU 函数替换为 mock，只验证调度、release、输出提交及缓存复用，不能称为本次真实 GPU 生成验证。MediaPipe、FER、六类规则与 Inception 则使用真实模型在 CPU 上执行。

## 完整真实视频链路

[verified_real_positive_self/results.json](verified_real_positive_self/results.json)、[report.md](verified_real_positive_self/report.md)、[运行日志](verified_real_positive_self.log)。

样本 `naturalistic_V02_S3394_I00000303_local_f0a7c70ff4da`，51 帧，两侧均使用同一份原视频；没有人工填入反应 CSV。全部 51 帧通过 MediaPipe/FER/人脸关联检查，规则输出两种类别、三个匹配事件，两侧各 51 个反应帧。

| 指标 | 自配对结果 | 解释 |
| --- | ---: | --- |
| R-F1 | 0.3333333333333333 | 两个存在类别各得 1，另外四类缺席得 0，六类平均为 1/3 |
| R-tIoU | 1.0 | 所有匹配区间相同 |
| R-ATD | 0.0 | 起止和时长无偏差 |
| R-FID | 0.0 | 两侧真实 Inception 特征池一致 |

高维小样本协方差有数值舍入，本例原始 FID 为约 -2.53e-5，原 GLARE 实现在相对容差内截为 0，日志明确记录；没有改写评分公式。

[cache_replay.json](cache_replay.json) / [日志](cache_replay.log)：同协议重跑完整评分，结果一致，检测完成标记时间不变；测试拦截了任何意外的 `detect-one` 再启动。

## 真实漏检路径

[verified_mixed/results.json](verified_mixed/results.json) / [日志](verified_mixed.log)：原 debug 前三段全部保留，1 对有效、2 对失败。失败视频分别有 50/59 帧和 3/73 帧未满足严格联合检测条件；逐帧原因在对应 detection 的 `quality.json`。

返回状态 `incomplete`、`metrics=null`，CLI exit 2。未将缺失检测填成 neutral，也未只对通过的一段报告原三段集合的分数。前面自配对的一段单列为工程正向测试，不作为挑选正式评测集的方法。

## 训练隔离与版本

训练绑定的 70 个源码文件全部 SHA256 一致，见 [validation.json](validation.json)。新增包为 `src/p2_glare_eval`，没有修改 `src/p2_listen` / v4 复用源。所有新模型测试明确关闭 CUDA；检测依赖在独立环境，训练保持运行。

上游、六类规则参数、模型与环境锁见 [configs](../../configs/p2_glare_upstream.lock.json)，环境检查记录见 [environment_check.json](environment_check.json)。源码版本与主要验收文件哈希汇总在 [validation.json](validation.json)。同目录不带 `verified_` 的运行是开发阶段记录；最终验收使用本页链接的 `verified_` 结果。
