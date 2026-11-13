# ViCo 所有待测方法：推理优化与实测（2026-11-10）

最新补充：已补齐飞书遗漏候选、合并别名，并增加LM Listener评分和GPAvatar恢复优化，见[后续核查与实测](../p2_vico_additional_optimization_20261110/README.md)。下文为前一轮四种生成器测速。

当前能运行的四种方法均已接入快速入口：DyStream、Learning2Listen 四套人物权重、DualTalk、ListenFormer。Avatar Forcing 按用户要求排除。其余方法保留明确的资源缺项状态，不重复启动 GPU 或等待不存在的权重。

## 覆盖与实现

| 方法 | 已实现 | 恢复粒度 |
| --- | --- | --- |
| DyStream | 完成校验先于模型加载；按剩余任务/显存启动最多4个 worker；单 worker 内原生编码、评分视频导出与下一段官方推理重叠 | 单片段 |
| L2L Conan/Fallon/Stephen/Trevor | speaker/audio/history 留在 GPU；去掉每个自回归步的 token GPU→CPU→GPU；去掉未使用概率数组回传；四个模型共享一次 speaker 平滑；减少进度文件写入 | 整个人物模型的43段批次 |
| DualTalk | 完成结果校验直接返回；有效预测复用，只计算缺失项 | 单片段；未修改内部 forward |
| ListenFormer | 完成复用与断点恢复；PIRender 语义窗口预计算、异步编码 | 单片段 |

L2L 的原始 `create_data_vq` 仅在运行时替换三处输入搬运语句，其余函数体沿用官方代码。保留 decode→requantize、nucleus sampling、FP32、batch=43 和样本顺序。**不能把某个人物模型的一批43段拆成“只重跑一个丢失样本”**，否则随机采样序列会改变；缺损时重算该人物整批，跳过其他完整人物模型。

DyStream 保留官方5步、CFG、逐片段 seed137、原始渲染、512/25→256/30时空转换、两套编码参数。异步导出每个 worker 最多挂起一段任务，错误向主线程传播。批量最多4个 worker 与旧入口一致；下表只测了单 worker 内的流水线收益，不声称已测完整4-worker吞吐提升。

## 实测速度与一致性

| 流程 | 原耗时中位数（秒） | 优化后（秒） | 耗时减少 |
| --- | ---: | ---: | ---: |
| DyStream | 11.445 | 9.509 | 16.91% |
| L2L/conan | 0.388 | 0.376 | 3.09% |
| L2L/fallon | 0.444 | 0.425 | 4.36% |
| L2L/stephen | 0.396 | 0.387 | 2.49% |
| L2L/trevor | 0.394 | 0.382 | 3.29% |
| ListenFormer/PIRender | 8.929 | 8.272 | 7.36% |

- DyStream：两段真实 ViCo 输入各截取2秒，AB/BA两轮。每段49个原生帧、60个评分帧；包含动作生成、渲染、原生视频及评分视频导出，计时从模型加载后开始。四次运行的动作数组和评分视频文件 SHA256 一致。[原始记录](benchmark_dystream/BENCHMARK.json)
- L2L：每个官方人物模型固定43段批次、生成前128帧，交替3轮。所有 token 与解码动作数组逐元素完全相同；计时包括推理中的输入搬运、采样和解码，排除模型加载、特征准备及保存输出。[四套原始记录](benchmark_l2l/BENCHMARK.json)
- ListenFormer：沿用上一轮132帧、3轮交替测速。RGB与MP4逐字节一致。[原始记录](benchmark/BENCHMARK.json)
- DualTalk：本轮没有改变模型 forward，不给出内部推理加速百分比。

以上是限定输入、共享 GPU 上的实测，不外推完整43段速度。数值表：[PERFORMANCE.csv](PERFORMANCE.csv)。

## 重复运行无需 GPU

统一校验结果文件、关联清单和权重后返回。最新全方法检查见 [INVENTORY.json](INVENTORY.json)：ListenFormer43段、DualTalk43段、DyStream43段、L2L172段均已成功命中。DyStream需要读取多份大权重及视频，校验约11秒；L2L约1.2秒，DualTalk约3.1秒，ListenFormer预测约0.09秒。它们是热文件缓存下的校验时间，不是模型推理时间。

普通 Python 可以检查整个清单：

```bash
python compare/deployment/vico_inference_fast.py all generate --check-only
python compare/deployment/vico_inference_fast.py dystream generate --check-only
python compare/deployment/vico_inference_fast.py learning2listen generate --check-only
```

已有完整结果时，去掉 `--check-only` 同样会直接复用。实际缺失需要计算时，应使用对应节点与环境；`all` 不负责跨节点调度：

```bash
# node8 GPU7：DyStream
CUDA_VISIBLE_DEVICES=7 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
WANDB_MODE=disabled TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
compare/.envs/dystream/bin/python compare/deployment/vico_inference_fast.py dystream generate

# node8 GPU7：L2L（四套人物权重）
CUDA_VISIBLE_DEVICES=7 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
compare/.envs/l2l/bin/python compare/deployment/vico_inference_fast.py learning2listen generate

# node11 GPU7：ListenFormer / DualTalk
CUDA_VISIBLE_DEVICES=7 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
compare/.envs/vico_more/bin/python compare/deployment/vico_inference_fast.py listenformer generate
CUDA_VISIBLE_DEVICES=7 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
compare/.envs/vico_more/bin/python compare/deployment/vico_inference_fast.py listenformer render
CUDA_VISIBLE_DEVICES=7 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
compare/.envs/vico_more/bin/python compare/deployment/vico_inference_fast.py dualtalk generate
```

默认新输出在本目录下按方法分目录，也可用 `--output-root` 指定新目录。完整旧产物按 SHA256 校验后只读复用；重算前会解除指向旧结果的链接，避免写入归档。每个方法/阶段有运行锁，保持 GPU UUID和显存检查。

缓存仅针对当前固定数据、权重、代码与预处理协议。换数据或协议要建立对应新实验，不能把此入口当作任意配置的通用缓存。本轮L2L快速生成入口处理原生DECA动作；独立L2L＋GPAvatar支线已有172条写实视频，评分待完成，见[最新补查](../p2_vico_additional_optimization_20261110/README.md)；ListenFormer仍使用此前本地完整重训权重。

## 当前缺资源的方法

下表来自已有本地审计，本次没有重新检索在线发布。统一入口会立即返回 `awaiting_assets`，不将其记为“已经完成推理优化/测速”。

| 方法 | 本地缺项 |
| --- | --- |
| dim | No usable core generator checkpoint in the audited local assets |
| rlhg | No completed local training checkpoint; only training benchmarks exist |
| vico | Core baseline checkpoint unavailable |
| mm2022 | Core generator checkpoint unavailable |
| reactface | Core checkpoint and renderer resources unavailable |
| ditailistener | Runnable official generator assets unavailable |
| difflistener | CVPR2025 official runnable generator unavailable |
| infp | Official runnable generator unavailable |
| omniresponse | Complete generator code/weights unavailable |
| speakervid | Repository has curation/metrics, no runnable generator |

## 检查和指标报告

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
compare/.envs/vico_more/bin/python -m unittest discover \
  -s compare/deployment -p 'test_vico*fast*.py'
```

缓存测试覆盖损坏/缺失文件、模型批次损坏、输入驱动变化、缺失汇总清单的免GPU恢复、失效归档链接解除。真实 GPU 一致性验证见上述两套新增 benchmark。原始官方源码、权重、既有指标结果保持原样。

原数值对比：[report README](../p2_vico_report_alignment_20261110/README.md)、[report与本地指标CSV](../p2_vico_report_alignment_20261110/report_vs_local_all.csv)。本次测试用于验证执行优化，不改变此前关于论文数据集、权重、表示空间差异的结论，也未重新跑完整43段指标。
