# ViCo 推理等待优化（2026-11-10）

已增加独立快速入口，适用于当前固定 ViCo43 的 ListenFormer、DualTalk 归档与续跑。保留官方模型 forward、FP32、渲染 batch=4、帧率与视频编码参数。新计算写入本目录，旧实验和报告不覆盖。

## 实测结果

| 渲染路径 | 132 帧耗时中位数 | 相对速度 |
| --- | ---: | ---: |
| 原始同步写入流程 | 8.929 s | 1.000× |
| 预计算语义窗口 + 异步写入 | 8.272 s | 1.079× |

耗时减少 7.36%。使用真实 ListenFormer 预测、官方 PIRender EMA 权重，在 node11 GPU7 上交替运行 3 轮。六次 RGB 字节 SHA256 与 MP4 SHA256 均相同，输出逐像素、逐字节一致。原始基线采用同一个 `encode(..., fast=False)`，对应旧代码的逐批语义转换与同步 pipe 写入。

这是共享 GPU、单段 132 帧、模型预热后的基准；计时包括语义预处理、推理、数据传回 CPU、编码及编码器退出，不含模型加载与事后完整解码。没有据此宣称完整 43 段同幅加速。模型加载本次为 1.002 s。完整记录见 [BENCHMARK.json](benchmark/BENCHMARK.json)。

重复启动现在先校验结果和关联清单，完成后直接返回：

| 路径 | 内部校验时间 | 包含进程启动的总时间 |
| --- | ---: | ---: |
| listenformer generate（43 段） | 0.049 s | 0.445 s |
| listenformer render（43 段） | 0.220 s | 0.894 s |
| dualtalk generate（43 段） | 2.300 s | 2.428 s |

以上三条路径均未加载模型、未请求 GPU。数据来自本机文件缓存已热的运行，包含输出 SHA256 检查和相关权重检查；冷文件缓存时间可能更长。DualTalk 需读取约 2.59 GB 权重计算 SHA256，故校验时间较长。见 [CACHE_BENCHMARK.json](CACHE_BENCHMARK.json)。

## 已实现

- 完成检查放在 torch 导入、模型加载及 GPU 检查之前，避免完成任务仍等待 GPU。
- ListenFormer 按片段复用有效预测，缺失片段才加载一次模型；恢复缺失的汇总清单无需 GPU。
- DualTalk 将有效旧预测只读链接到新目录，再调用既有官方模型适配入口计算其余片段。
- ListenFormer 渲染复用有效旧视频，只处理缺失片段；每段一次构造并传输语义窗口。
- 单独线程写入 ffmpeg，最多排队两个四帧数据块，让编码与 GPU 推理重叠；编码错误会向主线程抛出。
- 新任务使用文件锁防止同目录同阶段重复执行，预测和视频完成后原子替换。

当前只服务固定的归档数据、权重和预处理协议。若更改数据/预处理配置，应新建对应实验适配流程，不能借用此入口当作通用跨配置缓存。ListenFormer 仍是本地训练权重；加速不改变此前的复现结论。DyStream、L2L 本次未改动，Avatar Forcing 未纳入。

## 使用

在项目根目录运行。已有完整结果时，普通 Python 即可完成校验和复用：

```bash
python compare/deployment/vico_inference_fast.py listenformer generate
python compare/deployment/vico_inference_fast.py listenformer render
python compare/deployment/vico_inference_fast.py dualtalk generate
```

只检查是否可复用，可添加 `--check-only`。缺失结果需要实际计算时，使用已分配的 **node11 GPU7** 和原来的依赖环境：

```bash
CUDA_VISIBLE_DEVICES=7 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
  compare/.envs/vico_more/bin/python \
  compare/deployment/vico_inference_fast.py listenformer render
```

默认新增输出在本目录的 `listenformer/` 或 `dualtalk/`；可用 `--output-root` 指定另一新目录。保持原有 GPU UUID/可用显存检查，不停止其他训练。

基准重跑同样在 node11 GPU7 执行：

```bash
CUDA_VISIBLE_DEVICES=7 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
  compare/.envs/vico_more/bin/python compare/deployment/vico_render_fast.py \
  --output experiments/p2_vico_inference_fast_20261110/benchmark_repeat
```

## 验证与原指标

缓存测试覆盖损坏、缺失、重复或错位样本不命中，新驱动不复用旧视频，以及汇总文件丢失后复用全部 43 段预测/视频且不申请 GPU。三个测试通过；真实 GPU 渲染的六次输出完全一致。

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
  compare/.envs/vico_more/bin/python compare/deployment/test_vico_inference_fast.py
```

此前报告目录 COMPLETE.json 所列文件的 SHA256 全部复核通过，见 [ARCHIVE_CHECK.json](ARCHIVE_CHECK.json)。

指标表和复现限制沿用 [原始对比 README](../p2_vico_report_alignment_20261110/README.md) 与 [数值 CSV](../p2_vico_report_alignment_20261110/report_vs_local_all.csv)。本轮没有重跑全量指标；该短片的字节一致验证说明此次优化没有改变其评分输入。
