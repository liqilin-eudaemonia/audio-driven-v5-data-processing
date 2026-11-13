# ViCo 推理基础设施优化验收（2026-11-10）

本轮补齐 DualTalk 的实际 forward 优化，以及 L2L → GPAvatar 的渲染、缩放、编码流水线。已完成真实 GPU 对照、命令入口冒烟、断点恢复和错误清理测试，可以验收本文覆盖的推理基础设施改动。Avatar Forcing 不在本次范围。

## 数值对比

均为各自固定工作量下的实测中位数；“耗时减少” = 1 − 优化后/优化前。不同方法工作量不同，不能据此横向比较模型速度，也不能将收益相加。

| 方法/阶段 | 测试范围 | 优化前(s) | 优化后(s) | 耗时减少 | 一致性 |
|---|---|---:|---:|---:|---|
| **DualTalk，本轮** | 全部43条完整输入，模型 forward | 3.009 | 2.390 | **20.56%** | 全部原生预测逐元素相同 |
| **L2L → GPAvatar，本轮** | 60帧真实渲染 + 三种视频导出 | 11.579 | 10.274 | **11.26%** | 512原图、512复原图、256评分视频的MP4 SHA全部相同 |
| DyStream，前轮 | 2条×2秒，生成+渲染+导出 | 11.445 | 9.509 | 16.91% | motion和评分视频相同 |
| L2L Conan，前轮 | 43条×128帧，motion推理 | 0.388 | 0.376 | 3.09% | token和motion相同 |
| L2L Fallon，前轮 | 同上 | 0.444 | 0.425 | 4.36% | 同上 |
| L2L Stephen，前轮 | 同上 | 0.396 | 0.387 | 2.49% | 同上 |
| L2L Trevor，前轮 | 同上 | 0.394 | 0.382 | 3.29% | 同上 |
| ListenFormer → PIRender，前轮 | 132帧，渲染+编码 | 8.929 | 8.272 | 7.36% | RGB和MP4相同 |

完整精度见 [PERFORMANCE.csv](PERFORMANCE.csv)。前轮原始证据位于 [原优化目录](../p2_vico_inference_fast_20261110/README.md)，本轮没有重复测量这些方法。

DualTalk 每条输入都实际运行模型，三个优化轮次均从**空静音特征缓存**开始，每轮21个长度首次计算、22次复用。复用的是官方 Wav2Vec2 对相同长度全零自音频的投影特征；对方音频编码、动作编码及后续预测均重新计算。不是复用历史预测文件。时间不含模型加载和公共输入预处理；本次公共预处理为1.032秒，不能宣称整个任务缩短20.56%。

GPAvatar 每轮均重新渲染60帧、启动三个ffmpeg并等编码完成，计时包含编码启动和收尾，不含模型/身份初始化。采用ABBAAB交替顺序，各路径三轮。显卡上保留了其他现有任务，因此数值是该共享环境的实测，不是独占硬件上限。证据见 [DualTalk](DUALTALK_BENCHMARK.json)、[GPAvatar](gp_benchmark/BENCHMARK.json)。

## 实际改动

- DualTalk：静音分支按长度缓存，最多64项；非静音、训练、梯度开启、CUDA autocast或不支持的批形状走原始路径。检测参数替换/原地更新后清空特征缓存。继续使用官方FP32计算、权重和注意力实现。
- GPAvatar：保持官方逐帧模型计算，将CPU评分缩放和三个ffmpeg的写入放到后台，与下一帧渲染重叠。最多两个待导出的帧对，避免整段视频同时堆积在Python列表。固定仿射网格只构造一次，缩放、量化、编码参数保持一致。
- GPU申请前检查视频/预测、输入动作、协议和权重来源。GPAvatar新入口可以复用原有172条视频，也可以只补缺失片段；DualTalk修复损坏的局部结果时只删除新目录的链接和文件，不写入链接指向的归档。
- 编码失败向上报错并清理临时文件；全部编码成功后发布视频，最后写完成凭据。原始官方实现和已封存的指标文件保持只读。

尝试后未启用的方案：DualTalk关闭注意力返回值会引入最高约2.19e-5差异，且收益小；GPAvatar仅对超分模块使用CUDA Graph，8帧中位耗时约0.793→0.781秒，另有0.215秒建立开销。本轮生产入口均未启用这两项实验选项。

## 验收结果

1. DualTalk全部43条完整原生预测、三个优化轮次：最大绝对差0。
2. GPAvatar两条身份输入（最短/最长）×四套L2L动作×首/中/尾帧：固定网格优化的浮点输出差0。
3. 新GPAvatar命令入口强制重新生成8条冒烟结果、共24个MP4，与归档中的官方路径冒烟视频逐字节相同，见 [CLI_SMOKE_PARITY.json](CLI_SMOKE_PARITY.json)。
4. 再次运行冒烟入口，以及检查原有172条完整视频，均直接复用且未导入torch、未申请GPU，见 [GP_RESUME.json](GP_RESUME.json)。
5. 12项回归检查通过：参数更新/替换失效、缓存容量和回退、视频损坏、编码失败清理、归档链接保护、已有各方法断点恢复，见 [REGRESSION.log](REGRESSION.log)。
6. 之前指标报告封存的87个文件SHA未改变，见 [ARCHIVE_CHECK.json](ARCHIVE_CHECK.json)。

## 使用

在项目根目录执行。DualTalk和GPAvatar需要在**node11物理GPU7**上运行；入口继续执行GPU UUID及至少32GiB空闲显存检查，不停止其他任务。以下不是跨节点自动调度命令。

```bash
# DualTalk：已有完整结果直接复用；有缺失时自动启用静音分支优化
CUDA_VISIBLE_DEVICES=7 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
  compare/.envs/vico_more/bin/python compare/deployment/vico_inference_fast.py dualtalk generate

# GPAvatar：核对已有172条完整结果；缺失时运行新流水线
CUDA_VISIBLE_DEVICES=7 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
  compare/.envs/vico_more/bin/python compare/deployment/vico_gpavatar_render_fast.py

# 只检查，不触发推理；原有 vico_gpavatar_eval_fast.py render 也已接入新流水线
compare/.envs/vico_more/bin/python compare/deployment/vico_gpavatar_render_fast.py --check-only

# 强制在独立目录生成8条冒烟结果；仍支持恢复该目录已有的正确结果
CUDA_VISIBLE_DEVICES=7 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
  compare/.envs/vico_more/bin/python compare/deployment/vico_gpavatar_render_fast.py \
  --smoke --no-archive-reuse --output-root experiments/p2_vico_infra_acceptance_20261110/gp_cli_smoke
```

GPAvatar新结果默认保存到本目录的 `gpavatar/`；可用 `--output-root` 指定新目录。CUDA扩展复用原工程的扩展目录，入口补全CUDA工具链PATH及protobuf兼容环境。DyStream/L2L仍沿用前轮的节点分配，不能把它们直接放到node11执行。

## 验收边界

本轮验收的是已具备可运行代码和所需权重的上述路径，不代表把论文指标复现问题解决了。ListenFormer仍使用此前记录的本地重训权重；缺少完整权重、代码或评估资产的方法仍未被标为完成，见 [待测清单](../p2_vico_additional_optimization_20261110/PENDING_METHODS.csv)。LM Listener索引优化仅属CPU评估准备，不混入模型推理收益。

本轮没有重跑GPAvatar全量172条渲染或整套EMOCA/图像指标，也没有测出包含模型冷加载及全量评估的端到端提速。GPAvatar核心网络数值计算保持原样，收益来自整个渲染导出流程。

本目录的 `DELIVERY.json` 记录本次最终代码与证据的SHA；早期目录中的DELIVERY是各自交付时的快照，本次修改过的入口以本目录版本为准。
