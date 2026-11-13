# 其他待测方法、清单纠正与优化（2026-11-10）

**还有其他候选。上一轮“所有当前可运行方法”的范围仅覆盖已接通的四种生成器，没有完整覆盖飞书方法清单。** 本轮补齐遗漏候选、合并重复命名，增加 LM Listener 评分优化和 L2L＋GPAvatar 免重复渲染入口，修复后者的评分启动依赖。

## 候选与状态

| 方法／任务 | 本轮核对结论 | 当前可做的优化 |
| --- | --- | --- |
| LM Listener | 官方仓库有Trevor预训练模型下载链接；本地生成权重尚未接入。现有评分器可用；“我们的音频RGB模型评测LM Listener数据集”另缺经过验证的原始媒体映射 | 已实现并实测按需帧索引，官方评分结果一致 |
| GLARE | 已有独立500h数据缓存和训练接入任务，正式训练未完成；4步短训权重不作为性能基线 | 原任务已有缓存提速，本轮未修改/重复启动 |
| ELP、MFR-Net | 本轮查看论文及定向检索，未取得本地可运行官方生成器 | 记录为待接入，不能报告推理加速 |
| CustomListener | 官方项目页的Code按钮指向论文，找到的是展示页资源 | 记录为待接入 |
| ARIG | 找到作者项目页仓库，内容为页面与演示资源 | 记录为待接入 |
| ECHO、Latent Behavior Diffusion | 已定位对应论文，本轮未找到并取得可运行官方代码/权重 | 记录为待接入 |
| ViCo baseline / RLHG | 同一个方法；本地已有原生短训诊断，尚无完成正式训练的生成权重 | 合并别名，避免重复排队 |
| PCH / MM2022 | 同一个方法；已有部分代码接入，缺完整生成模型资源 | 合并别名，避免重复排队 |
| DIM、ReactFace、DiTaiListener、DiffListener、INFP、OmniResponse、SpeakerVid | 延续已有本地资源缺项；本轮未重做这些方法的完整在线审计 | 保留明确缺项，立即返回状态，不空等GPU |
| L2L＋GPAvatar | 这是L2L的额外渲染/评分支线，并非另一种动作生成方法；已有172条视频，评分尚未完成 | 已加入校验后直接复用，并修复protobuf启动问题 |

另有两点范围区别：飞书的“L2L-trevor”文字与所链接仓库标题不完全一致，本轮LM Listener严格指 `sanjayss34/lm-listener` 的《Can Language Models Learn to Listen?》，不与L2L公开Trevor人物权重混用。Latent Behavior Diffusion论文实验在REACT2024上，不能把其结果直接作为ViCo report；见[论文实验设置](https://arxiv.org/html/2505.07901v1#S4.SS1)。

“未取得可运行资源”只描述本次核查和本地状态，不等于断言作者从未发布代码。Avatar Forcing按用户要求继续排除。我们的DIM best10队列仍保留原暂停状态，没有将它误算为一个新的DIM生成器。

公开依据：[LM Listener官方README](https://github.com/sanjayss34/lm-listener#pre-trained-model)、[CustomListener项目页](https://customlistener.github.io/)、[ARIG项目仓库](https://github.com/jinyugy21/ARIG)、[ELP论文](https://arxiv.org/abs/2310.00068)、[MFR-Net论文](https://arxiv.org/abs/2308.16635)、[ECHO论文](https://arxiv.org/abs/2603.17427)、[Latent Behavior Diffusion论文](https://arxiv.org/abs/2505.07901)。本轮页面下载记录在[sources/SOURCES.json](sources/SOURCES.json)，方法清单见[PENDING_METHODS.csv](PENDING_METHODS.csv)。

## LM Listener：已执行的优化

原评分器在每次运行时，逐帧构造listener/speaker两张大型字典，并且对每个预测重复扫描全部frame key（结果未使用）。新入口保留官方评分主体，按视频建立半开区间索引，按需读取系数，最多缓存8个segment系数块；去掉未使用的全量扫描。

重复／重叠区间仍按官方“后出现的记录覆盖前记录”；没有静默去重、修改计分范围或替换指标。官方源码只读，新入口绑定源码SHA，版本改变会要求复核。

| 已测项目 | 原实现 | 优化实现 |
| --- | ---: | ---: |
| 128条真实记录、两侧索引构建 | 0.981066 s | 0.003441 s |
| 索引构建Python峰值内存 | 23.446 MiB | 0.039 MiB |

同一组真实数据逐项比较 **4,196,400个标量完全相同**，另验证重叠、重复和缺口。单次微基准只测索引构建，内存数字不含已加载输入tensor、原生库内存或后续LRU系数块；不能将索引速度比当作模型推理或整套评测加速倍数。

进一步在原60帧GT自对照输入上完整执行两版官方评分，固定NumPy seed137后，MSE、FD、paired FD、valence、diversity、Var所有输出逐项完全一致。此处是技术自对照，不是LM Listener或我们模型的生成成绩。

- [索引测速与一致性](LM_LISTENER_INDEX_BENCHMARK.json)
- [完整官方评分一致性](LM_LISTENER_SCORER_PARITY.json)
- 新入口：`compare/deployment/lm_listener_eval_fast.py`

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
compare/.envs/vico_more/bin/python compare/deployment/lm_listener_eval_fast.py \
  --output_dir /absolute/path/to/predictions \
  --segments_path evaluation/lm-listener/dataset/trevor/segments_test_no_overlap.pth \
  --mean_std_path evaluation/lm-listener/dataset/trevor
```

输入／输出格式和官方参数一致，分数JSON仍写到指定预测目录。已有预测不需要重新生成。

## L2L＋GPAvatar：纠正旧状态、减少重复等待

当前[RENDER_COMPLETE.json](../p2_l2l_public_renderer_20261110/RENDER_COMPLETE.json)确有172条完整视频；原控制器停在EMOCA启动时的WandB/protobuf导入错误。之前“没有L2L写实renderer”的说法仅适用于作者原始管线，现在本地已有另行接入的GPAvatar公开renderer，必须明确区分。

新入口在加载模型、申请GPU之前核对172条视频、原动作与协议哈希，实测 **1.081秒** 返回已完成，无需重做渲染。恢复入口设置 `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`，不升级或修改共享环境包。

修复后已在CPU真实反序列化同一官方EMOCA checkpoint，两个encoder各确认322个state项，CUDA未初始化。[依赖验证](GPAVATAR_DEPENDENCY_PROBE.json)。GPU smoke实际尝试因剩余显存26639MiB低于原32768MiB门槛而停止；没有降低门槛或停止其他任务，因此尚不能宣称GPU评分已恢复完成。

```bash
# 普通Python：确认全部视频可复用
python compare/deployment/vico_gpavatar_eval_fast.py status

# CPU：验证实际权重反序列化和protobuf修复
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
compare/.envs/vico_more/bin/python compare/deployment/vico_gpavatar_eval_fast.py probe

# node11 GPU7容量满足时：原EMOCA入口、原配置，带依赖修复
CUDA_VISIBLE_DEVICES=7 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
compare/.envs/vico_more/bin/python compare/deployment/vico_gpavatar_eval_fast.py smoke
CUDA_VISIBLE_DEVICES=7 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
compare/.envs/vico_more/bin/python compare/deployment/vico_gpavatar_eval_fast.py extract
```

## 统一待测入口

扩展了上轮快速入口的清单，加入遗漏方法以及RLHG→ViCo、PCH→MM2022、LM Listener的别名。缺项方法立即返回原因。L2L＋GPAvatar和LM Listener评分使用上面的专用入口，不将其假装成已经接通新的原生生成模型。

```bash
python compare/deployment/vico_inference_fast.py all generate --check-only
```

原有四方法加速记录见[上一轮README](../p2_vico_inference_fast_20261110/README.md)。本轮未改任何原论文数字、旧评测分数、官方模型forward或正在运行的GLARE训练/缓存任务。
