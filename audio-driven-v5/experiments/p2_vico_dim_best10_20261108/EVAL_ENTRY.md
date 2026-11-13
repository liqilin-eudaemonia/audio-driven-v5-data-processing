# ViCo / DIM eval config 与评测入口

更新日期：2026-11-09。

**当前状态：按 DIM 公开实现重建的评测配置已固定，单次评测已完成；十次采样补测的入口已准备、CPU 校验通过，当前队列暂停，尚无十次补测结果。不能将这个状态描述为“还没有进行任何评测”，也不能称为“已经严格复现论文原表”。**

本文是本机工作区的入口说明，不包含模型或数据的重新打包。所有命令和路径均指向现有项目。

## 1. 关键绝对路径

| 内容 | 绝对路径 |
|---|---|
| 项目根目录 | `/private/liqi_workspace/project_code/audio-driven-v5` |
| 十次补测工作目录 | `/private/liqi_workspace/project_code/audio-driven-v5/experiments/p2_vico_dim_best10_20261108` |
| 冻结的十次补测协议 | [protocol.json](/private/liqi_workspace/project_code/audio-driven-v5/experiments/p2_vico_dim_best10_20261108/protocol.json) |
| 执行入口 | [p2_vico_dim_best10.py](/private/liqi_workspace/project_code/audio-driven-v5/scripts/p2_vico_dim_best10.py) |
| 队列状态 | [status.json](/private/liqi_workspace/project_code/audio-driven-v5/experiments/p2_vico_dim_best10_20261108/status.json) |
| DIM 指标及范围配置 | [p2_vico_dim_inferred_v1.json](/private/liqi_workspace/project_code/audio-driven-v5/configs/p2_vico_dim_inferred_v1.json) |
| DIM 原版指标的单次结果 | [RESULTS.md](/private/liqi_workspace/project_code/audio-driven-v5/experiments/p2_vico_literature_protocol_20261108/RESULTS.md) |
| 用户指定 Var 的单次结果 | [RESULTS.md](/private/liqi_workspace/project_code/audio-driven-v5/experiments/p2_vico_dim_var_per_coefficient_20261108_v2/RESULTS.md) |
| 用户指定 Var 配置 | [p2_vico_dim_variance_per_coefficient_v1.json](/private/liqi_workspace/project_code/audio-driven-v5/configs/p2_vico_dim_variance_per_coefficient_v1.json) |
| 公开证据确认报告 | [README.md](/private/liqi_workspace/project_code/audio-driven-v5/experiments/p2_vico_dim_public_confirmation_20261109/README.md) |

`protocol.json` 是十次补测实际读取的固定协议。`p2_vico_dim_inferred_v1.json` 记录指标、时间索引及范围敏感性设置，不是启动生成的入口；十次补测不会自动跑完其中全部敏感性实验。

## 2. 实际评测配置

| 项目 | 设置 |
|---|---|
| 模型 | 已封存的联合 8000 步＋SFT2752，不是新训练的七个基线短跑检查点 |
| 测试范围 | ViCo 官方 test43，按独立 listener UUID 保留，不混 OOD |
| 已有候选 | seed137 的 43 条输出及动作特征 |
| 待补候选 | seed138–146，各 43 条，共 387 条；最终 430 个候选 |
| 生成条件 | 复用原封存条件；只改变生成随机种子，不重新采样规划文字 |
| 生成参数 | Euler 30 步，video CFG 3、audio CFG 7 |
| 动作提取 | `EMOCA_v2_lr_mse_20`，原始 pose6＋exp50，不做归一化 |
| 视频流程 | 原生 512×512／25fps，沿已有流程转为评分 256×256／30fps |
| 时间索引 | listener t1…T−1，speaker t0…T−2 |
| 候选选择 | 每段按完整 56 维 GT Fréchet 距离选一个候选；所有指标使用同一候选 |
| 有效帧 | GT、speaker 和全部十个候选的共同有效帧；单次与十次结果使用相同覆盖 |
| FD／P-FD／MSE | DIM 发布实现，逐片段计算、等权平均 |
| SID | GT 拟合 KMeans，表情 40 类／姿态 20 类；全局直方图、log2 熵 |
| 自动产出的 Var | DIM 发布代码的全帧全系数拉平方差 |
| speaker | 使用真实动作供评分；不照搬默认加载器的全一常量 |

十次择优使用真值，是评测中的 oracle 选择，不代表部署时可以使用的选择方式。公开入口的规则已确认；作者 Table 1 实际采用哪条运行分支及哪些有效帧，仍缺直接执行材料。

## 3. 查看已有结果和当前状态

以下命令只读取记录，不启动 GPU：

```bash
cd /private/liqi_workspace/project_code/audio-driven-v5

python -m json.tool /private/liqi_workspace/project_code/audio-driven-v5/experiments/p2_vico_dim_best10_20261108/status.json

cat /private/liqi_workspace/project_code/audio-driven-v5/experiments/p2_vico_literature_protocol_20261108/RESULTS.md
```

本次核对时，状态为 `paused_by_user_for_31gpu_training`。seed138–146 均无完成标记；十次工作目录尚无 `RESULTS.json`、`RESULTS.md`、`COMPLETE.json`。这表示十次补测未完成，不影响已有单次结果。

## 4. 恢复前的只读校验

本命令校验冻结源码及来源回执，不加载生成模型、不恢复队列。2026-11-09 已执行通过。

```bash
cd /private/liqi_workspace/project_code/audio-driven-v5

env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
  /private/liqi_workspace/project_code/audio-driven-v5/compare/.envs/vico_more/bin/python - <<'PY'
import sys
sys.path.insert(0, '/private/liqi_workspace/project_code/audio-driven-v5/scripts')
from p2_vico_dim_best10 import verify
p = verify()
print('配置校验通过')
print('模型:', p['model'])
print('测试条数:', len(p['ids']))
print('种子:', p['seeds'])
print('待补生成条数:', p['additional_outputs'])
PY
```

预期为模型 `combined8000_sft2752`、43 条、seed137–146、待补 387 条。源码／来源校验失败时，应检查错误对应的文件，不直接删除指纹或重做 `prepare` 绕过它。

## 5. 恢复十次补测的入口

**必须在 node11 执行。** Worker 会核对物理 GPU7 的 UUID：`GPU-4d5ee808-e829-c824-c506-cad80ff133f5`。在其他节点运行不会自动迁移到 node11。

可先检查是否已有该队列进程及 GPU7 的实际状态：

```bash
pgrep -af '[p]2_vico_dim_best10.py'
nvidia-smi -i 7 --query-gpu=uuid,memory.free --format=csv,noheader,nounits
```

恢复命令如下。**执行它会恢复暂停的队列，并在显存达到门槛后启动 GPU 工作；本次交付文档时没有执行此命令。**

```bash
cd /private/liqi_workspace/project_code/audio-driven-v5

env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
  /private/liqi_workspace/project_code/audio-driven-v5/compare/.envs/vico_more/bin/python \
  /private/liqi_workspace/project_code/audio-driven-v5/scripts/p2_vico_dim_best10.py controller
```

控制器依次执行每个种子的生成、解码和 EMOCA 提取，最后进行 CPU 评分；使用已有完成标记跳过已完成阶段，`controller.lock` 阻止同目录重复控制器。`status.json` 是状态记录，不是暂停开关；执行上述命令即开始恢复流程。

只使用 node11 物理 GPU7，不停止其他任务。显存门槛为：首次生成 88,000MiB；实际 OOM 后的保留输出重试 110,000MiB；解码 60,000MiB；EMOCA 32,768MiB。显存不足时等待，不代表已经完成生成。

## 6. 查看进度与完成产物

工作目录固定为：

```text
/private/liqi_workspace/project_code/audio-driven-v5/experiments/p2_vico_dim_best10_20261108
```

关注文件：

- `status.json`：当前阶段、种子或等待原因。
- `ACTIVE.json`：最近启动的 worker、PID 和日志位置；PID 可能已退出，需结合状态判断。
- `draws/<seed>/latent_*.log`、`decode_*.log`、`emoca_*.log`：逐阶段日志。
- `draws/<seed>/LATENT_GPU7_COMPLETE.json`、`DECODE_GPU7_COMPLETE.json`、`emoca/EXTRACTION_COMPLETE.json`：各阶段完成标记。
- `scoring.log`：最终评分日志。

全部完成后自动生成以下文件，目前尚不存在：

```text
/private/liqi_workspace/project_code/audio-driven-v5/experiments/p2_vico_dim_best10_20261108/RESULTS.json
/private/liqi_workspace/project_code/audio-driven-v5/experiments/p2_vico_dim_best10_20261108/RESULTS.md
/private/liqi_workspace/project_code/audio-driven-v5/experiments/p2_vico_dim_best10_20261108/COMPLETE.json
```

结果包括 `single_shared_coverage` 和 `best10_shared_coverage`，二者在相同的十候选共同有效帧上评分。该处的单次结果可能与历史单次表略有变化，应保留两份表及覆盖说明。

如所有种子的 EMOCA 特征已完成、仅需重新执行最终 CPU 评分，可使用：

```bash
cd /private/liqi_workspace/project_code/audio-driven-v5

env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
  /private/liqi_workspace/project_code/audio-driven-v5/compare/.envs/vico_more/bin/python \
  /private/liqi_workspace/project_code/audio-driven-v5/scripts/p2_vico_dim_best10.py score
```

`score` 不会补生成或提取缺失特征，也不是查看现有结果的命令。

## 7. 用户指定 Var 的后处理尚未自动接入

现有控制器自动输出的是 **DIM 发布公式 Var**。用户指定的“各系数跨评分帧算方差，再对系数取平均”保存在：

[REQUESTED_VARIANCE_READOUT.json](/private/liqi_workspace/project_code/audio-driven-v5/experiments/p2_vico_dim_best10_20261108/REQUESTED_VARIANCE_READOUT.json)

这份文件记录的是后续要求，当前控制器没有自动读取和执行它。十次结果完成后，还需按 `RESULTS.json` 中每条的 `selection.listener_frame_indices` 和选中 `seed`，对 GT、单次及 best10 精确同覆盖读出逐系数 Var，并另存结果，保留原版 Var。

现有 `/private/liqi_workspace/project_code/audio-driven-v5/scripts/p2_vico_dim_variance_readout.py` 针对旧单次结果结构，不能不经适配就当作十次结果的完整后处理入口。因此，“DIM 原版十次评测入口已准备”不包含“用户指定 Var 的十次自动读出也已接好”。

## 8. 报告结果时的范围

可写：使用指定 EMOCA 和 DIM 发布指标／十次择优规则，在固定 ViCo test43 上补测。

仍需说明：作者原表实际 GT／有效帧及运行分支尚未直接确认；我们从 RGB 视频反提取动作，而 DIM 原生模型直接预测动作系数。现有七方法训练短跑检查点不属于这项评测。
