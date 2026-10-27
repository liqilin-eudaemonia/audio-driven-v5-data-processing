# 四组正式训练代码的固定备份

固定标签：`p2-fourway-actual-runs-20261027`。对应已完成的AV、AV＋Event、AV＋Feature、AV＋Event＋Feature四组1,268步实验，不覆盖旧备份。

| arm | 实验 | 实际版本 | 训练源码绑定 | 副本CPU测试 |
|---|---|---|---:|---:|
| E0 | AV | p2_preflight_v1 | 73/73 | 167通过 |
| E1 | AV＋Event | 训练时主版本 | 70/70 | 146通过 |
| E0F | AV＋0.1×视频Feature＋0.1×音频Feature | p2_feature_v1，arm=E0 | 73/73 | 184通过 |
| E1F | AV＋0.05×Event＋0.1×视频Feature＋0.1×音频Feature | p2_feature_v1，arm=E1 | 73/73 | 184通过 |

四组均训练text-attention LoRA，其梯度来自AV；E0/E0F事件头为detach(U)的probe。共同训练seed为20261023、global batch=24、steps=1268、save_every=100；Feature分支两路权重各0.1。使用旧30,415段50h数据，不是后来重新筛选的后处理50h。

## 备份布局

- `EXPERIMENTS.json`：四组实际run、checkpoint引用、CLI入口。
- `arms/<arm>/snapshot/`：各自独立的audio-driven-v5、audio-driven-v4及Omni-LTX-stage2源码、测试、配置和历史说明。
- `arms/<arm>/evidence/run.json`、`config.json`：原运行记录，未改写。
- `arms/<arm>/evidence/source_bindings.json`：每个训练源码的原路径、SHA256、快照路径和取回来源。
- `arms/<arm>/evidence/step_001268.complete.json`、`checkpoint_reference.json`：完成标记及原机器权重位置，不包含权重本身。
- `evidence/prior_backup_audit.json`：原备份E0缺5个、两组Feature各缺8个源码内容；本次全部补齐。
- `evidence/external_source_check.json`：223个LTX源码/元数据文件与此前固定快照及当前工作树一致。
- `validation/<arm>/cpu_tests.json`：测试结果与实际导入路径。
- `docs/`：四组索引及下一轮显式规划讨论草案，草案不代表已批准实施。

旧主目录部分文件已改动，本备份按run.json指纹选择训练时字节。四个独立目录避免不同历史版本同路径覆盖。原文档中关于“尚未训练”等语句是历史记录，最终run身份以evidence和EXPERIMENTS为准。

## 本次验证

四组共289条源码绑定全部匹配，CPU测试共681次通过。每组导入均来自自己的快照，而非原工作区。每个Feature快照显式排除3项需要原机器发布状态或集群启动配置的检查；本次不重新执行训练、部署或真实模型推理。原GPU验证证据保留在snapshot的实验目录中，不冒充本次重新运行的结果。

```bash
python verify_backup.py

# 使用已有匹配依赖环境，不需要GPU或预训练模型。
P2_PYTHON=/private/liqi_workspace/project_code/Omni-LTX-stage2/.venv/bin/python
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 "$P2_PYTHON" cpu_check.py E0
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 "$P2_PYTHON" cpu_check.py E1
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 "$P2_PYTHON" cpu_check.py E0F
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 "$P2_PYTHON" cpu_check.py E1F
```

`BACKUP_MANIFEST.json`列出快照、证据、说明和工具的文件哈希；验证日志单独保留，不因重复运行日志改变而使源码验证失效。

## 恢复限制

没有上传模型权重、checkpoint二进制、媒体、特征缓存、TensorBoard event二进制、虚拟环境或凭据。它们仍在原机器；不能把代码备份描述为全部训练资产的异地备份。

保留每个arm内的三个同级目录，并安装/使用匹配依赖。历史依赖说明来自先前LoRA备份；本次运行环境另存`validation/environment.json`。上游lock文件不是自动保证兼容的完整环境镜像。

原配置包含原机器绝对路径。CPU测试用配置只将reuse_root改成相对路径，逐项记录于`portable_config_changes.json`；原配置和run.json保持不变。严格续训需要恢复原路径布局及匹配的配置、数据、缓存、资产和rank数，不允许绕过源码/资产身份校验。备份副本能导入和通过CPU检查，不代表可在任意新路径直接续训。

新音频后处理训练是另一版本，旧音频teacher不能直接用于新音频目标。本备份用于保存已完成四组实验，不自动适配下一轮显式规划训练。
