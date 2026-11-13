# 后处理音频训练读取版本

基于`p2_feature_v1`的训练、检查点恢复、text-attention LoRA、TensorBoard和共享预检查，补齐后处理数据读取。**当前数据版本只允许关闭Feature decoder的E0（AV）/E1（AV＋Event）**；未重新提取后处理音频teacher，不允许把旧teacher接到新音频。

入口：`releases/p2_audio_postprocess_v1/scripts/p2_listen.py`。使用后处理视图的`config.json`；不要使用旧Feature训练版本读取这些视图。

```bash
P2_PY=/private/liqi_workspace/project_code/Omni-LTX-stage2/.venv/bin/python
# 仅核对配置，不训练
"$P2_PY" releases/p2_audio_postprocess_v1/scripts/p2_listen.py audit \
  --config experiments/p2_train50h_audio_postprocess_20261026/view/config.json --suite config
```

正式训练沿用`train --config ... --arm E0|E1 --train-seed ...`以及`--resume ...`参数。多机共享预检查需每次启动提供同一个全新的`P2_PREFLIGHT_SESSION`。训练budget/batch/LoRA配置仍由所选训练config决定，本次未替用户决定下一轮训练配置或启动训练。

读取规则：

1. 新音轨：原input/text缓存＋原video latent＋新audio latent/support＋新events。
2. `reuse_base_audio=true`：原input/text/video/audio latent＋**新events**，不能回退至旧事件标签。
3. 缺失或不匹配的配方、音频/事件缓存直接报错；不重新读取波形执行静音或VAE，也不回退原音轨。
4. 共享预检查登记policy、逐条recipe、新audio complete/payload、新events以及真实base VAE路径。只由rank0执行完整检查，复用时检查依赖变化；训练实际读取仍校验载荷SHA。
5. checkpoint/source identity包含实际训练读取与后处理实现及兼容源码，恢复时版本变化不能被悄悄忽略。

## 已发布50h缓存的严格兼容

50h缓存生产者的`overlay.py` SHA为`6e970966...`。后来全量候选缓存版本`2a152ab6...`只增加`require_admitted=True`及显式cache-only可选参数；训练准入默认保持严格。直接用当前主目录旧入口读取50h会因为producer代码哈希不同被拒绝。

本版本保存已发布50h源码`compat/overlay_train50h.py`。只接受上述两个明确版本；历史兼容要求当前源码与历史副本SHA均准确、完整缓存contract仅该overlay代码SHA不同。配方、编码器、base VAE、来源、policy和payload SHA均继续严格校验，缓存文件不改写。其他任何差异均拒绝。全量新缓存按其原身份正常读取。

## 数据范围与准入

- 原50h视图：30,415段，原训练范围不重选。
- 全量处理视图：722,706段都已缓存，但原有960段未准入属性仍保留。**全量缓存清单不能直接冒充全部可训练清单**；正式训练视图需显式保留721,746段原已准入样本，或使用已有的已准入子集。此入口不偷偷跳过、不放宽准入，读到未准入标签会报错。
- 全量`view/config.json`是数据入口模板，当前batch/预算与LoRA字段不等同于前轮50h实验设置；新训练需使用明确冻结的实验config。

## 验证

`experiments/p2_audio_training_reader_20261027/`：26项CPU测试；两种数据源各16条新音频＋16条复用音频，共64条真实读取；未准入样本拒绝；每种视图32条完整预检查及重复复用；另8条真实CUDA读取与CPU latent/事件逐元素一致。未加载生成/训练模型，未启动训练，未重建缓存。
