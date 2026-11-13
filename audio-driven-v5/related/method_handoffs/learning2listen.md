# Learning2Listen（L2L，CVPR 2022）：运行与交接

**Seamless完整test6624已接续（2026-11-12）：** 历史双200已完成；全量沿用同模型和指标，由GPU4/GPU7常驻Monitor按容量接续。[全量进度/数值](../../experiments/p2_reference_methods_seamless6624_20261112/RESULTS.md) · [运行与Monitor](../../experiments/p2_reference_methods_seamless6624_20261112/README.md)。

**2026-11-12 双200补测进行中：** 本轮仅历史固定ResponseNet200＋Seamless200。[实时数值表](../../experiments/p2_reference_methods_dualtest_20261112/RESULTS.md) · [本轮模型、输入适配与恢复](../../experiments/p2_reference_methods_dualtest_20261112/README.md)。历史ViCo结果继续保留；进行中不代表双200已完成。

更新：2026-11-11。状态：**四套官方人物权重各43条动作及GPAvatar视频已完成**。本页汇总已验证的本地证据；历史资产核查不代表今天重新验证了外部下载链接。

## 当前训练、权重位置和断点恢复

原生VQ、predictor及统计必须配对；四套权重属于同一方法。

- conan：[delta_v6_er2er_best.pth](<../../compare/learning2listen/src/models/delta_v6_er2er_best.pth>)、[l2_32_smoothSS_er2er_best.pth](<../../compare/learning2listen/src/vqgan/models/l2_32_smoothSS_er2er_best.pth>)、[delta_v6_er2er_preprocess_core.npz](<../../compare/learning2listen/src/models/delta_v6_er2er_preprocess_core.npz>)、[l2_32_smoothSS_er2er_preprocess_core.npz](<../../compare/learning2listen/src/vqgan/models/l2_32_smoothSS_er2er_preprocess_core.npz>)。
- fallon：[fallon_delta_v6_er2er_best.pth](<../../compare/learning2listen/src/models/fallon_delta_v6_er2er_best.pth>)、[fallon_er2er_best.pth](<../../compare/learning2listen/src/vqgan/models/person_specific/fallon_er2er_best.pth>)、[fallon_delta_v6_er2er_preprocess_core.npz](<../../compare/learning2listen/src/models/fallon_delta_v6_er2er_preprocess_core.npz>)、[fallon_er2er_preprocess_core.npz](<../../compare/learning2listen/src/vqgan/models/person_specific/fallon_er2er_preprocess_core.npz>)。
- stephen：[stephen_delta_v6_er2er_best.pth](<../../compare/learning2listen/src/models/stephen_delta_v6_er2er_best.pth>)、[stephen_er2er_best.pth](<../../compare/learning2listen/src/vqgan/models/person_specific/stephen_er2er_best.pth>)、[stephen_delta_v6_er2er_preprocess_core.npz](<../../compare/learning2listen/src/models/stephen_delta_v6_er2er_preprocess_core.npz>)、[stephen_er2er_preprocess_core.npz](<../../compare/learning2listen/src/vqgan/models/person_specific/stephen_er2er_preprocess_core.npz>)。
- trevor：[trevor_delta_v6_er2er_best.pth](<../../compare/learning2listen/src/models/trevor_delta_v6_er2er_best.pth>)、[trevor_er2er_best.pth](<../../compare/learning2listen/src/vqgan/models/person_specific/trevor_er2er_best.pth>)、[trevor_delta_v6_er2er_preprocess_core.npz](<../../compare/learning2listen/src/models/trevor_delta_v6_er2er_preprocess_core.npz>)、[trevor_er2er_preprocess_core.npz](<../../compare/learning2listen/src/vqgan/models/person_specific/trevor_er2er_preprocess_core.npz>)。

另有[真实数据短训及恢复验收](<../../experiments/baseline_training_realdata_20261109/README.md>)，每阶段仅4次更新，用于接通训练和冷恢复，不作为论文成绩。本地诊断检查点：[codec/checkpoint.pt](<../../experiments/baseline_training_realdata_20261109/runs/l2l/codec/checkpoint.pt>)、[predictor/checkpoint.pt](<../../experiments/baseline_training_realdata_20261109/runs/l2l/predictor/checkpoint.pt>)。 原生ViCo生成按人物的43条整批恢复：先验证权重与输出SHA；某人物缺损时重算该人物整批，不能只重跑一条而改变随机采样序列。其他完整人物批次可跳过。

## 官方推理命令和输入参数

[官方README](<../../compare/learning2listen/README.md>)。在 `compare/learning2listen/src` 运行；输入为预处理DECA动作和128维mel，不是直接任意WAV。
```bash
python test_vq_decoder.py --config configs/vq/delta_v6.json \
  --checkpoint models/delta_v6_er2er_best.pth --speaker conan
```
其余人物使用各自 `{fallon,stephen,trevor}_delta_v6.json` 和对应predictor。

## 数据处理、缓存及训练／推理适配

30fps、expression50＋pose6，额外使用speaker视频动作；mel按4t取片，标准化零历史和尾部补齐。[统一推理加速入口](<../../compare/deployment/vico_inference_fast.py>)先检查缓存再加载模型，已完成结果无需GPU。L2L新生成仍带node8/GPU7约束，不应直接作为node11启动命令。[GPAvatar渲染接入](<../../experiments/p2_l2l_public_renderer_20261110/README.md>)使用外部公开渲染器，缓存和模型常驻减少重复加载；与原生动作结果分别报告。

## 评测结果与下一步

[已完成43条评测及比较表](<../../experiments/p2_vico_full_eval_20261111/README.md>)。四个人物均报告，不能按GT挑最优。L2L＋GPAvatar不是官方原写实renderer，也不是DIM论文中的ViCo重训L2L。

[返回方法总索引](README.md)。已有结果按各自协议保留，不把短训、辅助权重或仅通过导入检查记为正式方法复现。

**后续ResponseNet完整586已入队：** Seamless完整6624完成后自动接续，使用GPU4/GPU7空闲资源。[完整586进度与结果](../../experiments/p2_reference_methods_responsenet586_20261112/RESULTS.md) · [运行与队列](../../experiments/p2_reference_methods_responsenet586_20261112/README.md)。

**2026-11-13最新资源限制：** Seamless6624及后续ResponseNet586仅使用node11 GPU7；GPU4不可用。历史双200及已完成产物保留。
