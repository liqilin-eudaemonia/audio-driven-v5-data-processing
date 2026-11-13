# DualTalk：运行与交接

**Seamless完整test6624已接续（2026-11-12）：** 历史双200已完成；全量沿用同模型和指标，由GPU4/GPU7常驻Monitor按容量接续。[全量进度/数值](../../experiments/p2_reference_methods_seamless6624_20261112/RESULTS.md) · [运行与Monitor](../../experiments/p2_reference_methods_seamless6624_20261112/README.md)。

**2026-11-12 双200补测进行中：** 本轮仅历史固定ResponseNet200＋Seamless200。[实时数值表](../../experiments/p2_reference_methods_dualtest_20261112/RESULTS.md) · [本轮模型、输入适配与恢复](../../experiments/p2_reference_methods_dualtest_20261112/README.md)。历史ViCo结果继续保留；进行中不代表双200已完成。

更新：2026-11-11。状态：**官方权重43条原生动作完成；RGB和统一EMOCA待接入**。本页汇总已验证的本地证据；历史资产核查不代表今天重新验证了外部下载链接。

## 当前训练、权重位置和断点恢复

官方主权重：[DualTalk.pth](<../../experiments/p2_vico_additional_baselines_20261107/assets/dualtalk/DualTalk.pth>)；加载核对与demo一致性在[结果目录](<../../experiments/p2_vico_additional_baselines_20261107/dualtalk>)。未启动新的训练；生成回执含checkpoint指纹，缺失条目可恢复。

## 官方推理命令和输入参数

[官方README](<../../experiments/p2_vico_additional_baselines_20261107/vendor/dualtalk/README.md>)，在vendor/dualtalk内运行（--save_path是模型所在目录）：
```bash
python demo.py --audio1_path /path/speaker2.wav \
  --audio2_path /path/speaker1.wav --bs2_path /path/speaker1.npz \
  --save_path /path/model --result_path /path/results
```


## 数据处理、缓存及训练／推理适配

原生25fps expression50＋jaw3＋neck3，适配到30fps；自身音轨零、speaker动作与音频保留。静音编码按长度和权重缓存，校验不匹配回退原路径。[统一推理加速入口](<../../compare/deployment/vico_inference_fast.py>)的dualtalk generate在node11/GPU7运行。
```bash
CUDA_VISIBLE_DEVICES=7 compare/.envs/vico_more/bin/python compare/deployment/vico_inference_fast.py dualtalk generate
```


## 评测结果与下一步

[已完成43条评测及比较表](<../../experiments/p2_vico_full_eval_20261111/README.md>)及[原生输出](<../../experiments/p2_vico_additional_baselines_20261107/dualtalk>)。neck/global映射和RGB renderer尚未确认为统一协议；不将原生56维分数直接填EMOCA行，不补造图像指标。

[返回方法总索引](README.md)。已有结果按各自协议保留，不把短训、辅助权重或仅通过导入检查记为正式方法复现。

**后续ResponseNet完整586已入队：** Seamless完整6624完成后自动接续，使用GPU4/GPU7空闲资源。[完整586进度与结果](../../experiments/p2_reference_methods_responsenet586_20261112/RESULTS.md) · [运行与队列](../../experiments/p2_reference_methods_responsenet586_20261112/README.md)。

**2026-11-13最新资源限制：** Seamless6624及后续ResponseNet586仅使用node11 GPU7；GPU4不可用。历史双200及已完成产物保留。
