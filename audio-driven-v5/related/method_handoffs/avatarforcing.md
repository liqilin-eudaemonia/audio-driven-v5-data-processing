# Avatar Forcing：运行与交接

**Seamless完整test6624已接续（2026-11-12）：** 历史双200已完成；全量沿用同模型和指标，由GPU4/GPU7常驻Monitor按容量接续。[全量进度/数值](../../experiments/p2_reference_methods_seamless6624_20261112/RESULTS.md) · [运行与Monitor](../../experiments/p2_reference_methods_seamless6624_20261112/README.md)。

**2026-11-12 双200补测进行中：** 本轮仅历史固定ResponseNet200＋Seamless200。[实时数值表](../../experiments/p2_reference_methods_dualtest_20261112/RESULTS.md) · [本轮模型、输入适配与恢复](../../experiments/p2_reference_methods_dualtest_20261112/README.md)。历史ViCo结果继续保留；进行中不代表双200已完成。

更新：2026-11-11。状态：**官方权重43条结果；统一动作与五项图像重测完成**。本页汇总已验证的本地证据；历史资产核查不代表今天重新验证了外部下载链接。

## 当前训练、权重位置和断点恢复

官方生成资产：[flow_transformer.pth](<../../compare/AvatarForcing/pretrained_dir/flow_transformer.pth>)、[motion_autoencoder.pth](<../../compare/AvatarForcing/pretrained_dir/motion_autoencoder.pth>)、[wav2vec2](<../../compare/AvatarForcing/pretrained_dir/wav2vec2-base-960h>)。按实际权重加载审计和SHA识别，不凭文件名推断内容。另有[真实数据短训及恢复验收](<../../experiments/baseline_training_realdata_20261109/README.md>)，每阶段仅4次更新，用于接通训练和冷恢复，不作为论文成绩。本地诊断检查点：[finetune/checkpoint.pt](<../../experiments/baseline_training_realdata_20261109/runs/avatarforcing/finetune/checkpoint.pt>)。 本轮采用官方权重；43条生成回执在[历史生成目录](<../../experiments/p2_vico43_unified_20261107/avatarforcing/generated>)，有效条目无需重跑。

## 官方推理命令和输入参数

[官方README Inference](<../../compare/AvatarForcing/README.md>)；在该仓库运行，四路条件必须明确：
```bash
CUDA_VISIBLE_DEVICES=7 python inference.py \
  --input_image_path data/rumi.jpg --avatar_audio_path data/avatar.wav \
  --user_audio_path data/user.wav --user_video_path data/user/ \
  --a_cfg_scale 2 --u_cfg_scale 1 --nfe 10
```


## 数据处理、缓存及训练／推理适配

共同输入适配由[vico_avatarforcing.py](<../../compare/deployment/vico_avatarforcing.py>)实现：listener首帧、avatar_audio零波形、真实speaker音轨、user_visual零latent，不读未来listener或GT listener声音。seed137、NFE10、CFG a2/u1、KV缓存；完整音频绕过CLI的30秒上限。原生512/25→256/30最近时间戳与bicubic处理。历史共同输入分支使用兼容Torch环境；严格沿用官方模型，weight_norm仅做状态名兼容。

[本轮输入/权重/特征核验](<../../experiments/p2_vico_avatarforcing_added_20261111/INPUT_AUDIT.json>)验证43条及当前共同评分源。图像评分batch64，CPBD按片CPU并发；评分回执可断点继续。

## 评测结果与下一步

本轮输出目录：[AF追加评测及数值表](<../../experiments/p2_vico_avatarforcing_added_20261111/README.md>)；五项图像：PSNR16.183142、SSIM0.572531、FID39.140994、CSIM0.466332、CPBD0.190193；FD Exp37.633272、Pose0.068397。图像16,288帧，动作共同16,238帧。[九行统一动作表](<../../experiments/p2_vico_avatarforcing_added_20261111/motion_metrics.csv>)。论文Table4采用另一套输入与SPECTRE口径，不能直接把本轮EMOCA数值称为Table4复现。真实speaker视觉分支的后续状态见[完整复现过程](<../../experiments/p2_avatarforcing_table4_full_reproduction_20261110/README.md>)。最新音源分离已完成43对/86路并停止，尚未据此重新生成或评分：[V5e状态](<../../experiments/p2_avatarforcing_table4_audio_preparation_cpu_20261110_v5e/README.md>)。本次没有自动续跑这条独立实验。

[返回方法总索引](README.md)。已有结果按各自协议保留，不把短训、辅助权重或仅通过导入检查记为正式方法复现。

**后续ResponseNet完整586已入队：** Seamless完整6624完成后自动接续，使用GPU4/GPU7空闲资源。[完整586进度与结果](../../experiments/p2_reference_methods_responsenet586_20261112/RESULTS.md) · [运行与队列](../../experiments/p2_reference_methods_responsenet586_20261112/README.md)。

**2026-11-13最新资源限制：** Seamless6624及后续ResponseNet586仅使用node11 GPU7；GPU4不可用。历史双200及已完成产物保留。
