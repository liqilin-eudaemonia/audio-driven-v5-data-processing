# GLARE：运行与交接

更新：2026-11-11。状态：**当前训练记录已推进至v9；自训练权重短片推理通过，正式ViCo43待测**。本页汇总已验证的本地证据；历史资产核查不代表今天重新验证了外部下载链接。

## 当前训练、权重位置和断点恢复

以[训练运行文档](<../../experiments/glare_paper_settings_node11_gpu7_20261111/README.md>)及其启动回执核对活跃任务。用户给定v8恢复参考现已保存在[v8启动命令](<../../experiments/glare_paper_settings_node11_gpu7_20261111/launch_initial20epochs.json>)：21960步，initial_steps10980，延长段LR5e-5→5e-6。对应延长到30轮前快照为[before_extension30epochs.pt](<../../experiments/glare_paper_settings_node11_gpu7_20261111/before_extension30epochs.pt>)。

文档顶部现为**v9／32940步**，当前命令见[launch.json](<../../experiments/glare_paper_settings_node11_gpu7_20261111/launch.json>)，有previous_steps21960。当前[latest.pt](<../../experiments/glare_paper_settings_node11_gpu7_20261111/latest.pt>)用于当前调度接续，[best.pt](<../../experiments/glare_paper_settings_node11_gpu7_20261111/best.pt>)用于已选最佳模型推理。不能把正在更新的v9 latest与v8调度直接混用；本次仅记录差异，不回退或另起训练。完整恢复保留AdamW、GradScaler、RNG和epoch游标。

## 官方推理命令和输入参数

[官方README Inference](<../glare/README.md>)提供generate.py，主要参数ref_path、aud_path、ckpt_path、res_video_path、lia_path、qwen2audio_path以及各CFG。官方示例（在related/glare运行，路径替换为实际资源）：

```bash
CUDA_VISIBLE_DEVICES=7 python generate.py \
  --ref_path /path/listener.jpg --aud_path /path/speaker.wav \
  --ckpt_path /path/glare_checkpoint.pth --res_video_path /path/output.mp4 \
  --lia_path /path/float.pth --lia_version LIA \
  --qwen2audio_path /path/Qwen2-Audio-7B-Instruct \
  --num_prev_frames 10 --num_ref_frames 125 --wav2vec_sec 10 \
  --a_cfg_scale 2 --e_cfg_scale 1 --r_cfg_scale 1 \
  --s_cfg_scale 1 --ref_cfg_scale 1 --i_cfg_scale 1
```

适配当前自训练checkpoint的实际入口已通过短片：
```bash
CUDA_VISIBLE_DEVICES=7 bash scripts/run_glare_inference.sh \
  --checkpoint experiments/glare_paper_settings_node11_gpu7_20261111/best.pt \
  --reference /path/reference.png --audio /path/speaker.wav \
  --output /path/output.mp4 --seed 137 --nfe 10 --audio-guidance 2
```
[推理命令与验收](<../../experiments/glare_inference_20261111/README.md>)。

## 数据处理、缓存及训练／推理适配

[完整数据/训练交接](<../P2_GLARE_500h复现完整方案与进度交接_20261110.md>)。沿用官方FLOAT latent、ODE、LIA；反应标签按本项目生产配置，官方NULL零强度处理照旧。首帧参考、零历史，音频前640采样处理及N-1输出按适配协议；MP4携带speaker音轨不等于生成listener声音。[常驻推理和缓存](<../../experiments/glare_resident_inference_20261111/README.md>)：模型/Qwen加载一次，音频SHA＋协议缓存，JSONL字段reference/audio/output/seed/nfe。decoder batch8的短测最大差1.96e-5，不声称逐位一致。

## 评测结果与下一步

正式ViCo43尚未完成；训练loss及短片通过不替代最终评测。后续固定具体checkpoint指纹，再整批生成与同表评测。不要为了本页说明停止当前训练。

[返回方法总索引](README.md)。已有结果按各自协议保留，不把短训、辅助权重或仅通过导入检查记为正式方法复现。
