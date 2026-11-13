# ReactFace：运行与交接

更新：2026-11-11。状态：**输入预处理和辅助模型通过；核心模型/渲染资源缺项**。本页汇总已验证的本地证据；历史资产核查不代表今天重新验证了外部下载链接。

## 当前训练、权重位置和断点恢复

官方best_checkpoint.pth未取得；FaceVerse已存在：[faceverse_simple_v2.npy](<../../compare/ReactFace/external/FaceVerse/data/faceverse_simple_v2.npy>)。此前PIRender及reference资源下载受限，具体历史证据见[扩展方法部署与缺项](<../../compare/deployment/ADDED_PROJECTS.md>)。没有正式训练恢复点。

## 官方推理命令和输入参数

[官方README Customized Inference](<../../compare/ReactFace/README.md>)，在该仓库运行：
```bash
python dyadic_reaction_inference.py \
  --speaker-video /path/speaker.mp4 --speaker-audio /path/speaker.wav \
  --listener-portrait /path/portrait.png --window-size 8 --momentum 0.9 \
  --output-dir /path/results --checkpoint /path/best_checkpoint.pth
```


## 数据处理、缓存及训练／推理适配

需要speaker视频、音频及listener人像。现有8帧输入预处理和FaceVerse加载验证不构成完整模型推理。官方renderer还需cur_model_fold.pth及mean_face/std_face/reference_full。环境compare/.envs/reactface，旧部署入口在node8/GPU7。

## 评测结果与下一步

暂无完整生成与ViCo43指标；补齐资源后直接沿用官方自定义推理，不建立额外质量筛选门槛。

[返回方法总索引](README.md)。已有结果按各自协议保留，不把短训、辅助权重或仅通过导入检查记为正式方法复现。
