# SpeakerVid-5M：运行与交接

更新：2026-11-11。状态：**数据整理/指标工具可用；无已接入对话生成器**。本页汇总已验证的本地证据；历史资产核查不代表今天重新验证了外部下载链接。

## 当前训练、权重位置和断点恢复

官方63个辅助资源已核验，见[扩展方法部署与缺项](<../../compare/deployment/ADDED_PROJECTS.md>)；这些不是论文AR视频生成器checkpoint，当前没有生成模型训练断点。

## 官方推理命令和输入参数

[官方README](<../../compare/SpeakerVid-5M-Code/README.md>)。已验证数据工具示例，从compare目录运行：
```bash
python deployment/run.py speakervid-luminance --total 1 --index 0 \
  --num_proc 1 --video_list_path /path/list.txt --save_dir /path/results
```
这是亮度分析，不是生成命令。

## 数据处理、缓存及训练／推理适配

现有流水线包含场景切分、姿态、ASR、模糊/亮度与FVD特征；环境compare/.envs/speakervid。真实视频32帧FVD特征验证只证明组件可用，不作为模型FVD总分。

## 评测结果与下一步

没有ViCo43生成结果；需要实际生成代码及其权重后才能纳入方法比较。

[返回方法总索引](README.md)。已有结果按各自协议保留，不把短训、辅助权重或仅通过导入检查记为正式方法复现。
